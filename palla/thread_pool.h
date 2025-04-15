#pragma once

#include <span>
#include <vector>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <atomic>
#include <cassert>

namespace palla {
    namespace details {
        namespace thread_pool_namespace {


            // Hack while waiting for std::move_only_function.

            // Generic template so we can specialize it.
            template<class T>
            class move_only_function;
            
            // Partial specialization for function types.
            template<class R, class... Ts>
            class move_only_function<R(Ts...)> {
            private:
                // Private types.

                // A virtual interface for any callable function with signature R(Ts...).
                struct callable {
                    virtual R call(Ts...) = 0;
                    virtual ~callable() {};
                };

                // A concrete callable function given by type T.
                template<class T>
                struct callable_impl : callable {
                    callable_impl(T func) : m_func(std::move(func)) {}
                    virtual R call(Ts... args) override { m_func(args...); }
                    T m_func;
                };

                // Private members.
                std::unique_ptr<callable> m_ptr;    // Holds the function. We could use SBO to optimize the case when sizeof(T) <= sizeof(unique_ptr), but this seems hard.

            public:
                // Public functions. Basically the same api as std::function.

                // Constructors.
                move_only_function() = default;

                template<class T>
                move_only_function(T func) {
                    m_ptr = std::make_unique<callable_impl<T>>(std::move(func));
                }

                // Calls the function.
                R operator()(Ts... args) { return m_ptr->call(args...); }

                // Returns whether there this owns a function.
                operator bool() const { return m_ptr != nullptr; }
            };


            // Singleton CRTP base class.
            // The singleton can be accessed by T::get().
            template<class T>
            class singleton {
            public:
                static T& get() {
                    static T instance;
                    return instance;
                }

                singleton(singleton const&) = delete;
                singleton(singleton&&) = delete;
                singleton& operator=(singleton const&) = delete;
                singleton& operator=(singleton&&) = delete;

            protected:
                singleton() = default;
            };


            // Returns an async function with signature void() and its future which will be filled once the function completes.
            // The function arguments are copied, not moved.
            template<class F, class... Ts, class R = std::decay_t<decltype(std::declval<F>()(std::declval<Ts>()...))>>
            [[nodiscard]] auto make_async_func(F&& func, Ts... args) {
                struct {
                    move_only_function<void()> async_func;
                    std::future<R> future;
                } result;

                std::promise<R> promise;
                result.future = promise.get_future();
                result.async_func = [f = std::move(func), p = std::move(promise), args...]() mutable {
                    if constexpr (!std::is_void_v<R>) {
                        p.set_value(f(args...));
                    }
                    else {
                        f(args...);
                        p.set_value();
                    }
                };
                return result;
            }


            // A thread that runs in an infinite loop until it is destroyed.
            class worker_thread {
            private:
                // Private types.

                // The state of the thread.
                enum class state {
                    initializing,                       // Intial state until the thread begins. The constructor has not yet finished.
                    available,                          // State when the thread is waiting for work.
                    working,                            // State when the thread is working on a user function.
                    exiting                             // State when terminate() has been called.
                };

                // Private members.
                std::thread m_thread;                   // The thread itself.                   
                std::mutex m_mutex;                     // Used to lock the thread to give it work.
                std::condition_variable m_wake_up;      // Used to wake up the thread to give it work.
                move_only_function<void()> m_func;      // The user function to execute.
                state m_state = state::initializing;    // The state we are in.

                // Private functions.

                // Infinite loop.
                void loop() {
                    std::unique_lock lock(m_mutex);
                    while (true) {
                        // Change the state to available and notify the caller.
                        assert(m_state == state::working || m_state == state::initializing);
                        m_state = state::available;
                        m_wake_up.notify_all();

                        // Wait until the state changes.
                        m_wake_up.wait(lock, [this] { return m_state != state::available; });

                        // If we need to exit, do it now.
                        if (m_state == state::exiting)
                            return;

                        // Call the function.
                        assert(m_state == state::working);
                        m_func();
                    }
                }

            public:
                // Public functions.

                // Constructor.
                worker_thread() : m_thread(&worker_thread::loop, this) {
                    // Wait until the thread become available.
                    std::unique_lock lock(m_mutex);
                    m_wake_up.wait(lock, [this]() { return m_state == state::available; });
                }

                // Destructor.
                ~worker_thread() {
                    // Terminate then wait for the thread to end before destroying the data.
                    terminate();
                    m_thread.join();
                }

                // Returns true if we are inside this thread.
                [[nodiscard]] bool is_inside() const { return std::this_thread::get_id() == m_thread.get_id(); }

                // Send a function to this thread.
                void give_work(move_only_function<void()> func) {
                    // Wait until the thread become available.
                    std::unique_lock lock(m_mutex);
                    m_wake_up.wait(lock, [this]() { return m_state == state::available; });

                    // Give the function to the thread.
                    m_state = state::working;
                    m_func = std::move(func);

                    // Wake up the thread.
                    lock.unlock();
                    m_wake_up.notify_all();
                }

                // Intentional termination function.
                void terminate() {
                    // Wait until the thread is not working.
                    std::unique_lock lock(m_mutex);
                    m_wake_up.wait(lock, [this]() { return m_state == state::available || m_state == state::exiting; });

                    // Tell the thread to exit.
                    m_state = state::exiting;

                    // Wake up the thread.
                    lock.unlock();
                    m_wake_up.notify_all();
                }
            };


            // A pool of worker threads.
            class thread_pool : public singleton<thread_pool> {
            private:
                // Private members.
                std::vector<std::unique_ptr<worker_thread>> m_all_workers;  // All the workers, be they owned by this or by sub pools.
                std::vector<worker_thread*> m_available_workers;            // The workers that are owned by this.
                size_t m_desired_size = 0;                                  // The number of workers we are striving forward (required in case a sub pool owns the workers).
                size_t m_prev_size = 0;                                     // The previous size before disable() was called. The pool will be resized to this when enable() is called.
                mutable std::shared_mutex m_mutex;                          // A mutex for read/write locks.

                // Private functions.
                // Non thread-safe functions are marked by the suffix _unsafe.

                // Obtains a read lock for const functions, or a write lock for non-const functions.
                [[nodiscard]] auto get_lock() const { return std::shared_lock(m_mutex); }
                [[nodiscard]] auto get_lock() { return std::unique_lock(m_mutex); }

                // Returns a pointer to this worker thread if we are inside a worker.
                // If we are not in a worker, returns nullptr.
                [[nodiscard]] worker_thread* get_self_worker_unsafe() const {
                    auto it = std::find_if(m_all_workers.begin(), m_all_workers.end(), [](const std::unique_ptr<worker_thread>& worker) { return worker->is_inside(); });
                    return it == m_all_workers.end() ? nullptr : it->get();
                }

                // Takes back ownership of workers. If this would lead to too many workers, destroy them.
                void receive_workers(std::span<worker_thread*> workers) {
                    auto lock = get_lock();
                    receive_workers_unsafe(workers);
                }
                void receive_workers_unsafe(std::span<worker_thread*> workers) {
                    // Remove workers if we want to downsize.
                    for (auto worker : workers) {
                        // Check that the worker is not the current thread, because then a parent sub_pool owns it.
                        if (worker->is_inside())
                            continue;

                        // Either add the worker to the pool or destroy it.
                        if (m_desired_size == m_all_workers.size()) {
                            // Add this worker back to the pool.
                            m_available_workers.push_back(worker);
                        }
                        else {
                            // Cast it into the fire, destroy it!
                            std::swap(*std::find_if(m_all_workers.begin(), m_all_workers.end(), [worker](const std::unique_ptr<worker_thread>& other_worker) {
                                return other_worker.get() == worker;
                            }), m_all_workers.back());
                            m_all_workers.pop_back();
                        }
                    }
                }

                // Give ownership of workers away. If the current thread is one of the workers, always give it first.
                [[nodiscard]] std::vector<worker_thread*> lend_workers(size_t nb_workers) {
                    auto lock = get_lock();
                    return lend_workers_unsafe(nb_workers);
                }
                [[nodiscard]] std::vector<worker_thread*> lend_workers_unsafe(size_t nb_workers) {
                    if (nb_workers == 0)
                        return {};

                    // If we are inside a worker, add it.
                    std::vector<worker_thread*> workers;
                    if (auto self = get_self_worker_unsafe())
                        workers.push_back(self);

                    // If we need more, add more.
                    while (!m_available_workers.empty() && workers.size() < nb_workers) {
                        workers.push_back(m_available_workers.back());
                        m_available_workers.pop_back();
                    }

                    return workers;
                }

                // Change the number of workers.
                void resize_unsafe(size_t nb_workers) {
                    // Save the size.
                    if (m_desired_size > 0)
                        m_prev_size = m_desired_size;
                    m_desired_size = nb_workers;

                    // Add new workers if we need more.
                    while (m_desired_size > m_all_workers.size()) {
                        m_available_workers.push_back(m_all_workers.emplace_back(std::make_unique<worker_thread>()).get());
                    }

                    // Remove existing workers if we need less by pretending to lend them all out and immediately retaking them.
                    // Apparently you cant convert vector&& to span. Wild.
                    auto workers = lend_workers_unsafe(std::numeric_limits<size_t>::max());
                    receive_workers_unsafe(workers);
                }

            public:
                // Public types.

                // A sub pool which holds workers temporarily until it is either destroyed or releases them, after which they return to the main pool.
                class sub_pool {
                    friend class thread_pool; // So the main pool can access the private constructor.
                private:
                    // Private members.
                    std::vector<worker_thread*> m_workers;
                    size_t m_desired_size = 0;

#ifdef _DEBUG       // Used to ensure that this object is never shared across threads.
                    std::thread::id m_thread_id;
                    bool is_same_thread() { return std::this_thread::get_id() == m_thread_id; }
#endif
                    // Private constructor used by thread_pool.
                    sub_pool(size_t nb_desired) {
#ifdef _DEBUG           // Get the current thread in debug only.
                        m_thread_id = std::this_thread::get_id();
#endif                  // Other members.
                        m_desired_size = nb_desired;
                        m_workers = thread_pool::get().lend_workers(nb_desired);
                    }

                public:
                    // sub_pool is default constructible and movable, but not copyable.
                    sub_pool() = default;

                    sub_pool(const sub_pool&) = delete;
                    sub_pool& operator=(const sub_pool&) = delete;
                    sub_pool(sub_pool&&) = default;
                    sub_pool& operator=(sub_pool&&) = default;

                    // Destructor. Releases all the workers back to the main pool.
                    ~sub_pool() { release(); }

                    // Releases all the workers back to the main pool.
                    void release() {
                        thread_pool::get().receive_workers(m_workers);
                        m_workers.clear();
                        m_desired_size = 0;
                    }

                    // Various getters.
                    [[nodiscard]] size_t size() const { return m_workers.size(); }
                    [[nodiscard]] size_t desired_size() const { return m_desired_size; }
                    [[nodiscard]] bool empty() const { return size() == 0; }
                    [[nodiscard]] bool full() const { return size() == desired_size(); }

                    // Dispatchers.

                    // Dispatches a function func(int index) -> R to size() threads, where index is the thread index from 0 to size() - 1, and R is the return value.
                    // Note that if size() == 0, func() will not be called at all.
                    // Waits for all threads to complete, then returns a vector of R, one for each thread. If the return type is void, we still wait but return nothing.
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_reserved(F&& func) {
                        assert(is_same_thread()); // sub_thread_pool cannot be shared across threads!

                        // Dispatch.
                        move_only_function<void()> self_func;
                        std::vector<std::future<R>> futures(m_workers.size());
                        for (size_t i = 0; i < size(); i++) {
                            auto [async_func, future] = make_async_func(func, (int)i);
                            futures[i] = std::move(future);
                            if (m_workers[i]->is_inside()) {
                                self_func = std::move(async_func);
                            }
                            else {
                                m_workers[i]->give_work(std::move(async_func));
                            }
                        }

                        // If we are one of the workers, call the function ourselves.
                        if (self_func) {
                            self_func();
                        }

                        // Wait for the results.
                        if constexpr (std::is_void_v<R>) {
                            for (size_t i = 0; i < size(); i++)
                                futures[i].get();
                        }
                        else {
                            std::vector<R> results(size());
                            for (size_t i = 0; i < size(); i++)
                                results[i] = futures[i].get();
                            return results;
                        }
                    }

                    // Dispatches a function func(int index) -> R to size() threads, where index is the thread index from 0 to size() - 1, and R is the return value.
                    // Note that if size() == 0, func(0) will be called synchronously.
                    // Waits for all threads to complete, then returns a vector of R, one for each thread. If the return type is void, we still wait but return nothing.
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_at_least_one(F&& func) {
                        assert(is_same_thread()); // sub_thread_pool cannot be shared across threads!

                        if (!empty()) {
                            return dispatch_to_reserved(func);
                        }
                        else {
                            if constexpr (std::is_void_v<R>)
                                func(0);
                            else
                                return std::vector<R> { func(0) };
                        }
                    }

                    // Dispatches a function func(int index) -> R to desired_size() threads, where index is the thread index from 0 to desired_size() - 1, and R is the return value.
                    // If size() < desired_size(), the threads will simply queue up the work.
                    // Waits for all threads to complete, then returns a vector of R, one for each thread. If the return type is void, we still wait but return nothing.
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_all(F&& func) {
                        assert(is_same_thread()); // sub_thread_pool cannot be shared across threads!

                        if (full()) {
                            // No need for any fancy shenanigans. 
                            return dispatch_to_reserved(func);
                        }
                        else {
                            // Wrap func inside a loop with a shared index.
                            std::atomic<int> shared_index{};
                            std::conditional_t<std::is_void_v<R>, char, std::vector<R>> results;
                            if constexpr (!std::is_void_v<R>)
                                results.resize(m_desired_size);

                            auto full_func = [&](size_t) {
                                for (int i = shared_index++; i < (int)m_desired_size; i = shared_index++) {
                                    if constexpr (std::is_void_v<R>)
                                        func(i);
                                    else
                                        results[i] = func(i);
                                }
                            };

                            // Call the wrapped function at least once.
                            dispatch_to_at_least_one(full_func);
                            if constexpr (!std::is_void_v<R>)
                                return results;
                        }
                    }
                };

                // Public functions.
                // Every function should be thread-safe and recursion-safe.

                // Constructor and destructor. By default the pool starts with every logical core on the system.
                thread_pool() { resize(std::thread::hardware_concurrency()); }
                ~thread_pool() { disable(); }

                // Disables the pool. This prevents it from lending new workers to sub pools.
                // Workers owned by sub pools will be destroyed once they return back to this.
                void disable() { resize(0); }

                // Reenables the pool to the previous number of workers from before disable() or resize(0) was called.
                void enable() {
                    auto lock = get_lock();
                    if (m_desired_size == 0)
                        resize_unsafe(m_prev_size);
                }

                // Adds or removes workers.
                // The pool keeps track of how many workers it has lended to sub pools.
                // When resizing down, available workers are immediately deleted, and workers owned by sub pools will be deleted once they are returned.
                // When resizing up, new workers are created until the total number of workers (owned by this or by sub pools) reaches the desired number.
                // By default, the pool has one worker per logical core on the system.
                void resize(size_t nb_threads) {
                    auto lock = get_lock();
                    resize_unsafe(nb_threads);
                }

                // Returns the total number of workers, both owned by this or by sub pools.
                [[nodiscard]] size_t size() const {
                    auto lock = get_lock();
                    return m_all_workers.size();
                }

                // Returns whether the pool contains no workers.
                [[nodiscard]] bool empty() const { return size() == 0; }

                // Returns the number of available workers (owned by this, not a sub pool).
                // Note that if we are inside a worker, it will always be available.
                [[nodiscard]] size_t nb_available() const {
                    auto lock = get_lock();
                    return m_available_workers.size() + (get_self_worker_unsafe() != nullptr);
                }

                // Returns the number of workers owned by sub pools.
                [[nodiscard]] size_t nb_working() const {
                    auto lock = get_lock();
                    return m_all_workers.size() - m_available_workers.size();
                }

                // Returns whether we are inside a worker.
                [[nodiscard]] bool is_worker() const {
                    auto lock = get_lock();
                    return get_self_worker_unsafe() != nullptr;
                }

                // Returns a sub pool containing up to nb_desired workers.
                [[nodiscard]] sub_pool reserve(size_t nb_desired) { return sub_pool(nb_desired); }
            };


        } // namespace thread_pool_namespace
    } // namespace details


    // Exports.
    using details::thread_pool_namespace::thread_pool;


} // namespace palla