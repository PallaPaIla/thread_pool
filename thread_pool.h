#pragma once

#include <vector>
#include <optional>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>
#include <cassert>

namespace palla {
    namespace details {
        namespace thread_pool_namespace {



            // Hack while waiting for std::move_only_function.
            template<class T>
            struct move_only_function;

            template<class R, class... Ts>
            struct move_only_function<R(Ts...)> {
                struct callable {
                    virtual R call(Ts...) = 0;
                    virtual ~callable() {}
                };

                template<class T>
                struct callable_impl : callable {
                    callable_impl(T func) : m_func(std::move(func)) {}
                    virtual R call(Ts... args) override { m_func(args...); }
                    T m_func;
                };

                move_only_function() = default;

                template<class T>
                move_only_function(T func) {
                    m_ptr = std::make_unique<callable_impl<T>>(std::move(func));
                }

                R operator()(Ts... args) { return m_ptr->call(args...); }

                operator bool() const { return m_ptr != nullptr; }

                std::unique_ptr<callable> m_ptr;
            };



            // Singleton CRTP base class.
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



            // Returns an async function and its future.
            template<class F, class... Ts, class R = std::decay_t<decltype(std::declval<F>()(std::declval<Ts>()...))>>
            auto make_async_func(F&& func, Ts... args) {

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



            // Various synchronization objects per thread in the pool.
            struct thread_data {
                std::thread thread;
                std::mutex mutex;
                std::condition_variable wake_up;
                move_only_function<void()> func;

                // Should only be called once on thread_pool construction.
                thread_data() = default;
                thread_data(thread_data&&) noexcept {}
                thread_data& operator=(thread_data&&) noexcept { return *this; }
            };



            // Holds a single thread while waiting to dispatch a function.
            // The thread automatically returns the the pool upon dispatch(), or on destruction of this object.
            class thread_dispatcher {
                friend class thread_pool; // So thread_pool can access the contructor.

            public:
                thread_dispatcher(const thread_dispatcher&) = delete;
                thread_dispatcher& operator=(const thread_dispatcher&) = delete;
                thread_dispatcher(thread_dispatcher&&) = default;
                thread_dispatcher& operator=(thread_dispatcher&&) = default;
                ~thread_dispatcher() { if (m_on_destruction_without_call) m_on_destruction_without_call(); }

                std::thread::id thread_id() const { return m_data->thread.get_id(); }

                void dispatch(move_only_function<void()> func) {
                    {
                        std::unique_lock lock(m_data->mutex);
                        m_data->func = std::move(func);
                    }
                    m_data->wake_up.notify_one();

                    // Clean up the dispatcher.
                    m_data = nullptr;
                    m_on_destruction_without_call = {};
                }

            private:
                // Private constructor that the thread pool can access by being a friend.
                thread_dispatcher(thread_data* data, move_only_function<void()> on_destruction_without_call) : m_data(data), m_on_destruction_without_call(std::move(on_destruction_without_call)) {}

                thread_data* m_data;
                move_only_function<void()> m_on_destruction_without_call;
            };



            // Holds multiple threads while waiting to dispatch a function.
            // The threads are automatically returned to the pool after dispatch_to...(), release(), or destruction of this object.
            // This object should never be shared accross threads.
            class thread_holder {
                friend class thread_pool; // So thread_pool can access the contructor.

            public:
                thread_holder() = default;
                thread_holder(const thread_holder&) = delete;
                thread_holder& operator=(const thread_holder&) = delete;
                thread_holder(thread_holder&&) = default;
                thread_holder& operator=(thread_holder&&) = default;
                ~thread_holder() { release(); }

                // Returns the number of threads we are holding.
                size_t size() const { return m_dispatchers.size(); }
                size_t desired_size() const { return m_nb_desired; }
                bool empty() const { return size() == 0; }
                bool full() const { return size() == m_nb_desired; }
    
                // Dispatches a function func(size_t index) -> R to size() threads, where index is the thread index from 0 to size() - 1, and R is the return value.
                // Note that if size() == 0, func() will not be called at all.
                // Waits for all threads to complete, then returns a vector of R, one for each thread.
                // If the return type is void, we still wait but return nothing.
                // This function can only be called once. Afterwards the threads return to the pool.
                template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                auto dispatch_to_reserved(F&& func) {

                    // Dispatch.
                    move_only_function<void()> self_func;
                    std::vector<std::future<R>> futures(m_dispatchers.size());
                    for (size_t i = 0; i < size(); i++) {
                        auto [async_func, future] = make_async_func(func, i);
                        futures[i] = std::move(future);
                        if (m_dispatchers[i].thread_id() == std::this_thread::get_id()) {
                            self_func = std::move(async_func);
                        }
                        else {
                            m_dispatchers[i].dispatch(std::move(async_func));
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

                // Dispatches a function func(size_t index) -> R to size() threads, where index is the thread index from 0 to size() - 1, and R is the return value.
                // Note that if size() == 0, func(0) will be called synchronously.
                // Waits for all threads to complete, then returns a vector of R, one for each thread.
                // If the return type is void, we still wait but return nothing.
                // This function can only be called once. Afterwards the threads return to the pool.
                template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                auto dispatch_to_at_least_one(F&& func) {
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

                // Dispatches a function func(size_t index) -> R to desired_size() threads, where index is the thread index from 0 to desired_size() - 1, and R is the return value.
                // If size() < desired_size(), the threads will simply queue up the work.
                // Waits for all threads to complete, then returns a vector of R, one for each thread.
                // If the return type is void, we still wait but return nothing.
                // This function can only be called once. Afterwards the threads return to the pool.
                template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                auto dispatch_to_all(F&& func) {

                    if (full()) {
                        return dispatch_to_reserved(func);
                    }
                    else {
                        std::atomic<size_t> shared_index {};
                        std::conditional_t<std::is_void_v<R>, char, std::vector<R>> results;
                        if constexpr (!std::is_void_v<R>)
                            results.resize(m_nb_desired);

                        auto full_func = [&](size_t) {
                            for (size_t i = shared_index++; i < m_nb_desired; i = shared_index++) {
                                if constexpr (std::is_void_v<R>)
                                    func(i);
                                else
                                    results[i] = func(i);
                            }
                        };
                        dispatch_to_at_least_one(full_func);

                        if constexpr (!std::is_void_v<R>)
                            return results;
                    }
                }

                // Returns the threads to the pool without doing anything.
                void release() { m_dispatchers.clear(); }

            private:
                // Private constructor that the thread pool can access by being a friend.
                thread_holder(size_t nb_desired, std::vector<thread_dispatcher> dispatchers) : m_nb_desired(nb_desired), m_dispatchers(std::move(dispatchers)) {}

                size_t m_nb_desired = 0;
                std::vector<thread_dispatcher> m_dispatchers;
            };
               


            // A simple thread pool.
            // 
            // Basic usage:
            // 
            // // Singlethreaded loop over 100 elements.
            // for(int i = 0; i < 100; i++)
            //     std::cout << i << '\n'; 
            // 
            // // Equivalent loop using multithreading.
            // thread_pool::get().reserve(100).dispatch_to_all([](int i) { 
            //     std::cout << i << '\n'; 
            // });
            // 
            // There are 3 flavors of dispatch():
            //  -reserve(N).dispatch_to_all()           will call the function exactly N times, possibly reusing the same threads.
            //  -reserve(N).dispatch_to_at_least_one()  will call the function between 1 and N times, never reusing the same threads.
            //  -reserve(N).dispatch_to_reserved()      will call the function between 0 and N times, never reusing the same threads.
            // 
            // You can inquire the object returned by reserve() for more info.
            //
            // Note that the thread pool is recursive. Jobs dispatched to it can also dispatch their own jobs.
            // In this case, the current thread will call its own job synchronously while dispatching to other available threads.
            // The only functions that are disallowed recursively are enable() and disable().
            //
            class thread_pool: public singleton<thread_pool> {
                friend class thread_holder;

            private:
                std::vector<size_t> m_available_threads;
                std::vector<thread_data> m_data;
                mutable std::mutex m_global_mutex;
                std::atomic<bool> m_enabled = {};

            public:

                thread_pool(): m_data(std::thread::hardware_concurrency()) { enable(); }
                ~thread_pool() { disable(); }

                // Various getters.
                bool is_enabled() const { return m_enabled; }
                size_t size() const { return m_data.size(); }

                size_t nb_working() const {
                    if (!m_enabled)
                        return 0;
                    return size() - nb_available();
                }

                size_t nb_available() const {
                    if (!m_enabled)
                        return 0;
                    std::unique_lock lock(m_global_mutex); 
                    return m_available_threads.size() + self_thread_index().has_value(); 
                }

                // Returns true if we are inside a worker.
                bool is_worker() const { return self_thread_index().has_value(); }

                // Returns an object holding up to nb_desired threads.
                // You can get get the number of threads using size(), and when you are ready call dispatch() to send a function to all the threads.
                [[nodiscard]] thread_holder reserve(size_t nb_desired) { return thread_holder(nb_desired, create_dispatchers(nb_desired)); }

                // Starts the workers. If they are already started, does nothing.
                void enable() {
                    assert(!is_worker()); // This function cannot be called recursively.
                    if (m_enabled)
                        return;

                    std::unique_lock lock(m_global_mutex);

                    m_enabled = true;
                    m_available_threads.resize(m_data.size());
                    for (size_t i = 0; i < m_data.size(); i++) {
                        m_available_threads[i] = i;
                        m_data[i].thread = std::thread(&thread_pool::worker_loop, this, i);
                    }
                }

                // Stops the workers. If they are already stopped, does nothing.
                void disable() {
                    assert(!is_worker()); // This function cannot be called recursively.
                    if (!m_enabled)
                        return;

                    std::unique_lock lock(m_global_mutex);

                    // Signal all to exit.
                    m_enabled = false;
                    for (size_t i = 0; i < m_data.size(); i++)
                        m_data[i].wake_up.notify_one();

                    // Unlock while we wait for all threads to exit.
                    lock.unlock();
                    for (size_t i = 0; i < m_data.size(); i++)
                        m_data[i].thread.join();

                    // Lock again and mark all threads as unavailable.
                    lock.lock();
                    m_available_threads.clear();
                }

            private:
                // Infinite loop that ends when stop() is called.
                void worker_loop(size_t i) {
                    auto& data = m_data[i];
                    move_only_function<void()> func;
                    while (true) {
                        {
                            // Wait until we have something to do.
                            std::unique_lock lock(data.mutex);
                            data.wake_up.wait(lock, [this, &data] { return data.func || !m_enabled; });
                            func = std::move(data.func);
                        }

                        // If we have to exit, do it now.
                        if (!m_enabled)
                            return;

                        // Call the function.
                        func();

                        {
                            // Mark the thread as available.
                            std::unique_lock lock(m_global_mutex);
                            if(m_enabled)
                                m_available_threads.push_back(i);
                        }

                    }
                }

                // If the current thread is part of the pool, return its index.
                std::optional<size_t> self_thread_index() const {
                    auto it = std::find_if(m_data.begin(), m_data.end(), [](const thread_data& data) { return data.thread.get_id() == std::this_thread::get_id(); });
                    if (it == m_data.end())
                        return std::nullopt;
                    else
                        return it - m_data.begin();
                }

                // Creates a dispatcher for a single thread.
                thread_dispatcher create_dispatcher_unsafe(size_t i) {

                    move_only_function<void()> add_back_to_list;
                    if (auto it = std::find(m_available_threads.begin(), m_available_threads.end(), i); it != m_available_threads.end()) {
                        // Might not be available if called from the same thread.
                        m_available_threads.erase(it); 
                        add_back_to_list = [this, i]() { m_available_threads.push_back(i); };
                    }

                    return thread_dispatcher(&m_data[i], std::move(add_back_to_list));
                }

                // Creates up to nb_desired dispatchers.
                // If the current thread is part of the pool, it is always returned.
                std::vector<thread_dispatcher> create_dispatchers(size_t nb_desired) {
                    if (nb_desired == 0)
                        return {};

                    std::vector<thread_dispatcher> dispatchers;
                    dispatchers.reserve(std::min(nb_desired, size()));
                    std::unique_lock lock(m_global_mutex);

                    // If the calling thread is a worker, return its data for sure.
                    if (auto self_index = self_thread_index()) {
                        dispatchers.push_back(create_dispatcher_unsafe(*self_index));
                        nb_desired--;
                    }

                    // Get other threads.
                    while (nb_desired > 0 && !m_available_threads.empty()) {
                        dispatchers.push_back(create_dispatcher_unsafe(m_available_threads.back()));
                        nb_desired--;
                    }

                    return dispatchers;
                }


            };


        } // namespace thread_pool_namespace
    } // namespace details



    // Exports.
    using details::thread_pool_namespace::thread_holder;
    using details::thread_pool_namespace::thread_pool;



} // namespace palla