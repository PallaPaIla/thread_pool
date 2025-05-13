#pragma once

#include <vector>

namespace palla {
    namespace details {
        namespace thread_pool_namespace {

            // A fake thread pool which does everything synchronously with the same API as the real thread pool.
            class thread_pool {
            public:
                // Public types.

                // A dummy sub pool which holds no threads.
                class sub_pool {
                    friend class thread_pool;
                private:
                    size_t m_desired_size = 0;
                    sub_pool(size_t desired_size) : m_desired_size(desired_size) {}
                public:
                    // sub_pool is default constructible and movable, but not copyable.
                    sub_pool() = default;

                    void release() { m_desired_size = 0; }
                    size_t size() const { return 0; }
                    size_t desired_size() const { return m_desired_size; }
                    bool empty() const { return size() == 0; }
                    bool full() const { return size() == desired_size(); }

                    // Dispatchers.
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_reserved(F&& func) {
                        if constexpr (!std::is_void_v<R>)
                            return std::vector<R> {};
                    }
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_at_least_one(F&& func) {
                        if constexpr (std::is_void_v<R>)
                            func(0);
                        else
                            return std::vector<R> { func(0) };
                    }
                    template<class F, class R = std::decay_t<decltype(std::declval<F>()(0))>>
                    auto dispatch_to_all(F&& func) {
                        if constexpr (std::is_void_v<R>) {
                            for (int i = 0; i < (int)m_desired_size; i++) {
                                func(i);
                            }
                        }
                        else {
                            std::vector<R> results(m_desired_size);
                            for (int i = 0; i < (int)m_desired_size; i++) {
                                results[i] = func(i);
                            }
                            return results;
                        }
                    }
                };

                // Public functions.
                static thread_pool& get() { static thread_pool pool; return pool; }
                void disable() {}
                void enable() {}
                void resize(size_t) {}
                size_t size() const { return 0; }
                bool empty() const { return true; }
                size_t nb_available() const { return 0; }
                size_t nb_working() const { return 0; }
                bool is_worker() const { return false; }
                sub_pool reserve(size_t nb_desired) { return sub_pool(nb_desired); }
            };

        } // namespace thread_pool_namespace
    } // namespace details


    // Exports.
    using details::thread_pool_namespace::thread_pool;


} // namespace palla