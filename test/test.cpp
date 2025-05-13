
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>

#include "../header/thread_pool.h"

namespace fake {
#include "../header/fake_thread_pool.h"
}

// Console color codes.
namespace colors {
    static const char* const white =    "\033[0m";
    static const char* const green =    "\033[92m";
    static const char* const yellow =   "\033[93m";
    static const char* const red =      "\033[91m";
}

// Utility function to terminate the test.
void make_test_fail(const char* text) {
    std::cout << colors::red << "\nFAIL: " << colors::white << text << "\n\n";
    std::exit(0);
}

// Various static asserts.
void dummy_void(int) {}
int dummy_int(int) { return 0; }

static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_all(&dummy_void))>,          "Incorrect dispatch return type.");
static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_at_least_one(&dummy_void))>, "Incorrect dispatch return type.");
static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_reserved(&dummy_void))>,     "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_all(&dummy_int))>,          "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_at_least_one(&dummy_int))>, "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_reserved(&dummy_int))>,     "Incorrect dispatch return type.");

static_assert(!std::is_copy_constructible_v<palla::thread_pool> &&
              !std::is_copy_assignable_v<palla::thread_pool> &&
              !std::is_move_assignable_v<palla::thread_pool> &&
              !std::is_move_assignable_v<palla::thread_pool>, 
    "thread_pool should neither be movable nor copyable.");

static_assert(!std::is_copy_constructible_v<palla::thread_pool::sub_pool> &&
              !std::is_copy_assignable_v<palla::thread_pool::sub_pool>&&
              std::is_move_assignable_v<palla::thread_pool::sub_pool>&&
              std::is_move_assignable_v<palla::thread_pool::sub_pool>, 
    "sub_pool should be movable but not copyable.");

// An enum representing each dispatch function.
enum class dispatch_func {
    dispatch_to_reserved,
    dispatch_to_at_least_one,
    dispatch_to_all
};

// Verifies the size of a pool.
template<class T>
void verify_pool(const T& pool, size_t expected_size, size_t expected_working, bool is_worker) {

    if (pool.size() != expected_size)
        make_test_fail("Incorrect thread_pool::size().");

    if (pool.nb_working() != expected_working)
        make_test_fail("Incorrect thread_pool::nb_working().");

    if (pool.nb_available() != expected_size - expected_working - is_worker)
        make_test_fail("Incorrect thread_pool::nb_available().");

    if (pool.is_worker() != is_worker)
        make_test_fail("Incorrect thread_pool::is_worker().");

}

// Verifies the size of a sub pool and the dispatch functions.
template<class T>
void verify_sub_pool(T& sub_pool, size_t desired_size, size_t expected_size) {
    // Test the size.
    if (sub_pool.size() != expected_size)
        make_test_fail("Incorrect sub_pool::size().");

    if (sub_pool.desired_size() != desired_size)
        make_test_fail("Incorrect sub_pool::desired_size().");

    if (sub_pool.empty() != (sub_pool.size() == 0))
        make_test_fail("Incorrect sub_pool::empty().");

    if (sub_pool.full() != (sub_pool.size() == desired_size))
        make_test_fail("Incorrect sub_pool::full().");

    // Test dispatch functions.
    auto test_dispatch = [&sub_pool](dispatch_func dispatch_func, size_t expected_size) {

        // Check that each thread is visited exactly once.
        std::vector<std::atomic<size_t>> thread_count(expected_size);
        auto func = [&thread_count](size_t i) {
            thread_count[i]++;
            return i;
        };
        std::vector<size_t> thread_count_return(expected_size), expected_thread_count_return(expected_size);
        std::iota(expected_thread_count_return.begin(), expected_thread_count_return.end(), 0);

        switch (dispatch_func) {
        case dispatch_func::dispatch_to_reserved:      thread_count_return = sub_pool.dispatch_to_reserved(func);      break;
        case dispatch_func::dispatch_to_at_least_one:  thread_count_return = sub_pool.dispatch_to_at_least_one(func);  break;
        case dispatch_func::dispatch_to_all:           thread_count_return = sub_pool.dispatch_to_all(func);           break;
        }

        if (!std::all_of(thread_count.begin(), thread_count.end(), [](size_t i) { return i == 1; }))
            make_test_fail("Incorrect sub_pool::dispatch().");

        if (thread_count_return != expected_thread_count_return)
            make_test_fail("Incorrect sub_pool::dispatch() return value.");
    };

    test_dispatch(dispatch_func::dispatch_to_reserved, expected_size);
    test_dispatch(dispatch_func::dispatch_to_at_least_one, std::max<size_t>(1, expected_size));
    test_dispatch(dispatch_func::dispatch_to_all, desired_size);
}

// Test that functions return the proper values.
void test_functionality() {
    std::cout << "\nTesting functionality.\n" << colors::yellow << "TESTING..." << colors::white << '\r';

    auto& pool = palla::thread_pool::get();

    // Default size.
    if (pool.empty() || pool.size() != std::thread::hardware_concurrency())
        make_test_fail("The pool should have 1 thread per logical core by default.");

    // is_worker().
    if (pool.is_worker())
        make_test_fail("Incorrect thread_pool::is_worker().");
    if (!pool.reserve(1).dispatch_to_reserved([](int) { return palla::thread_pool::get().is_worker(); })[0])
        make_test_fail("Incorrect thread_pool::is_worker().");

    // Test enabling and disabling with various sub pool sizes.
    size_t regular_size = 8;
    for (size_t desired_size_1 = 0; desired_size_1 < regular_size + 2; desired_size_1++) {
        for (size_t desired_size_2 = 0; desired_size_2 < regular_size + 2; desired_size_2++) {

            size_t sub_pool_1_size = std::min(desired_size_1, regular_size);
            size_t sub_pool_2_size = std::min(desired_size_2, regular_size - sub_pool_1_size);

            // Resize to regular_size.
            pool.resize(regular_size);
            verify_pool(pool, regular_size, 0, false);

            // Create and verify a sub pool.
            auto sub_pool_1 = pool.reserve(desired_size_1);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_pool(pool, regular_size, sub_pool_1_size, false);

            // Disable and verify the threads are all removed except those in sub_pool_1.
            pool.disable();
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_pool(pool, sub_pool_1_size, sub_pool_1_size, false);

            // Attempt to create another sub pool when the pool is disabled.
            auto sub_pool_2 = pool.reserve(desired_size_2);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, 0);
            verify_pool(pool, sub_pool_1_size, sub_pool_1_size, false);

            // Reenable the pool and create another sub pool.
            pool.enable();
            sub_pool_2 = pool.reserve(desired_size_2);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(pool, regular_size, sub_pool_1_size + sub_pool_2_size, false);

            // Disable again.
            pool.disable();
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(pool, sub_pool_1_size + sub_pool_2_size, sub_pool_1_size + sub_pool_2_size, false);

            // Release one pool.
            sub_pool_1.release();
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(pool, sub_pool_2_size, sub_pool_2_size, false);

            // Resize to 1.
            pool.resize(1);
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(pool, std::max<size_t>(1, sub_pool_2_size), sub_pool_2_size, false);

            // Release the other pool.
            sub_pool_2.release();
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, 0, 0);
            verify_pool(pool, 1, 0, false);
        }
    }

    std::cout << colors::green << "PASS              " << colors::white;
}


// Test that functions return the proper values.
void test_fake_pool() {
    std::cout << "\nTesting the fake pool.\n" << colors::yellow << "TESTING..." << colors::white << '\r';

    auto& pool = fake::palla::thread_pool::get();
    verify_pool(pool, 0, 0, false);
    pool.resize(42);
    verify_pool(pool, 0, 0, false);
    pool.enable();
    verify_pool(pool, 0, 0, false);

    for (size_t desired_size = 0; desired_size < 4; desired_size++) {
        auto sub_pool = pool.reserve(desired_size);
        verify_sub_pool(sub_pool, desired_size, 0);
    }

    std::cout << colors::green << "PASS              " << colors::white;
}


// Time several threads and ensure we arrive at the expected time.
void verify_concurrency(size_t nb_threads_in_pool, size_t nb_threads_desired, std::chrono::duration<double> time_to_sleep) {
    auto func = [time_to_sleep](int) { std::this_thread::sleep_for(time_to_sleep); };

    auto& pool = palla::thread_pool::get();
    pool.resize(nb_threads_in_pool);

    auto sub_pool = pool.reserve(nb_threads_desired);

    auto test_dispatch = [&](dispatch_func dispatch_func, std::chrono::duration<double> expected_time) {

        auto start = std::chrono::steady_clock::now();
        switch (dispatch_func) {
        case dispatch_func::dispatch_to_reserved:      sub_pool.dispatch_to_reserved(func);      break;
        case dispatch_func::dispatch_to_at_least_one:  sub_pool.dispatch_to_at_least_one(func);  break;
        case dispatch_func::dispatch_to_all:           sub_pool.dispatch_to_all(func);           break;
        }
        auto end = std::chrono::steady_clock::now();
        auto actual_time = end - start;
        if (abs(expected_time - actual_time) > std::max(expected_time, time_to_sleep) * 0.2)
            make_test_fail("The pool is not concurrent.");
    };

    test_dispatch(dispatch_func::dispatch_to_reserved, sub_pool.empty() ? std::chrono::seconds(0) : time_to_sleep);
    test_dispatch(dispatch_func::dispatch_to_at_least_one, time_to_sleep);
    test_dispatch(dispatch_func::dispatch_to_all, std::ceil((double)sub_pool.desired_size() / sub_pool.size()) * time_to_sleep);

}

// Test that the pool actually uses multiple threads and is not just a singlethreaded loop.
void test_concurrency() {
    std::cout << "\nTesting concurrency.\n" << colors::yellow << "TESTING..." << colors::white << '\r';

    verify_concurrency(0, 2, std::chrono::milliseconds(200));
    verify_concurrency(4, 2, std::chrono::milliseconds(200));
    verify_concurrency(4, 4, std::chrono::milliseconds(200));
    verify_concurrency(4, 6, std::chrono::milliseconds(200));
    verify_concurrency(8, 100, std::chrono::milliseconds(100));

    std::cout << colors::green << "PASS              " << colors::white;
}



// Sleeps for a random time in microseconds.
void sleep_random_us(int us) {
    if (us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// Makes various thread_pool calls recursively up to a certain depth.
void recursive_call(std::chrono::time_point<std::chrono::steady_clock> until, size_t max_depth) {
    auto time = std::chrono::steady_clock::now();
    if (time > until || max_depth < 0)
        return;

    auto& pool = palla::thread_pool::get();

    auto recurse = [until, max_depth](int) { recursive_call(until, max_depth - 1); };

    constexpr int SLEEP_TIME_MAX = 1; // In microseconds.
    std::minstd_rand rand((std::uint32_t)time.time_since_epoch().count());
    auto sleep_rand = [&rand]() mutable {
        int us = std::uniform_int_distribution(0, SLEEP_TIME_MAX)(rand);
        if (us > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(us));
    };

    do {
        // Do various calls.
        sleep_rand();
        (void)pool.size();
        sleep_rand();
        (void)pool.empty();
        sleep_rand();
        (void)pool.nb_available();
        sleep_rand();
        (void)pool.nb_working();
        sleep_rand();
        (void)pool.is_worker();
        sleep_rand();
        pool.resize(std::uniform_int_distribution(0, 12)(rand));
        sleep_rand();
        (void)pool.reserve(std::uniform_int_distribution(0, 4)(rand));
        sleep_rand();
        auto sub_pool = pool.reserve(std::uniform_int_distribution(0, 4)(rand));
        sleep_rand();
        sub_pool.dispatch_to_all(recurse);
        sleep_rand();
        sub_pool.dispatch_to_at_least_one(recurse);
        sleep_rand();
        sub_pool.dispatch_to_reserved(recurse);
        sleep_rand();
    } while (std::chrono::steady_clock::now() < until);

}

// Test that there are no deadlocks.
void test_deadlocks() {
    std::cout << "\nTesting for deadlocks.\n" << colors::yellow;

    constexpr size_t TEST_TIME_SECONDS = 10;
    const auto start_time = std::chrono::steady_clock::now();

    auto future = std::async([start_time]() {
        for (size_t i = 0; i <= TEST_TIME_SECONDS; i++) {
            std::cout << "TESTING for " << std::chrono::seconds(TEST_TIME_SECONDS - i) << "..\r";
            std::this_thread::sleep_until(start_time + std::chrono::seconds(i));
        }
        std::cout << "DONE                               \r";
    });

    recursive_call(start_time + std::chrono::seconds(TEST_TIME_SECONDS), 3);

    std::cout << colors::green << "PASS" << colors::white;
}


// Main function.
int main() {

    std::cout << colors::white;

    test_functionality();
    test_fake_pool();
    test_concurrency();
    test_deadlocks();

    std::cout << "\n\nGlobal Result: " << colors::green << "PASS" << colors::white << "\n\n";

    return 0;
}