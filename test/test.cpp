
#include <iostream>
#include <numeric>
#include <random>

#include "../palla/thread_pool.h"

// Console color codes.
namespace colors {
    static const char* const white = "\033[0m";
    static const char* const green = "\033[92m";
    static const char* const yellow = "\033[93m";
    static const char* const red = "\033[91m";
}

// Utility function to terminate the test.
void make_test_fail(const char* text) {
    std::cout << colors::red << "\nFAIL: " << colors::white << text << "\n\n";
    std::exit(0);
}

// Various static asserts.
void dummy_void(size_t) {}
int dummy_int(size_t) { return 0; }

static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_all(&dummy_void))>, "Incorrect dispatch return type.");
static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_at_least_one(&dummy_void))>, "Incorrect dispatch return type.");
static_assert(std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_reserved(&dummy_void))>, "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_all(&dummy_int))>, "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_at_least_one(&dummy_int))>, "Incorrect dispatch return type.");
static_assert(!std::is_void_v<decltype(palla::thread_pool::get().reserve(8).dispatch_to_reserved(&dummy_int))>, "Incorrect dispatch return type.");

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

// Verifies the size of a pool.
void verify_pool(size_t expected_size, size_t expected_working, bool is_worker) {

    auto& pool = palla::thread_pool::get();

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
void verify_sub_pool(palla::thread_pool::sub_pool& sub_pool, size_t desired_size, size_t expected_size) {

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
    enum class dispatch_func {
        dispatch_to_reserved,
        dispatch_to_at_least_one,
        dispatch_to_all
    };

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
    if (!pool.reserve(1).dispatch_to_reserved([](size_t) { return palla::thread_pool::get().is_worker(); })[0])
        make_test_fail("Incorrect thread_pool::is_worker().");

    // Test enabling and disabling with various sub pool sizes.
    size_t regular_size = 8;
    for (size_t desired_size_1 = 0; desired_size_1 < regular_size + 2; desired_size_1++) {
        for (size_t desired_size_2 = 0; desired_size_2 < regular_size + 2; desired_size_2++) {

            size_t sub_pool_1_size = std::min(desired_size_1, regular_size);
            size_t sub_pool_2_size = std::min(desired_size_2, regular_size - sub_pool_1_size);

            // Resize to regular_size.
            pool.resize(regular_size);
            verify_pool(regular_size, 0, false);

            // Create and verify a sub pool.
            auto sub_pool_1 = pool.reserve(desired_size_1);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_pool(regular_size, sub_pool_1_size, false);

            // Disable and verify the threads are all removed except those in sub_pool_1.
            pool.disable();
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_pool(sub_pool_1_size, sub_pool_1_size, false);

            // Attempt to create another sub pool when the pool is disabled.
            auto sub_pool_2 = pool.reserve(desired_size_2);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, 0);
            verify_pool(sub_pool_1_size, sub_pool_1_size, false);

            // Reenable the pool and create another sub pool.
            pool.enable();
            sub_pool_2 = pool.reserve(desired_size_2);
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(regular_size, sub_pool_1_size + sub_pool_2_size, false);

            // Disable again.
            pool.disable();
            verify_sub_pool(sub_pool_1, desired_size_1, sub_pool_1_size);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(sub_pool_1_size + sub_pool_2_size, sub_pool_1_size + sub_pool_2_size, false);

            // Release one pool.
            sub_pool_1.release();
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(sub_pool_2_size, sub_pool_2_size, false);

            // Resize to 1.
            pool.resize(1);
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, desired_size_2, sub_pool_2_size);
            verify_pool(std::max<size_t>(1, sub_pool_2_size), sub_pool_2_size, false);

            // Release the other pool.
            sub_pool_2.release();
            verify_sub_pool(sub_pool_1, 0, 0);
            verify_sub_pool(sub_pool_2, 0, 0);
            verify_pool(1, 0, false);

        }
    }

    std::cout << colors::green << "PASS              " << colors::white;
}



// A thread-safe random function since rand() is not garanteed to be thread-safe.
int thread_safe_random(int start, int end) {
    static std::atomic<int> seed = 42;
    thread_local std::minstd_rand rand(seed++);
    return std::uniform_int_distribution(start, end)(rand);
}

// Sleeps for a random time in microseconds.
void sleep_random_us(int min, int max) {
    auto sleep_time_us = thread_safe_random(min, max);
    if (sleep_time_us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
}

// Makes various thread_pool calls recursively up to a certain depth.
void recursive_call(size_t remaining_recursions) {

    if (remaining_recursions == 0)
        return;

    auto& pool = palla::thread_pool::get();

    const int sleep_time_max = 20; // In microseconds.
    auto recurse = [remaining_recursions](size_t) { recursive_call(remaining_recursions - 1); };

    // Do various calls.
    sleep_random_us(0, sleep_time_max);
    (void)pool.size();
    sleep_random_us(0, sleep_time_max);
    (void)pool.empty();
    sleep_random_us(0, sleep_time_max);
    (void)pool.nb_available();
    sleep_random_us(0, sleep_time_max);
    (void)pool.nb_working();
    sleep_random_us(0, sleep_time_max);
    (void)pool.is_worker();
    sleep_random_us(0, sleep_time_max);
    pool.resize(thread_safe_random(0, 12));
    sleep_random_us(0, sleep_time_max);
    (void)pool.reserve(thread_safe_random(0, 4));
    sleep_random_us(0, sleep_time_max);
    auto sub_pool = pool.reserve(thread_safe_random(0, 4));
    sleep_random_us(0, sleep_time_max);
    sub_pool.dispatch_to_all(recurse);
    sleep_random_us(0, sleep_time_max);
    sub_pool.dispatch_to_at_least_one(recurse);
    sleep_random_us(0, sleep_time_max);
    sub_pool.dispatch_to_reserved(recurse);
    sleep_random_us(0, sleep_time_max);
}


// Test that there are no deadlocks.
void test_deadlocks() {

    std::cout << "\nTesting for deadlocks.\n" << colors::yellow << "TESTING..." << colors::white << "  if this text stays up for 30s+ a deadlock has occurred.\r";

    recursive_call(3);

    std::cout << colors::green << "PASS                                                                                  " << colors::white;
}



// Main function.
int main() {

    test_functionality();
    test_deadlocks();


    std::cout << "\n\nGlobal Result: " << colors::green << "PASS" << colors::white << "\n\n";

    return 0;
}