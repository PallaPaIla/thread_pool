# thread_pool

A simple header-only thread pool for multithreading.

## Installation and requirements

Copy `header/thread_pool.h` and include it. Requires C++ 20, but is mostly compatible with C++17 barring `std::span`.

## Basic usage

```c++
// Prints the integers from 0 to 100.
palla::thread_pool::get().reserve(100).dispatch_to_all([](int i) {
    std::cout << i << '\n';
});
```

The thread pool is a singleton accessed by `palla::thread_pool::get()`. Each usage of the pool passes through the same `reserve()` and `dispatch()` idiom.

* `reserve()` creates a sub pool consisting of up to some desired number of threads (possibly less). The threads will go back to the main pool upon destruction of the sub pool or intentional `release()`.

* `dispatch()` is a collection of methods to assign a function to the threads. There are 3 flavors:

    * `dispatch_to_reseved()` calls the function once for each thread in the sub pool.
    * `dispatch_to_at_least_one()` acts exactly like `dispatch_to_reseved()` except that if there are 0 threads, the function will be called synchronously.
    * `dispatch_to_all()` acts as if the sub pool contained all the threads requested and dispatches the work evenly between the actual threads. If there are 0 threads it does everything synchronously.

<br>

The function to dispatch should have the signature `(int) -> T`. The integer parameter is for the thread index, and the return value can be any type. All dispatch flavors will return an `std::vector<T>` with 1 element per thread. If `T` is `void`, dispatch will wait for all the threads to complete but will not return anything.

Both the main pool and sub pools have various self-explanatory getters like `size()`, `nb_available()`, `nb_working()`, etc.

## Recursion

The thread pool fully supports recursion.
```c++
#include "thread_pool.h"

// A recursive fibonacci implementation using multiple threads.
int fibonacci(int val) {
    if (val <= 1)
        return val;

    auto results = palla::thread_pool::get().reserve(2).dispatch_to_all([val](int thread_index) {
        return fibonacci(val - 1 - thread_index);
    });

    return results[0] + results[1];
}

fibonacci(10); // 55
```
If the current thread is part of the pool, it will always be selected inside sub pools and `dispatch()` will give the function to other threads first before executing synchronously.

You can check if the thread is in the pool using `thread_pool::get().is_worker()`.


## Changing the number of threads

By default the thread pool starts with one thread per logical core. For example on a 4 core cpu with hyperthreading it will start with 8 threads. This can be changed using various calls:

* `disable()` empties the pool singleton.
* `enable()` returns to the previous state before `disable()` was called.
* `resize()` changes the number of threads in the pool.

The changes take effect immediately on available threads in the main pool, but threads owned by sub pools are only affected when they return to the main pool.

```c++
#include "thread_pool.h"

// Give the pool 8 threads for this example.
auto& pool = palla::thread_pool::get();
pool.resize(8);

// Create a sub pool containing 2 threads.
auto sub_pool = pool.reserve(2);
pool.size();            // 8
pool.nb_available();    // 6

// Disable the pool.
pool.disable();
pool.size();            // 2
pool.nb_available();    // 0

// Release the threads in the sub pool.
sub_pool.release();
pool.size();            // 0
pool.nb_available();    // 0

```