# 附录C 协作组

## C.1. Introduction

Cooperative Groups 是 CUDA 9 中引入的 CUDA 编程模型的扩展，用于组织通信线程组。协作组允许开发人员表达线程通信的粒度，帮助他们表达更丰富、更有效的并行分解。

从历史上看，CUDA 编程模型为同步协作线程提供了一个单一、简单的构造：线程块的所有线程之间的屏障，如使用 `__syncthreads()` 内部函数实现的那样。但是，程序员希望以其他粒度定义和同步线程组，以“集体”组范围功能接口的形式实现更高的性能、设计灵活性和软件重用。为了表达更广泛的并行交互模式，许多面向性能的程序员已经求助于编写自己的临时和不安全的原语来同步单个 warp 中的线程，或者跨运行在单个 GPU 上的线程块集。虽然实现的性能改进通常很有价值，但这导致了越来越多的脆弱代码集合，随着时间的推移和跨 GPU 架构的不同，这些代码的编写、调整和维护成本很高。合作组通过提供安全且面向未来的机制来启用高性能代码来解决这个问题。

## C.2. What's New in CUDA 11.0

* 使用网格范围的组不再需要单独编译，并且同步该组的速度现在提高了 `30%`。此外，我们在最新的 Windows 平台上启用了协作启动，并在 MPS 下运行时增加了对它们的支持。
* `grid_group `现在可以转换为 `thread_group`。
* 线程块切片和合并组的新集合：`reduce` 和 `memcpy_async`。
* 线程块切片和合并组的新分区操作：`labeled_pa​​rtition` 和 `binary_partition`。
* 新的 API，`meta_group_rank` 和 `meta_group_size`，它们提供有关导致创建该组的分区的信息。
* 线程块`tile`现在可以在类型中编码其父级，这允许对发出的代码进行更好的编译时优化。
* 接口更改：`grid_group` 必须在声明时使用 `this_grid()` 构造。默认构造函数被删除。
  

注意：在此版本中，我们正朝着要求 C++11 提供新功能的方向发展。在未来的版本中，所有现有 API 都需要这样做。

## C.3. Programming Model Concept
协作组编程模型描述了 CUDA 线程块内和跨线程块的同步模式。 它为应用程序提供了定义它们自己的线程组的方法，以及同步它们的接口。 它还提供了强制执行某些限制的新启动 API，因此可以保证同步正常工作。 这些原语在 CUDA 内启用了新的协作并行模式，包括生产者-消费者并行、机会并行和整个网格的全局同步。

合作组编程模型由以下元素组成：
* 表示协作线程组的数据类型；
* 获取由 CUDA 启动 API 定义的隐式组的操作（例如，线程块）；
* 将现有群体划分为新群体的集体；
* 用于数据移动和操作的集体算法（例如 `memcpy_async、reduce、scan`）；
* 同步组内所有线程的操作；
* 检查组属性的操作；
* 公开低级别、特定于组且通常是硬件加速的操作的集合。

协作组中的主要概念是对象命名作为其中一部分的线程集的对象。 这种将组表示为一等程序对象的方式改进了软件组合，因为集合函数可以接收表示参与线程组的显式对象。 该对象还明确了程序员的意图，从而消除了不合理的架构假设，这些假设会导致代码脆弱、对编译器优化的不良限制以及与新一代 GPU 的更好兼容性。

为了编写高效的代码，最好使用专门的组（通用会失去很多编译时优化），并通过引用打算以某种协作方式使用这些线程的函数来传递这些组对象。

合作组需要 CUDA 9.0 或更高版本。 要使用合作组，请包含头文件：
```C++
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>
```

并使用合作组命名空间：

```C++
using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;
```

可以使用 nvcc 以正常方式编译代码，但是如果您希望使用 `memcpy_async、reduce` 或 `scan` 功能并且您的主机编译器的默认不是 C++11 或更高版本，那么您必须添加 `--std=c++11`到命令行。

### C.3.1. Composition Example
为了说明组的概念，此示例尝试执行块范围的求和。 以前，编写此代码时对实现存在隐藏的约束：
```C++
__device__ int sum(int *x, int n) {
    // ...
    __syncthreads();
    return total;
}

__global__ void parallel_kernel(float *x) {
    // ...
    // Entire thread block must call sum
    sum(x, n);
}
```

线程块中的所有线程都必须到达` __syncthreads()` 屏障，但是，对于可能想要使用 `sum(...)` 的开发人员来说，这个约束是隐藏的。 对于合作组，更好的编写方式是：
```C++
__device__ int sum(const thread_block& g, int *x, int n) {
    // ...
    g.sync()
    return total;
}

__global__ void parallel_kernel(...) {
    // ...
    // Entire thread block must call sum
    thread_block tb = this_thread_block();
    sum(tb, x, n);
    // ...
}
```

## C.4. Group Types

### C.4.1. Implicit Groups

隐式组代表内核的启动配置。不管你的内核是如何编写的，它总是有一定数量的线程、块和块尺寸、单个网格和网格尺寸。另外，如果使用多设备协同启动API，它可以有多个网格（每个设备一个网格）。这些组为分解为更细粒度的组提供了起点，这些组通常是硬件加速的，并且更专门针对开发人员正在解决的问题。

尽管您可以在代码中的任何位置创建隐式组，但这样做很危险。为隐式组创建句柄是一项集体操作——组中的所有线程都必须参与。如果组是在并非所有线程都到达的条件分支中创建的，则可能导致死锁或数据损坏。出于这个原因，建议您预先为隐式组创建一个句柄（尽可能早，在任何分支发生之前）并在整个内核中使用该句柄。出于同样的原因，必须在声明时初始化组句柄（没有默认构造函数），并且不鼓励复制构造它们。

#### C.4.1.1. Thread Block Group

任何 CUDA 程序员都已经熟悉某一组线程：线程块。 Cooperative Groups 扩展引入了一个新的数据类型 `thread_block`，以在内核中明确表示这个概念。

```C++
class thread_block;
```
```C++
thread_block g = this_thread_block();
```

公开成员函数：

|`static void sync()`: |Synchronize the threads named in the group|
|----|----|
|`static unsigned int thread_rank()`:| Rank of the calling thread within [0, num_threads)|
|`static dim3 group_index()`:| 3-Dimensional index of the block within the launched grid|
|`static dim3 thread_index()`: |3-Dimensional index of the thread within the launched block|
|`static dim3 dim_threads()`: |Dimensions of the launched block in units of threads|
|`static unsigned int num_threads()`:|Total number of threads in the group|

旧版成员函数（别名）:
|`static unsigned int size()`: |Total number of threads in the group (alias of num_threads())|
|----|----|
|`static dim3 group_dim()`:| Dimensions of the launched block (alias of dim_threads())|


示例:
```C++
/// Loading an integer from global into shared memory
__global__ void kernel(int *globalInput) {
    __shared__ int x;
    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        // load from global into shared for all threads to work with
        x = (*globalInput);
    }
    // After loading data into shared memory, you want to synchronize
    // if all threads in your thread block need to see it
    g.sync(); // equivalent to __syncthreads();
}
```

#### 注意：组中的所有线程都必须参与集体操作，否则行为未定义。

相关：`thread_block` 数据类型派生自更通用的 `thread_group` 数据类型，可用于表示更广泛的组类。












