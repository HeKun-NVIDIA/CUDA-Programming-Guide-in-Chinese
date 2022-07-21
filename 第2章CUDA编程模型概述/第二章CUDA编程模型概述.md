# 2.编程模型
#### 本章通过概述CUDA编程模型是如何在c++中公开的，来介绍CUDA的主要概念。
#### [编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中给出了对 CUDA C++ 的广泛描述。

#### 本章和下一章中使用的向量加法示例的完整代码可以在 vectorAdd [CUDA示例](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition)中找到。


## 2.1 内核
CUDA C++ 通过允许程序员定义称为kernel的 C++ 函数来扩展 C++，当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。


使用 `__global__` 声明说明符定义内核，并使用新的 `<<<...>>>` 执行配置（execution configuration）语法指定内核调用时的 CUDA 线程数（请参阅 [C++ 语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)）。 每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问。


作为说明，以下示例代码使用内置变量 `threadIdx` 将两个大小为 N 的向量 A 和 B 相加，并将结果存储到向量 C 中：
```C++
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

这里，执行 VecAdd() 的 N 个线程中的每一个线程都会执行一个加法。

## 2.2 线程层次

为方便起见，threadIdx 是一个 3 分量`(3-component)`向量，因此可以使用一个一维、二维或三维的线程索引`(thread index)`来识别线程，形成一个具有一个维度、两个维度或三个维度的、由线程组成的块，我们称之为线程块`(thread block)`。 这提供了一种自然的方法来对某一范围（例如向量、矩阵或空间）内的元素进行访问并调用计算。

线程的索引和它的线程 ID 以一种直接的方式相互关联：对于一维块，它们是相同的； 对于大小为(Dx, Dy)的二维块，索引为(x, y)的线程的线程ID为(x + y\*Dx)； 对于大小为 (Dx, Dy, Dz) 的三维块，索引为 (x, y, z) 的线程的线程 ID 为 (x + y\*Dx + z\*Dx\*Dy)。

例如，下面的代码将两个大小为NxN的矩阵A和B相加，并将结果存储到矩阵C中:
```C++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
每个块的线程数量是有限制的，因为一个块中的所有线程都应当驻留在同一个处理器核心上，并且共享了该核心有限的内存资源。在当前的GPU中，一个线程块可能包含多达1024个线程。

但是，一个内核可以由多个形状相同的线程块执行，因此线程的总数等于每个块的线程数乘以块数。

块被组织成一维、二维或三维的线程块网格(`grid`)，如下图所示。网格中的线程块数量通常由正在处理的数据的大小决定，通常会超过系统中的处理器数量。

![grid-of-thread-blocks.png](grid-of-thread-blocks.png)

```<<<...>>> ```语法中所指定的每个块的线程数和每个网格的块数的类型为 ```int``` 或 `dim3` 类型。如上例所示，可以指定二维块或网格。

网格中的每个块都可以由一个具有一维、二维或三维的唯一索引进行识别，该索引可以通过内置的`blockIdx`变量在内核中访问。线程块的维度可以通过内置的`blockDim`变量在内核中访问。

扩展前面的`MatAdd()`示例对多个块进行处理，代码如下所示。
```C++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
线程块大小为16x16(256个线程)，尽管在本例中是任意更改的，但这是一种常见的选择。与前文一样，网格中可以创建足够多的块，这样使得矩阵中的每个元素都由一个线程进行处理。为简单起见，本例假设在每个维度中，一个网格所具有的线程数可以被该维度中一个块所具有的线程数所整除，尽管事实并非如此。

线程块需要具备独立执行的能力：必须可以以任何顺序执行它们，同时无论并行或串行均可以。 这种独立性的要求让线程块可以在任意数量的内核之间，以任意顺序来调度，如下图所示，这使程序员能够编写支持处理器核心数量扩展的代码。

![automatic-scalability.png](automatic-scalability.png)

一个块内的线程可以进行协作，协作通过使用一些共享内存`(shared memory)`来共享数据或通过同步彼此执行来协调内存访问实现。 更准确地说，可以通过调用 `__syncthreads()` 内部函数来指定内核中的同步点； `__syncthreads()` 充当屏障，块中的所有线程必须等待同步，然后才能继续运行。 [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) 给出了一个使用共享内存的例子。 除了` __syncthreads()` 之外，[Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) 还提供了一组丰富的线程同步示例。

为了高效协作，共享内存是每个处理器核心附近的低延迟内存（很像 L1 缓存），并且 `__syncthreads()` 是轻量级的。

## 2.3 存储体系结构
CUDA 线程在执行期间可以从多种内存空间中访问数据，如下图所示。每个线程都有私有的本地内存。 每个线程块都具有共享内存，该共享内存内存对该块中的所有线程可见，并且具有与该块相同的生命周期。 所有线程都可以访问相同的全局内存。

![memory-hierarchy.png](memory-hierarchy.png)

还有两个额外的只读内存空间可供所有线程访问：常量和纹理内存空间。 全局、常量和纹理内存空间针对不同的内存使用进行了优化（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。 纹理内存还为某些特定数据格式提供了不同的寻址模式以及数据过滤方法（请参阅[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)）。

全局、常量和纹理内存空间在同一应用程序的内核启动过程中是持久存在的。

## 2.4 异构编程
如下图所示，CUDA 编程模型假定 CUDA 线程在物理独立的设备`(device)`上执行，该设备作为运行 C++ 程序的主机`(host)`的协处理器运行。例如，当内核在 GPU 上执行而 C++ 程序的其余部分在 CPU 上执行时，就是这种情况。

![heterogeneous-programming.png](heterogeneous-programming.png)

CUDA 编程模型还假设主机(`host`)和设备(`device`)都在 DRAM 中维护自己独立的内存空间，分别称为主机内存`(host memory)`和设备内存`(device memory)`。因此，需要程序通过调用 CUDA 运行时（在[编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中描述）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存分配和释放以及主机和设备内存之间的数据传输。

统一内存提供托管内存`(managed memory)`来桥接主机和设备内存空间。托管内存可以被系统中的所有 CPU 和 GPU 访问，作为具有公共地址空间，构建一个单一的、连贯的内存映像。此功能可实现对设备内存的超额订阅`(oversubscription)`，并且无需在主机和设备上显式镜像数据，从而大大简化了移植应用程序的任务。有关统一内存的介绍，请参阅统一[内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)。

#### 注:串行代码在主机(`host`)上执行，并行代码在设备(`device`)上执行。

## 2.5 异步SIMT编程模型
在 CUDA 编程模型中，线程是进行计算或内存操作的最低级别的抽象。 从基于 NVIDIA Ampere GPU 架构的设备开始，CUDA 编程模型通过异步编程模型为访存操作提供加速。 异步编程模型定义了与 CUDA 线程相关的异步操作的行为。

异步编程模型为 CUDA 线程之间的同步定义了[异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier)的行为。 该模型还解释并定义了如何使用 cuda::memcpy_async 在 GPU计算时从全局内存中异步移动数据。

### 2.5.1 异步操作

异步操作定义为一种由CUDA线程发起的操作，并且与其他线程一样异步执行。在结构良好的程序中，一个或多个CUDA线程可与异步操作同步。发起异步操作的CUDA线程不需要在同步线程中。

这样的异步线程（as-if 线程）总是与发起异步操作的 CUDA 线程相关联。异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理（例如，`cuda::memcpy_async`）或在库中隐式管理（例如，`cooperative_groups::memcpy_async`）。

同步对象可以是 `cuda::barrier` 或 `cuda::pipeline`。这些对象在[Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) 和 [Asynchronous Data Copies using cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memcpy_async_pipeline)中进行了详细说明。这些同步对象可以在不同的线程范围内使用。作用域定义了一组线程，这些线程可以使用同步对象与异步操作进行同步。下表定义了CUDA C++中可用的线程作用域，以及可以与每个线程同步的线程。


| Thread Scope	| Description |
| ----| ----|
|cuda::thread_scope::thread_scope_thread|	Only the CUDA thread which initiated asynchronous operations synchronizes.|
|cuda::thread_scope::thread_scope_block	|All or any CUDA threads within the same thread block as the initiating thread synchronizes.|
|cuda::thread_scope::thread_scope_device|	All or any CUDA threads in the same GPU device as the initiating thread synchronizes.|
|cuda::thread_scope::thread_scope_system|	All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.|

这些线程作用域是在[CUDA 标准 C++库](https://nvidia.github.io/libcudacxx/extended_api/thread_scopes.html)中作为标准C++的扩展来实现的。

## 2.6 Compute Capability
设备的`Compute Capability`由版本号表示，有时也称其“ SM 版本”。该版本号标识 GPU 硬件支持的特性，并由应用程序在运行时使用，以确定当前GPU上可用的硬件特性和指令。

`Compute Capability`包括一个主要版本号X和一个次要版本号Y，用X.Y表示

主版本号相同的设备具有相同的核心架构。若设备的主要修订号是 8，则说明设备基于`NVIDIA Ampere GPU`的体系结构,7 为设备基于`Volta`设备架构,6为设备基于`Pascal`架构,5为设备基于`Maxwell`架构,3为设备基于`Kepler`架构的设备,2为设备基于`Fermi`架构,1为设备基于`Tesla`架构的设备。

次要修订号对应于对核心架构的增量改进，可能包括新特性。

`Turing`是计算能力7.5的设备架构，是基于Volta架构的增量更新。

[CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) 列出了所有支持 CUDA 的设备及其`Compute Capability`。[Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)给出了每个计算能力的技术规格。

#### 注意：特定 GPU 的 Compute Capability 版本不应与 CUDA 版本(如CUDA 7.5、CUDA 8、CUDA 9)混淆，CUDA 版本指的是 CUDA软件平台的版本。CUDA平台被应用开发人员用来创建那些可以运行在许多代GPU架构上的应用程序，包括未来尚未发明的 GPU架构。尽管CUDA平台的新版本通常会通过支持新GPU架构的Compute Capability版本来增加对于该架构的本地支持，但 CUDA平台的新版本通常也会包含软件功能，而这些是与硬件独立的。
从CUDA 7.0和CUDA 9.0开始，不再支持`Tesla`和`Fermi`架构。