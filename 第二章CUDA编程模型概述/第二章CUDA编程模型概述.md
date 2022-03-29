# 2.编程模型
#### 本章通过概述CUDA编程模型是如何在c++中公开的，来介绍CUDA的主要概念。
#### [编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中给出了对 CUDA C++ 的广泛描述。

#### 本章和下一章中使用的向量加法示例的完整代码可以在 vectorAdd [CUDA示例](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition)中找到。

## 2.1 内核
CUDA C++ 通过允许程序员定义称为kernel的 C++ 函数来扩展 C++，当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

使用 `__global__` 声明说明符定义内核，并使用新的 `<<<...>>>` 执行配置语法指定内核调用的 CUDA 线程数（请参阅 [C++ 语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)）。 每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问。

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
为方便起见，threadIdx 是一个 3 分量向量，因此可以使用一维、二维或三维的线程索引来识别线程，形成一个一维、二维或三维的线程块，称为block。 这提供了一种跨域的元素（例如向量、矩阵或体积）调用计算的方法。

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
每个块的线程数量是有限制的，因为一个块的所有线程都应该驻留在同一个处理器核心上，并且必须共享该核心有限的内存资源。在当前的gpu上，一个线程块可能包含多达1024个线程。

但是，一个内核可以由多个形状相同的线程块执行，因此线程总数等于每个块的线程数乘以块数。

块被组织成一维、二维或三维的线程块网格(`grid`)，如下图所示。网格中的线程块数量通常由正在处理的数据的大小决定，通常超过系统中的处理器数量。
![grid-of-thread-blocks.png](grid-of-thread-blocks.png)

```<<<...>>> ```语法中指定的每个块的线程数和每个网格的块数可以是 ```int``` 或 `dim3` 类型。如上例所示，可以指定二维块或网格。

网格中的每个块都可以由一个一维、二维或三维的惟一索引标识，该索引可以通过内置的`blockIdx`变量在内核中访问。线程块的维度可以通过内置的`blockDim`变量在内核中访问。

扩展前面的`MatAdd()`示例来处理多个块，代码如下所示。
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
线程块大小为16x16(256个线程)，尽管在本例中是任意更改的，但这是一种常见的选择。网格是用足够的块创建的，这样每个矩阵元素就有一个线程来处理。为简单起见，本例假设每个维度中每个网格的线程数可以被该维度中每个块的线程数整除，尽管事实并非如此。

程块需要独立执行：必须可以以任何顺序执行它们，并行或串行。 这种独立性要求允许跨任意数量的内核以任意顺序调度线程块，如下图所示，使程序员能够编写随内核数量扩展的代码。
![automatic-scalability.png](automatic-scalability.png)

块内的线程可以通过一些共享内存共享数据并通过同步它们的执行来协调内存访问来进行协作。 更准确地说，可以通过调用 `__syncthreads()` 内部函数来指定内核中的同步点； `__syncthreads()` 充当屏障，块中的所有线程必须等待，然后才能继续。 [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) 给出了一个使用共享内存的例子。 除了` __syncthreads()` 之外，[Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) 还提供了一组丰富的线程同步示例。

为了高效协作，共享内存是每个处理器内核附近的低延迟内存（很像 L1 缓存），并且 `__syncthreads()` 是轻量级的。

## 2.3 存储单元层次
CUDA 线程可以在执行期间从多个内存空间访问数据，如下图所示。每个线程都有私有的本地内存。 每个线程块都具有对该块的所有线程可见的共享内存，并且具有与该块相同的生命周期。 所有线程都可以访问相同的全局内存。

![memory-hierarchy.png](memory-hierarchy.png)

还有两个额外的只读内存空间可供所有线程访问：常量和纹理内存空间。 全局、常量和纹理内存空间针对不同的内存使用进行了优化（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。 纹理内存还为某些特定数据格式提供不同的寻址模式以及数据过滤（请参阅[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)）。

全局、常量和纹理内存空间在同一应用程序的内核启动中是持久的。

## 2.4 异构编程
如下图所示，CUDA 编程模型假定 CUDA 线程在物理独立的设备上执行，该设备作为运行 C++ 程序的主机的协处理器运行。例如，当内核在 GPU 上执行而 C++ 程序的其余部分在 CPU 上执行时，就是这种情况。

![heterogeneous-programming.png](heterogeneous-programming.png)

CUDA 编程模型还假设主机(`host`)和设备(`device`)都在 DRAM 中维护自己独立的内存空间，分别称为主机内存和设备内存。因此，程序通过调用 CUDA 运行时（在[编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中描述）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存分配和释放以及主机和设备内存之间的数据传输。

统一内存提供托管内存来桥接主机和设备内存空间。托管内存可从系统中的所有 CPU 和 GPU 访问，作为具有公共地址空间的单个连贯内存映像。此功能可实现设备内存的超额订阅，并且无需在主机和设备上显式镜像数据，从而大大简化了移植应用程序的任务。有关统一内存的介绍，请参阅统一[内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)。

#### 注:串行代码在主机(`host`)上执行，并行代码在设备(`device`)上执行。