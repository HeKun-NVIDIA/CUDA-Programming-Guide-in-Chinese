# 附录B 对C++扩展的详细描述

## B.1 函数执行空间说明符
函数执行空间说明符表示函数是在主机上执行还是在设备上执行，以及它是可从主机调用还是从设备调用。

### B.1.1 \_\_global\_\_
`__global__` 执行空间说明符将函数声明为内核。 它的功能是：

* 在设备上执行，
* 可从主机调用，
* 可在计算能力为 3.2 或更高的设备调用（有关更多详细信息，请参阅 [CUDA 动态并行性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)）。
`__global__` 函数必须具有 void 返回类型，并且不能是类的成员。

对 `__global__` 函数的任何调用都必须指定其执行配置，如[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)中所述。

对 `__global__` 函数的调用是异步的，这意味着它在设备完成执行之前返回。

### B.1.2 \_\_device\_\_
`__device__` 执行空间说明符声明了一个函数：

* 在设备上执行，
* 只能从设备调用。
`__global__` 和 `__device__` 执行空间说明符不能一起使用。

### B.1.3 \_\_host\_\_
`__host__` 执行空间说明符声明了一个函数：

* 在主机上执行，
* 只能从主机调用。
相当于声明一个函数只带有 `__host__` 执行空间说明符，或者声明它没有任何 `__host__` `、__device__` 或 `__global__` 执行空间说明符； 在任何一种情况下，该函数都仅为主机编译。

`__global__` 和 `__host__` 执行空间说明符不能一起使用。

但是， `__device__` 和 `__host__` 执行空间说明符可以一起使用，在这种情况下，该函数是为主机和设备编译的。 Application Compatibility 中引入的 `__CUDA_ARCH__ `宏可用于区分主机和设备之间的代码路径：
```C++
__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 800
   // Device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif __CUDA_ARCH__ >= 300
   // Device code path for compute capability 3.x
#elif !defined(__CUDA_ARCH__) 
   // Host code path
#endif
}
```

### B.1.4 Undefined behavior
在以下情况下，“跨执行空间”调用具有未定义的行为：
* `__CUDA_ARCH__` 定义了, 从 `__global__` 、 `__device__` 或 `__host__ __device__` 函数到 `__host__` 函数的调用。
* `__CUDA_ARCH__` 未定义，从 `__host__` 函数内部调用 `__device__` 函数。

#### B.1.5 `__noinline__` and `__forceinline__`

编译器在认为合适时内联任何 `__device__` 函数。

`__noinline__` 函数限定符可用作提示编译器尽可能不要内联函数。

`__forceinline__` 函数限定符可用于强制编译器内联函数。

`__noinline__` 和 `__forceinline__` 函数限定符不能一起使用，并且两个函数限定符都不能应用于内联函数。

## B.2 Variable Memory Space Specifiers

变量内存空间说明符表示变量在设备上的内存位置。

在设备代码中声明的没有本节中描述的任何 `__device__`、`__shared__` 和 `__constant__` 内存空间说明符的自动变量通常驻留在寄存器中。 但是，在某些情况下，编译器可能会选择将其放置在本地内存中，这可能会产生不利的性能后果，如[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中所述。

### B.2.1 \_\_device\_\_
`__device__` 内存空间说明符声明了一个驻留在设备上的变量。

在接下来的三个部分中定义的其他内存空间说明符中最多有一个可以与 `__device__` 一起使用，以进一步表示变量属于哪个内存空间。 如果它们都不存在，则变量：

* 驻留在全局内存空间中，
* 具有创建它的 CUDA 上下文的生命周期，
* 每个设备都有一个不同的对象，
* 可从网格内的所有线程和主机通过运行时库 (`cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()`) 访问。

### B.2.2. \_\_constant\_\_
`__constant__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 驻留在常量的内存空间中，
* 具有创建它的 CUDA 上下文的生命周期，
* 每个设备都有一个不同的对象，
* 可从网格内的所有线程和主机通过运行时库 (`cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()`) 访问。

### B.2.3 \_\_shared\_\_

`__shared__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 驻留在线程块的共享内存空间中，
* 具有块的生命周期，
* 每个块有一个不同的对象，
* 只能从块内的所有线程访问，
* 没有固定地址。

将共享内存中的变量声明为外部数组时，例如:
```C++
extern __shared__ float shared[];
```
数组的大小在启动时确定（请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)）。 以这种方式声明的所有变量都从内存中的相同地址开始，因此必须通过偏移量显式管理数组中变量的布局。 例如，如果想要在动态分配的共享内存中等价于，
```C++
short array0[128];
float array1[64];
int   array2[256];
```
可以通过以下方式声明和初始化数组：
```C++
extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array; 
    float* array1 = (float*)&array0[128];
    int*   array2 =   (int*)&array1[64];
}
```
#### 请注意，指针需要与它们指向的类型对齐，因此以下代码不起作用，因为 array1 未对齐到 4 个字节。
```C++
extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array; 
    float* array1 = (float*)&array0[127];
}
```
[表 4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types__alignment-requirements-in-device-code) 列出了内置向量类型的对齐要求。

### B.2.4. __managed__
`__managed__` 内存空间说明符，可选择与 `__device__` 一起使用，声明一个变量：

* 可以从设备和主机代码中引用，例如，可以获取其地址，也可以直接从设备或主机功能读取或写入。
* 具有应用程序的生命周期。
有关更多详细信息，请参阅 [`__managed__` 内存空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#managed-specifier)说明符。

### B.2.5. __restrict__
nvcc 通过 `__restrict__` 关键字支持受限指针。

C99中引入了受限指针，以缓解存在于c类型语言中的混叠问题，这种问题抑制了从代码重新排序到公共子表达式消除等各种优化。

下面是一个受混叠问题影响的例子，使用受限指针可以帮助编译器减少指令的数量：
```C++
void foo(const float* a,
         const float* b,
         float* c)
{
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}
```

此处的效果是减少了内存访问次数和减少了计算次数。 这通过由于“缓存”负载和常见子表达式而增加的寄存器压力来平衡。

由于寄存器压力在许多 CUDA 代码中是一个关键问题，因此由于占用率降低，使用受限指针会对 CUDA 代码产生负面性能影响。

## B.3. Built-in Vector Types

### B.3.1. char, short, int, long, longlong, float, double
这些是从基本整数和浮点类型派生的向量类型。 它们是结构，第一个、第二个、第三个和第四个组件可以分别通过字段 `x、y、z 和 w` 访问。 它们都带有 `make_<type name> `形式的构造函数； 例如，
```C++
int2 make_int2(int x, int y);
```
它创建了一个带有 `value(x, y)` 的 `int2` 类型的向量。
向量类型的对齐要求在下表中有详细说明。

|Type|	Alignment|
|----|----|
|char1, uchar1|	1|
|char2, uchar2|	2|
|char3, uchar3|	1|
|char4, uchar4	|4|
|short1, ushort1|	2|
|short2, ushort2|	4|
|short3, ushort3|	2|
|short4, ushort4|	8|
|int1, uint1	|4|
|int2, uint2	|8|
|int3, uint3	|4|
|int4, uint4|	16|
|long1, ulong1|	4 if sizeof(long) is equal to sizeof(int) 8, otherwise|
|long2, ulong2|	8 if sizeof(long) is equal to sizeof(int), 16, otherwise|
|long3, ulong3|	4 if sizeof(long) is equal to sizeof(int), 8, otherwise|
|long4, ulong4	|16|
|longlong1, ulonglong1|	8|
|longlong2, ulonglong2	|16|
|longlong3, ulonglong3|	8|
|longlong4, ulonglong4	|16|
|float1	|4|
|float2	|8|
|float3	|4|
|float4	|16|
|double1	|8|
|double2	|16|
|double3	|8|
|double4	|16|

### B.3.2. dim3
此类型是基于 uint3 的整数向量类型，用于指定维度。 定义 dim3 类型的变量时，任何未指定的组件都将初始化为 1。

## B.4. Built-in Variables

### B.4.1. gridDim
该变量的类型为 `dim3`（请参阅[ dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)）并包含网格的尺寸。

### B.4.2. blockIdx
该变量是 `uint3` 类型（请参见 [char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types)）并包含网格内的块索引。

### B.4.3. blockDim
该变量的类型为 `dim3`（请参阅 [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3)）并包含块的尺寸。

### B.4.4. threadIdx
此变量是 `uint3` 类型（请参见 [char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types) ）并包含块内的线程索引。

### B.4.5. warpSize
该变量是 `int` 类型，包含线程中的 `warp` 大小（有关 `warp` 的定义，请参见 [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture)）。


## B.5. Memory Fence Functions
CUDA 编程模型假设设备具有弱序内存模型，即 CUDA 线程将数据写入共享内存、全局内存、页面锁定主机内存或对等设备的内存的顺序不一定是 观察到数据被另一个 CUDA 或主机线程写入的顺序。 两个线程在没有同步的情况下读取或写入同一内存位置是未定义的行为。

在以下示例中，thread 1 执行 writeXY()，而thread 2 执行 readXY()。

```C++
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    int A = X;
}
```

两个线程同时从相同的内存位置 X 和 Y 读取和写入。 任何数据竞争都是未定义的行为，并且没有定义的语义。 A 和 B 的结果值可以是任何值。

内存栅栏函数可用于强制对内存访问进行一些排序。 内存栅栏功能在强制执行排序的范围上有所不同，但它们独立于访问的内存空间（共享内存、全局内存、页面锁定的主机内存和对等设备的内存）。
```C++
void __threadfence_block();
```
#### 请确保：
* 线程在调用 __threadfence_block() 之前对所有内存的所有写入都被线程的块中的所有线程观察到. 这发生在调用线程在调用 __threadfence_block() 之后对内存的所有写入之前；
* 线程在调用 __threadfence_block() 之前对所有内存进行的所有读取都排在线程在调用 __threadfence_block() 之后对所有内存的所有读取之前。

```C++
void __threadfence();
```
充当调用线程块中所有线程的 `__threadfence_block()` 并且还确保在调用 `__threadfence() `之后调用线程对所有内存的写入不会被设备中的任何线程观察到在任何写入之前发生 调用线程在调用 __threadfence() 之前产生的所有内存。 请注意，要使这种排序保证为真，观察线程必须真正观察内存而不是它的缓存版本； 这可以通过使用 [volatile 限定符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier)中详述的 volatile 关键字来确保。

```C++
void __threadfence_system()
```
充当调用线程块中所有线程的 `__threadfence_block()`，并确保设备中的所有线程、主机线程和所有线程在调用 `__threadfence_system()` 之前对调用线程所做的所有内存的所有写入都被观察到 对等设备中的线程在调用 `__threadfence_system()` 之后调用线程对所有内存的所有写入之前发生。

`__threadfence_system()` 仅受计算能力 2.x 及更高版本的设备支持。

在前面的代码示例中，我们可以在代码中插入栅栏，如下所示：
```C++
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```
对于此代码，可以观察到以下结果：
* A 等于 1，B 等于 2，
* A 等于 10，B 等于 2，
* A 等于 10，B 等于 20。

第四种结果是不可能的，因为第一次写入必须在第二次写入之前可见。 如果线程 1 和 2 属于同一个块，使用 __threadfence_block() 就足够了。 如果线程 1 和 2 不属于同一个块，如果它们是来自同一设备的 CUDA 线程，则必须使用 __threadfence()，如果它们是来自两个不同设备的 CUDA 线程，则必须使用 __threadfence_system()。

一个常见的用例是当线程消耗由其他线程产生的一些数据时，如以下内核代码示例所示，该内核在一次调用中计算 N 个数字的数组的总和。 每个块首先对数组的一个子集求和，并将结果存储在全局内存中。 当所有块都完成后，最后一个完成的块从全局内存中读取这些部分和中的每一个，并将它们相加以获得最终结果。 为了确定哪个块最后完成，每个块自动递增一个计数器以表示它已完成计算和存储其部分和（请参阅[原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)关于原子函数）。 最后一个块是接收等于 `gridDim.x-1` 的计数器值的块。 如果在存储部分和和递增计数器之间没有设置栅栏，则计数器可能会在存储部分和之前递增，因此可能会到达 gridDim.x-1 并让最后一个块在实际更新之前在Global Memory中开始读取部分和 。

#### 作者添加: 开发者指南中原文介绍threadfence的时候,比较长比较绕,可能对于新手开发朋友来说比较难理解.作者觉得,可以简单的理解为一种等待行为.让Warp中线程运行到threadfence这里等一下, 不然可能产生上面的还没写完,下面的就开始读的问题. 这种写后读,可能会读到错误的数据.

内存栅栏函数只影响线程内存操作的顺序； 它们不确保这些内存操作对其他线程可见（就像 `__syncthreads()` 对块内的线程所做的那样（请参阅[同步函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)））。 在下面的代码示例中，通过将结果变量声明为volatile 来确保对结果变量的内存操作的可见性（请参阅[volatile 限定符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier)）。
```C++
__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__global__ void sum(const float* array, unsigned int N,
                    volatile float* result)
{
    // Each block sums a subset of the input array.
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {

        // Thread 0 of each block stores the partial sum
        // to global memory. The compiler will use 
        // a store operation that bypasses the L1 cache
        // since the "result" variable is declared as
        // volatile. This ensures that the threads of
        // the last block will read the correct partial
        // sums computed by all other blocks.
        result[blockIdx.x] = partialSum;

        // Thread 0 makes sure that the incrementation
        // of the "count" variable is only performed after
        // the partial sum has been written to global memory.
        __threadfence();

        // Thread 0 signals that it is done.
        unsigned int value = atomicInc(&count, gridDim.x);

        // Thread 0 determines if its block is the last
        // block to be done.
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    // Synchronize to make sure that each thread reads
    // the correct value of isLastBlockDone.
    __syncthreads();

    if (isLastBlockDone) {

        // The last block sums the partial sums
        // stored in result[0 .. gridDim.x-1]
        float totalSum = calculateTotalSum(result);

        if (threadIdx.x == 0) {

            // Thread 0 of last block stores the total sum
            // to global memory and resets the count
            // varialble, so that the next kernel call
            // works properly.
            result[0] = totalSum;
            count = 0;
        }
    }
}
```

## B.6. Synchronization Functions

```C
void __syncthreads();
```
等待直到线程块中的所有线程都达到这一点，并且这些线程在 `__syncthreads()` 之前进行的所有全局和共享内存访问对块中的所有线程都是可见的。

`__syncthreads()` 用于协调同一块的线程之间的通信。 当块中的某些线程访问共享或全局内存中的相同地址时，对于其中一些内存访问，可能存在先读后写、先读后写或先写后写的风险。 通过在这些访问之间同步线程可以避免这些数据危害。

`__syncthreads()` 允许在条件代码中使用，但前提是条件在整个线程块中的计算结果相同，否则代码执行可能会挂起或产生意外的副作用。

计算能力 2.x 及更高版本的设备支持以下描述的三种 __syncthreads() 变体。

`int __syncthreads_count(int predicate)`与 __syncthreads() 相同，其附加功能是它为块的所有线程评估predicate并返回predicate评估为非零的线程数。

`int __syncthreads_and(int predicate)` 与 __syncthreads() 相同，其附加功能是它为块的所有线程计算predicate，并且当且仅当predicate对所有线程的计算结果都为非零时才返回非零。

`int __syncthreads_or(int predicate)` 与 __syncthreads() 相同，其附加功能是它为块的所有线程评估predicate，并且当且仅当predicate对其中任何一个线程评估为非零时才返回非零。

`void __syncwarp(unsigned mask=0xffffffff)` 将导致正在执行的线程等待，直到 mask 中命名的所有 warp 通道都执行了 __syncwarp()（具有相同的掩码），然后再恢复执行。 掩码中命名的所有未退出线程必须执行具有相同掩码的相应 __syncwarp()，否则结果未定义。

执行 __syncwarp() 保证参与屏障的线程之间的内存排序。 因此，warp 中希望通过内存进行通信的线程可以存储到内存，执行 __syncwarp()，然后安全地读取 warp 中其他线程存储的值。

#### 注意：对于 .target sm_6x 或更低版本，mask 中的所有线程在收敛时必须执行相同的 __syncwarp()，并且 mask 中所有值的并集必须等于活动掩码。 否则，行为未定义。

## B.7. Mathematical Functions
参考手册列出了设备代码支持的所有 C/C++ 标准库数学函数和仅设备代码支持的所有内部函数。

[数学函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix)为其中一些函数提供精度信息。

## B.8. Texture Functions

纹理对象在 [Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api) 中描述

纹理引用在 [[[DEPRECATED]] 纹理引用 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-reference-api) 中描述

纹理提取在[纹理提取](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching)中进行了描述。

### B.8.1. Texture Object API
#### B.8.1.1. tex1Dfetch()
```C
template<class T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);
```
从使用整数纹理坐标 x 的一维纹理对象 texObj 指定的线性内存区域中获取。 tex1Dfetch() 仅适用于非归一化坐标，因此仅支持边界和钳位寻址模式。 它不执行任何纹理过滤。 对于整数类型，它可以选择将整数提升为单精度浮点数。

#### B.8.1.2。 tex1D()
```C
template<class T>
T tex1D(cudaTextureObject_t texObj, float x);
```
从使用纹理坐标 x 的一维纹理对象 texObj 指定的 CUDA 数组中获取。

#### B.8.1.3。 tex1DLod()
```C
template<class T>
T tex1DLod(cudaTextureObject_t texObj, float x, float level);
```
使用细节级别的纹理坐标 x 从一维纹理对象 texObj 指定的 CUDA 数组中获取。

#### B.8.1.4。 tex1DGrad()
```C
template<class T>
T tex1DGrad(cudaTextureObject_t texObj, float x, float dx, float dy);
```
从使用纹理坐标 x 的一维纹理对象 texObj 指定的 CUDA 数组中获取。细节层次来源于 X 梯度 dx 和 Y 梯度 dy。

#### B.8.1.5。 tex2D()
```C
template<class T>
T tex2D(cudaTextureObject_t texObj, 浮点 x, 浮点 y);
```
从 CUDA 数组或由二维纹理对象 texObj 使用纹理坐标 (x,y) 指定的线性内存区域获取。

#### B.8.1.6。 tex2DLod()
```C
template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level);
```
从 CUDA 数组或二维纹理对象 texObj 指定的线性内存区域中获取，使用细节级别的纹理坐标 (x,y)。

#### B.8.1.7。 tex2DGrad()
```C++
template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
            float2 dx，float2 dy）；
```
使用纹理坐标 (x,y) 从二维纹理对象 texObj 指定的 CUDA 数组中获取。细节层次来源于 dx 和 dy 梯度。

#### B.8.1.8。 tex3D()
```C++
template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z);
```
使用纹理坐标 (x,y,z) 从三维纹理对象 texObj 指定的 CUDA 数组中获取。

#### B.8.1.9。 tex3DLod()
```C++
template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level);
```
使用细节级别的纹理坐标 `(x,y,z) `从 CUDA 数组或由三维纹理对象 `texObj` 指定的线性内存区域获取。

#### B.8.1.10。 tex3DGrad()
```C++
template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
            float4 dx，float4 dy）；
```
从由三维纹理对象 `texObj` 指定的 CUDA 数组中获取，使用纹理坐标 (x,y,z) 在从 `X` 和 `Y` 梯度 `dx` 和 `dy` 派生的细节级别。

#### B.8.1.11。 tex1DLlayered()
```C++
template<class T>
T tex1DLayered(cudaTextureObject_t texObj, float x, int layer);
```
使用纹理坐标 `x `和索`layer`从一维纹理对象 `texObj` 指定的 CUDA 数组中获取，如分层纹理中所述

#### B.8.1.12。 tex1DLlayeredLod()
```C++
template<class T>
T tex1DLayeredLod(cudaTextureObject_t texObj, float x, int layer, float level);
```
从使用纹理坐标 `x` 和细节级别级别的图层 `layer` 的一维分层纹理指定的 CUDA 数组中获取。

#### B.8.1.13。 tex1DLlayeredGrad()
```C++
template<class T>
T tex1DLayeredGrad(cudaTextureObject_t texObj, float x, int layer,
                   float dx, float dy);
```
使用纹理坐标 `x` 和从 `dx` 和 `dy` 梯度派生的细节层次从 `layer` 的一维分层纹理指定的 CUDA 数组中获取。

#### B.8.1.14。 tex2DLlayered()
```C++
template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
               float float y，int layer）；
```
使用纹理坐标 `(x,y)` 和索引层从二维纹理对象 `texObj` 指定的 CUDA 数组中获取，如分层纹理中所述。

#### B.8.1.15。 tex2DLlayeredLod()
```C++
template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                  float level）；
```
使用纹理坐标 `(x,y)` 从 `layer`  的二维分层纹理指定的 CUDA 数组中获取。

#### B.8.1.16。 tex2DLlayeredGrad()
```C++
template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                   float2 dx，float2 dy）；
```
使用纹理坐标 `(x,y)` 和从 `dx` 和 `dy`  梯度派生的细节层次从 layer  的二维分层纹理指定的 CUDA 数组中获取。

#### B.8.1.17。 texCubemap()
```C++
template<class T>
T texCubemap(cudaTextureObject_t texObj, float x, float y, float z);
```
使用纹理坐标 `(x,y,z)` 获取由立方体纹理对象 `texObj` 指定的 CUDA 数组，如[立方体纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-textures)中所述。

#### B.8.1.18。 texCubemapLod()
```C++
template<class T>
T texCubemapLod(cudaTextureObject_t texObj, float x, float, y, float z,
                float level）；
```
使用[立方体纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-textures)中描述的纹理坐标 (x,y,z) 从立方体纹理对象 `texObj` 指定的 CUDA 数组中获取。使用的详细级别由`level`给出。

#### B.8.1.19。 texCubemapLayered()
```C++
template<class T>
T texCubemapLayered(cudaTextureObject_t texObj,
                    float x，float y，float z，int layer）；
```
使用纹理坐标 (x,y,z) 和索引层从立方体分层纹理对象 `texObj` 指定的 CUDA 数组中获取，如[立方体分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-layered-textures)中所述。

#### B.8.1.20。 texCubemapLayeredLod()
```C++
template<class T>
T texCubemapLayeredLod(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer，float level）；
```
使用纹理坐标 (x,y,z) 和索引层从立方体分层纹理对象 `texObj` 指定的 CUDA 数组中获取，如[立方体分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-layered-textures)中所述，在细节级别级别。

#### B.8.1.21。 tex2Dgather()
```C++
template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
              float x，float y，int comp = 0);
```
从 2D 纹理对象 `texObj` 指定的 CUDA 数组中获取，使用纹理坐标 x 和 y 以及[纹理采集](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-gather)中描述的 `comp` 参数。


### B.8.2. Texture Reference API

#### B.8.2.1. tex1Dfetch()

```C++
template<class DataType>
Type tex1Dfetch(
   texture<DataType, cudaTextureType1D,
           cudaReadModeElementType> texRef,
   int x);

float tex1Dfetch(
   texture<unsigned char, cudaTextureType1D,
           cudaReadModeNormalizedFloat> texRef,
   int x);

float tex1Dfetch(
   texture<signed char, cudaTextureType1D,
           cudaReadModeNormalizedFloat> texRef,
   int x);

float tex1Dfetch(
   texture<unsigned short, cudaTextureType1D,
           cudaReadModeNormalizedFloat> texRef,
   int x);

float tex1Dfetch(
   texture<signed short, cudaTextureType1D,
           cudaReadModeNormalizedFloat> texRef,
   int x);
```
使用整数纹理坐标 x 从绑定到一维纹理引用 `texRef` 的线性内存区域中获取。 `tex1Dfetch()` 仅适用于非归一化坐标，因此仅支持边界和钳位寻址模式。 它不执行任何纹理过滤。 对于整数类型，它可以选择将整数提升为单精度浮点数。

除了上面显示的功能外，还支持 2 元组和 4 元组； 例如：
```C++
float4 tex1Dfetch(
   texture<uchar4, cudaTextureType1D,
           cudaReadModeNormalizedFloat> texRef,
   int x);
```

#### 作者添加: 因为这里的纹理引用API在当前版本被弃用,所以这里细节不再做过多描述.


## B.9. Surface Functions
Surface 函数仅受计算能力 2.0 及更高版本的设备支持。

Surface 对象在 [Surface Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api-appendix) 中描述

Surface引用在[Surface引用 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-reference-api-appendix) 中描述。

在下面的部分中，`boundaryMode` 指定了边界模式，即如何处理超出范围的表面坐标； 它等于 `cudaBoundaryModeClamp`，在这种情况下，超出范围的坐标被限制到有效范围，或 `cudaBoundaryModeZero`，在这种情况下，超出范围的读取返回零并且忽略超出范围的写入，或者 `cudaBoundaryModeTrap`， 在这种情况下，超出范围的访问会导致内核执行失败。

### B.9.1. Surface Object API
#### B.9.1.1. surf1Dread()
```C++
template<class T>
T surf1Dread(cudaSurfaceObject_t surfObj, int x,
               boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 读取由一维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.2. surf1Dwrite
```C++
template<class T>
void surf1Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x,
                  boundaryMode = cudaBoundaryModeTrap);
```
将数据写入由坐标 x 处的一维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.3. surf2Dread()
```C++
template<class T>
T surf2Dread(cudaSurfaceObject_t surfObj,
              int x, int y,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 读取二维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.4 surf2Dwrite()
```C++
template<class T>
void surf2Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y,
                  boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入由坐标 x 和 y 处的二维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.5. surf3Dread()
```C++
template<class T>
T surf3Dread(cudaSurfaceObject_t surfObj,
              int x, int y, int z,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf3Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x、y 和 z 读取由三维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.6. surf3Dwrite()
```C++
template<class T>
void surf3Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y, int z,
                  boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入由坐标 x、y 和 z 处的三维surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.7. surf1DLayeredread()
```C++
template<class T>
T surf1DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf1DLayeredread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和索引层读取一维分层surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.8. surf1DLayeredwrite()
```C++
template<class Type>
void surf1DLayeredwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入坐标 x 和索引层的二维分层surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.9. surf2DLayeredread()
```C++
template<class T>
T surf2DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2DLayeredread(T data,
                         cudaSurfaceObject_t surfObj,
                         int x, int y, int layer,	
                         boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及索引层读取二维分层surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.10. surf2DLayeredwrite()
```C++
template<class T>
void surf2DLayeredwrite(T data,
                          cudaSurfaceObject_t surfObj,
                          int x, int y, int layer,
                          boundaryMode = cudaBoundaryModeTrap);
```
将数据写入由坐标 x 和 y 处的一维分层surface对象 `surfObj` 和索引层指定的 CUDA 数组。

#### B.9.1.11. surfCubemapread()
```C++
template<class T>
T surfCubemapread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及面索引 face 读取立方体surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.12. surfCubemapwrite()
```C++
template<class T>
void surfCubemapwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
```
将数据写入由立方体对象 surfObj 在坐标 x 和 y 以及面索引 face 处指定的 CUDA 数组。 

#### B.9.1.13. surfCubemapLayeredread()
```C++
template<class T>
T surfCubemapLayeredread(
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapLayeredread(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及索引 `layerFace` 读取由立方体分层surface对象 `surfObj` 指定的 CUDA 数组。

#### B.9.1.14. surfCubemapLayeredwrite()
```C++
template<class T>
void surfCubemapLayeredwrite(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```
将数据写入由立方体分层对象 `surfObj` 在坐标 x 和 y 以及索引 layerFace 指定的 CUDA 数组。

### B.9.2. Surface Reference API
#### B.9.2.1. surf1Dread()
```C++
template<class Type>
Type surf1Dread(surface<void, cudaSurfaceType1D> surfRef,
                int x,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surf1Dread(Type data,
                surface<void, cudaSurfaceType1D> surfRef,
                int x,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 读取绑定到一维surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.2. surf1Dwrite
```C++
template<class Type>
void surf1Dwrite(Type data,
                 surface<void, cudaSurfaceType1D> surfRef,
                 int x,
                 boundaryMode = cudaBoundaryModeTrap);
```

#### B.9.2.3. surf2Dread()
```C++
template<class Type>
Type surf2Dread(surface<void, cudaSurfaceType2D> surfRef,
                int x, int y,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surf2Dread(Type* data,
                surface<void, cudaSurfaceType2D> surfRef,
                int x, int y,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 读取绑定到二维surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.4. surf2Dwrite()
```C++
template<class Type>
void surf3Dwrite(Type data,
                 surface<void, cudaSurfaceType3D> surfRef,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入绑定到坐标 x 和 y 处的二维surface引用 `surfRef` 的 CUDA 数组。 

#### B.9.2.5. surf3Dread()
```C++
template<class Type>
Type surf3Dread(surface<void, cudaSurfaceType3D> surfRef,
                int x, int y, int z,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surf3Dread(Type* data,
                surface<void, cudaSurfaceType3D> surfRef,
                int x, int y, int z,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x、y 和 z 读取绑定到三维surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.6. surf3Dwrite()
```C++
template<class Type>
void surf3Dwrite(Type data,
                 surface<void, cudaSurfaceType3D> surfRef,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);
```
将数据写入绑定到坐标 x、y 和 z 处的surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.7. surf1DLayeredread()
```C++
template<class Type>
Type surf1DLayeredread(
                surface<void, cudaSurfaceType1DLayered> surfRef,
                int x, int layer,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surf1DLayeredread(Type data,
                surface<void, cudaSurfaceType1DLayered> surfRef,
                int x, int layer,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和索引层读取绑定到一维分层surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.8. surf1DLayeredwrite()
```C++
template<class Type>
void surf1DLayeredwrite(Type data,
                surface<void, cudaSurfaceType1DLayered> surfRef,
                int x, int layer,
                boundaryMode = cudaBoundaryModeTrap);
```
将数据写入绑定到坐标 x 和索引层的二维分层surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.9. surf2DLayeredread()
```C++
template<class Type>
Type surf2DLayeredread(
                surface<void, cudaSurfaceType2DLayered> surfRef,
                int x, int y, int layer,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surf2DLayeredread(Type data,
                surface<void, cudaSurfaceType2DLayered> surfRef,
                int x, int y, int layer,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及索引层读取绑定到二维分层surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.10. surf2DLayeredwrite()
```C++
template<class Type>
void surf2DLayeredwrite(Type data,
                surface<void, cudaSurfaceType2DLayered> surfRef,
                int x, int y, int layer,
                boundaryMode = cudaBoundaryModeTrap);
```
将数据写入绑定到坐标 x 和 y 处的一维分层surface引用 `surfRef` 和索引层的 CUDA 数组。

#### B.9.2.11. surfCubemapread()
```C++
template<class Type>
Type surfCubemapread(
                surface<void, cudaSurfaceTypeCubemap> surfRef,
                int x, int y, int face,
                boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surfCubemapread(Type data,
                surface<void, cudaSurfaceTypeCubemap> surfRef,
                int x, int y, int face,
                boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及面索引 `face` 读取绑定到立方体surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.12. surfCubemapwrite()
```C++
template<class Type>
void surfCubemapwrite(Type data,
                surface<void, cudaSurfaceTypeCubemap> surfRef,
                int x, int y, int face,
                boundaryMode = cudaBoundaryModeTrap);
```
将数据写入绑定到位于坐标 x , y 和面索引 face 处的立方体引用 `surfRef` 的 CUDA 数组。

B.9.2.13. surfCubemapLayeredread()
```C++
template<class Type>
Type surfCubemapLayeredread(
            surface<void, cudaSurfaceTypeCubemapLayered> surfRef,
            int x, int y, int layerFace,
            boundaryMode = cudaBoundaryModeTrap);
template<class Type>
void surfCubemapLayeredread(Type data,
            surface<void, cudaSurfaceTypeCubemapLayered> surfRef,
            int x, int y, int layerFace,
            boundaryMode = cudaBoundaryModeTrap);
```
使用坐标 x 和 y 以及索引 layerFace 读取绑定到立方体分层surface引用 `surfRef` 的 CUDA 数组。

#### B.9.2.14. surfCubemapLayeredwrite()
```C++
template<class Type>
void surfCubemapLayeredwrite(Type data,
            surface<void, cudaSurfaceTypeCubemapLayered> surfRef,
            int x, int y, int layerFace,
            boundaryMode = cudaBoundaryModeTrap);
```
将数据写入绑定到位于坐标 x , y 和索引 layerFace处的立方体分层引用 `surfRef`  的 CUDA 数组。

