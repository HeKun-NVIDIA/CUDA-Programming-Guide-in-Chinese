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

## B.10. Read-Only Data Cache Load Function
只读数据缓存加载功能仅支持计算能力3.5及以上的设备。
```C++
T __ldg(const T* address);
```
返回位于地址`address`的 T 类型数据，其中 T 为 `char、signed char、short、int、long、long longunsigned char、unsigned short、unsigned int、unsigned long、unsigned long long、char2、char4、short2、short4、 int2、int4、longlong2uchar2、uchar4、ushort2、ushort4、uint2、uint4、ulonglong2float、float2、float4、double` 或 `double2`. 包含 `cuda_fp16.h` 头文件，T 可以是 `__half` 或 `__half2`。 同样，包含 cuda_bf16.h 头文件后，T 也可以是 `__nv_bfloat16` 或 `__nv_bfloat162`。 该操作缓存在只读数据缓存中（请参阅[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0)）。

## B.11. Load Functions Using Cache Hints
这些加载功能仅受计算能力 3.5 及更高版本的设备支持。
```C++
T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);
```
返回位于地址`address`的 T 类型数据，其中 T 为 `char、signed char、short、int、long、long longunsigned char、unsigned short、unsigned int、unsigned long、unsigned long long、char2、char4、short2、short4、 int2、int4、longlong2uchar2、uchar4、ushort2、ushort4、uint2、uint4、ulonglong2float、float2、float4、double 或 double2`。 包含 `cuda_fp16.h` 头文件，T 可以是 `__half` 或 `__half2`。 同样，包含 cuda_bf16.h 头文件后，T 也可以是 `__nv_bfloat16` 或 `__nv_bfloat162`。 该操作正在使用相应的缓存运算符（请参阅 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)）

## B.12. Store Functions Using Cache Hints
这些存储功能仅受计算能力 3.5 及更高版本的设备支持。
```C++
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```
将类型 T 的`value`参数存储到地址 `address` 的位置，其中 T 是 `char、signed char、short、int、long、long longunsigned char、unsigned short、unsigned int、unsigned long、unsigned long long、char2、char4、short2 、short4、int2、int4、longlong2uchar2、uchar4、ushort2、ushort4、uint2、uint4、ulonglong2float、float2、float4、double 或 double2`。 包含 `cuda_fp16.h` 头文件，T 可以是 `__half` 或 `__half2`。 同样，包含 cuda_bf16.h 头文件后，T 也可以是 `__nv_bfloat16` 或 `__nv_bfloat162`。 该操作正在使用相应的缓存运算符（请参阅 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)）

## B.13. Time Function
```C++
clock_t clock();
long long int clock64();
```
在设备代码中执行时，返回每个时钟周期递增的每个多处理器计数器的值。 在内核开始和结束时对该计数器进行采样，获取两个样本的差异，并记录每个线程的结果，为每个线程提供设备完全执行线程所花费的时钟周期数的度量， 但不是设备实际执行线程指令所花费的时钟周期数。 前一个数字大于后者，因为线程是时间切片的。

## B.14. Atomic Functions
原子函数对驻留在全局或共享内存中的一个 32 位或 64 位字执行读-修改-写原子操作。 例如，`atomicAdd()` 在全局或共享内存中的某个地址读取一个字，向其中加一个数字，然后将结果写回同一地址。 该操作是原子的，因为它保证在不受其他线程干扰的情况下执行。 换句话说，在操作完成之前，没有其他线程可以访问该地址。 原子函数不充当内存栅栏，也不意味着内存操作的同步或排序约束（有关内存栅栏的更多详细信息，请参阅[内存栅栏函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）。 原子函数只能在设备函数中使用。

原子函数仅相对于特定集合的线程执行的其他操作是原子的：

* 系统范围的原子：当前程序中所有线程的原子操作，包括系统中的其他 CPU 和 GPU。 这些以 `_system` 为后缀，例如 `atomicAdd_system`。
* 设备范围的原子：当前程序中所有 CUDA 线程的原子操作，在与当前线程相同的计算设备中执行。 这些没有后缀，只是以操作命名，例如 `atomicAdd`。
* Block-wide atomics：当前程序中所有 CUDA 线程的原子操作，在与当前线程相同的线程块中执行。 这些以 _block 为后缀，例如 `atomicAdd_block`。

在以下示例中，CPU 和 GPU 都以原子方式更新地址 `addr` 处的整数值：
```C++
__global__ void mykernel(int *addr) {
  atomicAdd_system(addr, 10);       // only available on devices with compute capability 6.x
}

void foo() {
  int *addr;
  cudaMallocManaged(&addr, 4);
  *addr = 0;

   mykernel<<<...>>>(addr);
   __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}
```

请注意，任何原子操作都可以基于 `atomicCAS()`（Compare And Swap）来实现。 例如，用于双精度浮点数的 atomicAdd() 在计算能力低于 6.0 的设备上不可用，但可以按如下方式实现：
```C++
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
```

以下设备范围的原子 API 有系统范围和块范围的变体，但以下情况除外：

* 计算能力低于 6.0 的设备只支持设备范围的原子操作，
* 计算能力低于 7.2 的 Tegra 设备不支持系统范围的原子操作。

### B.14.1. Arithmetic Functions
#### B.14.1.1. atomicAdd()
```C++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
__nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
__nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val);
```
读取位于全局或共享内存中地址 `address` 的 16 位、32 位或 64 位字 `old`，计算 `(old + val)`，并将结果存储回同一地址的内存中。这三个操作在一个原子事务中执行。该函数返回`old`。

`atomicAdd()` 的 32 位浮点版本仅受计算能力 2.x 及更高版本的设备支持。

`atomicAdd()` 的 64 位浮点版本仅受计算能力 6.x 及更高版本的设备支持。

`atomicAdd()` 的 32 位 `__half2` 浮点版本仅受计算能力 6.x 及更高版本的设备支持。 `__half2` 或 `__nv_bfloat162` 加法操作的原子性分别保证两个 `__half` 或 `__nv_bfloat16` 元素中的每一个；不保证整个 `__half2` 或 `__nv_bfloat162` 作为单个 32 位访问是原子的。

`atomicAdd()` 的 16 位 `__half` 浮点版本仅受计算能力 7.x 及更高版本的设备支持。

`atomicAdd()` 的 16 位 `__nv_bfloat16` 浮点版本仅受计算能力 8.x 及更高版本的设备支持。

#### B.14.1.2. atomicSub()
```C++
int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(old - val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

#### B.14.1.3. atomicExch()
```C++
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);
```
读取位于全局或共享内存中地址address的 32 位或 64 位字 `old` 并将 `val` 存储回同一地址的内存中。 这两个操作在一个原子事务中执行。 该函数返回`old`。

#### B.14.1.4. atomicMin()
```C++
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最小值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMin()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

#### B.14.1.5. atomicMax()
```C++
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最大值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMax()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

#### B.14.1.6. atomicInc()
```C++
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```

读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `((old >= val) ? 0 : (old+1))`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

#### B.14.1.7. atomicDec()
```C++
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(((old == 0) || (old > val)) ? val : (old-1) )`，并将结果存储回同一个地址的内存。 这三个操作在一个原子事务中执行。 该函数返回`old`。

#### B.14.1.8. atomicCAS()
```C++
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address, 
                             unsigned short int compare, 
                             unsigned short int val);
```
读取位于全局或共享内存中地址`address`的 16 位、32 位或 64 位字 `old`，计算 `(old == compare ? val : old)` ，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`（Compare And Swap）。

### B.14.2. Bitwise Functions

#### B.14.2.1. atomicAnd()
```C++
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old & val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicAnd()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

#### B.14.2.2. atomicOr()
```C++
int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old | val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicOr()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

#### B.14.2.3. atomicXor()
```C++
int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old ^ val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicXor()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

## B.15. Address Space Predicate Functions
如果参数是空指针，则本节中描述的函数具有未指定的行为。

### B.15.1. __isGlobal()
```C++
__device__ unsigned int __isGlobal(const void *ptr);
```
如果 `ptr` 包含全局内存空间中对象的通用地址，则返回 1，否则返回 0。

### B.15.2. __isShared()
```C++
__device__ unsigned int __isShared(const void *ptr);
```
如果 `ptr` 包含共享内存空间中对象的通用地址，则返回 1，否则返回 0。

### B.15.3. __isConstant()
```C++
__device__ unsigned int __isConstant(const void *ptr);
```
如果 `ptr` 包含常量内存空间中对象的通用地址，则返回 1，否则返回 0。

### B.15.4. __isLocal()
```C++
__device__ unsigned int __isLocal(const void *ptr);
```
如果 `ptr` 包含本地内存空间中对象的通用地址，则返回 1，否则返回 0。

## B.16. Address Space Conversion Functions

### B.16.1. __cvta_generic_to_global()
```C++
__device__ size_t __cvta_generic_to_global(const void *ptr);
```
返回对 `ptr` 表示的通用地址执行 PTX `cvta.to.global` 指令的结果。

### B.16.2. __cvta_generic_to_shared()
```C++
__device__ size_t __cvta_generic_to_shared(const void *ptr);
```
返回对 `ptr` 表示的通用地址执行 PTX `cvta.to.shared` 指令的结果。

### B.16.3. __cvta_generic_to_constant()
```C++
__device__ size_t __cvta_generic_to_constant(const void *ptr);
```
返回对 `ptr` 表示的通用地址执行 PTX `cvta.to.const` 指令的结果。

### B.16.4. __cvta_generic_to_local()
```C++
__device__ void * __cvta_global_to_generic(size_t rawbits);
```
返回通过对 `rawbits` 提供的值执行 PTX `cvta.to.local` 指令获得的通用指针。

### B.16.5. __cvta_global_to_generic()
```C++
__device__ void * __cvta_global_to_generic(size_t rawbits);
```
返回通过对 `rawbits` 提供的值执行 PTX `cvta.global` 指令获得的通用指针。

### B.16.6. __cvta_shared_to_generic()
```C++
__device__ void * __cvta_shared_to_generic(size_t rawbits);
```
返回通过对 `rawbits` 提供的值执行 PTX `cvta.shared` 指令获得的通用指针。


### B.16.7. __cvta_constant_to_generic()
```C++
__device__ void * __cvta_constant_to_generic(size_t rawbits);
```
返回通过对 `rawbits` 提供的值执行 PTX `cvta.const` 指令获得的通用指针。

### B.16.8. __cvta_local_to_generic()
```C++
__device__ void * __cvta_local_to_generic(size_t rawbits);
```
返回通过对 `rawbits` 提供的值执行 PTX `cvta.local` 指令获得的通用指针。


## B.17. Alloca Function

### B.17.1. Synopsis
```C++
              __host__ __device__ void * alloca(size_t size);
```

### B.17.2. Description

`alloca()` 函数在调用者的堆栈帧(stack frame)中分配 `size` 个字节的内存。 返回值是一个指向分配内存的指针，当从设备代码调用函数时，内存的开头是 16 字节对齐的。 当 `alloca()` 的调用者返回时，分配的内存会自动释放。

注意：在 Windows 平台上，在使用 `alloca()` 之前必须包含 `<malloc.h>`。 使用 `alloca()` 可能会导致堆栈溢出，用户需要相应地调整堆栈大小。

它受计算能力 5.2 或更高版本的支持。

### B.17.3. Example
```C++
__device__ void foo(unsigned int num) {
	int4 *ptr = (int4 *)alloca(num * sizeof(int4));
	// use of ptr
	...
}
```

## B.18. Compiler Optimization Hint Functions
本节中描述的函数可用于向编译器优化器提供附加信息。

### B.18.1. __builtin_assume_aligned()
```C++
void * __builtin_assume_aligned (const void *exp, size_t align)
```
允许编译器假定参数指针至少对齐`align`字节，并返回参数指针。

Example:
```C++
void *res = __builtin_assume_aligned(ptr, 32); // compiler can assume 'res' is
                                               // at least 32-byte aligned
```     
三个参数版本:
```C++
      void * __builtin_assume_aligned (const void *exp, size_t align, 
                                       <integral type> offset)
```
允许编译器假设 `(char *)exp - offset` 至少对齐`align`字节，并返回参数指针。

Example:
```C++
void *res = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 
                                                  // '(char *)res - 8' is
                                                  // at least 32-byte aligned.
```

### B.18.2. __builtin_assume()
```C++
void __builtin_assume(bool exp)
```
允许编译器假定布尔参数为真。 如果参数在运行时不为真，则行为未定义。 该参数没有被评估，因此任何副作用都将被丢弃。

Example:
```C++
     __device__ int get(int *ptr, int idx) {
       __builtin_assume(idx <= 2);
       return ptr[idx];
    }
```

### B.18.3. __assume()
```C++
void __assume(bool exp)
```
允许编译器假定布尔参数为真。 如果参数在运行时不为真，则行为未定义。 该参数没有被评估，因此任何副作用都将被丢弃。

Example:
```C++
     __device__ int get(int *ptr, int idx) {
       __assume(idx <= 2);
       return ptr[idx];
    }
```

### B.18.4. __builtin_expect()
```C++
long __builtin_expect (long exp, long c)
```
向编译器指示期望 `exp == c`，并返回 `exp` 的值。 通常用于向编译器指示分支预测信息。
```C++
Example:

    // indicate to the compiler that likely "var == 0", 
    // so the body of the if-block is unlikely to be
    // executed at run time.
    if (__builtin_expect (var, 0))
      doit ();
```

### B.18.5. __builtin_unreachable()
```C++
void __builtin_unreachable(void)
```
向编译器指示控制流永远不会到达调用此函数的位置。 如果控制流在运行时确实到达了这一点，则程序具有未定义的行为。
```C++
Example:

    // indicates to the compiler that the default case label is never reached.
    switch (in) {
    case 1: return 4;
    case 2: return 10;
    default: __builtin_unreachable();
    }
```

### B.18.6. Restrictions
`__assume()` 仅在使用 cl.exe 主机编译器时受支持。 所有平台都支持其他功能，但受以下限制：

* 如果Host编译器支持该函数，则可以从translation unit中的任何位置调用该函数。
* 否则，必须从 `__device__/__global__` 函数的主体中调用该函数，或者仅在定义 `__CUDA_ARCH__` 宏时调用。
  
## B.19. Warp Vote Functions
```C++
        int __all_sync(unsigned mask, int predicate);
        int __any_sync(unsigned mask, int predicate);
        unsigned __ballot_sync(unsigned mask, int predicate);
        unsigned __activemask();
```
弃用通知：`__any、__all 和 __ballot` 在 CUDA 9.0 中已针对所有设备弃用。

删除通知：当面向具有 7.x 或更高计算能力的设备时，`__any、__all 和 __ballot` 不再可用，而应使用它们的同步变体。

warp 投票功能允许给定 warp 的线程执行缩减和广播操作。 这些函数将来自warp中每个线程的`int predicate`作为输入，并将这些值与零进行比较。 比较的结果通过以下方式之一在 warp 的活动线程中组合（减少），向每个参与线程广播单个返回值：
* `__all_sync(unsigned mask, predicate):`
评估`mask`中所有未退出线程的`predicate`，当且仅当`predicate`对所有线程的评估结果都为非零时，才返回非零值。
* `__any_sync(unsigned mask, predicate):`
评估`mask`中所有未退出线程的`predicate`，当且仅当`predicate`对其中任何一个的评估为非零时才返回非零。
* `__ballot_sync(unsigned mask, predicate):`
当且仅当 `predicate` 对 warp 的第 N 个线程计算为非零并且第 N 个线程处于活动状态时，为 `mask` 中所有未退出的线程计算`predicate`并返回一个其第 N 位被设置的整型。
* __activemask():
返回调用 warp 中所有当前活动线程的 32 位整数掩码。如果调用 `__activemask()` 时，warp 中的第 N 条通道处于活动状态，则设置第 N 位。非活动线程由返回掩码中的 0 位表示。退出程序的线程总是被标记为非活动的。请注意，在 `__activemask()` 调用中收敛的线程不能保证在后续指令中收敛，除非这些指令正在同步 warp 内置函数。

#### 注意:
对于` __all_sync、__any_sync 和 __ballot_sync`，必须传递一个掩码(`mask`)来指定参与调用的线程。 必须为每个参与线程设置一个表示线程通道 ID 的位，以确保它们在硬件执行内部函数之前正确收敛。 掩码中命名的所有活动线程必须使用相同的掩码执行相同的内部线程，否则结果未定义。

## B.20. Warp Match Functions
`__match_any_sync` 和 `__match_all_sync` 在 warp 中的线程之间执行变量的广播和比较操作。

由计算能力 7.x 或更高版本的设备支持。

### B.20.1. Synopsis
```C++
unsigned int __match_any_sync(unsigned mask, T value);
unsigned int __match_all_sync(unsigned mask, T value, int *pred);
```
`T` 可以是 `int、unsigned int、long、unsigned long、long long、unsigned long long、float 或 double`。

### B.20.2. Description
`__match_sync()`的intrinsics允许在对`mask`中命名的线程进行同步之后，在不同的线程之间广播和比较一个值。

`__match_any_sync`

返回`mask`中具有相同`value`的线程掩码

`__match_all_sync`

如果掩码中的所有线程的`value`值都相同，则返回`mask`； 否则返回 0。 如果 `mask` 中的所有线程具有相同的 `value` 值，则 `pred` 设置为 `true`； 否则predicate设置为假。

新的 `*_sync` 匹配内在函数采用一个掩码，指示参与调用的线程。 必须为每个参与线程设置一个表示线程通道 ID 的位，以确保它们在硬件执行内部函数之前正确收敛。 掩码中命名的所有非退出线程必须使用相同的掩码执行相同的内在函数，否则结果未定义。

## B.21. Warp Reduce Functions
`__reduce_sync(unsigned mask, T value)` 内在函数在同步 `mask` 中命名的线程后对 `value` 中提供的数据执行归约操作。 `T` 对于 `{add, min, max}` 可以是无符号的或有符号的，并且仅对于 `{and, or, xor}` 操作是无符号的。

由计算能力 8.x 或更高版本的设备支持。

### B.21.1. Synopsis
```C++
// add/min/max
unsigned __reduce_add_sync(unsigned mask, unsigned value);
unsigned __reduce_min_sync(unsigned mask, unsigned value);
unsigned __reduce_max_sync(unsigned mask, unsigned value);
int __reduce_add_sync(unsigned mask, int value);
int __reduce_min_sync(unsigned mask, int value);
int __reduce_max_sync(unsigned mask, int value);

// and/or/xor
unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync(unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);
```

### B.21.2. Description
`__reduce_add_sync、__reduce_min_sync、__reduce_max_sync`

返回对 `mask` 中命名的每个线程在 `value` 中提供的值应用算术加法、最小或最大规约操作的结果。

`__reduce_and_sync、__reduce_or_sync、__reduce_xor_sync`

返回对 `mask` 中命名的每个线程在 `value` 中提供的值应用逻辑 `AND、OR 或 XOR` 规约操作的结果。

## B.22. Warp Shuffle Functions
`__shfl_sync、__shfl_up_sync、__shfl_down_sync 和 __shfl_xor_sync` 在 [warp](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) 内的线程之间交换变量。

由计算能力 3.x 或更高版本的设备支持。

弃用通知：`__shfl、__shfl_up、__shfl_down 和 __shfl_xor` 在 CUDA 9.0 中已针对所有设备弃用。

删除通知：当面向具有 7.x 或更高计算能力的设备时，`__shfl、__shfl_up、__shfl_down 和 __shfl_xor` 不再可用，而应使用它们的同步变体。
#### 作者添加:这里可能大家对接下来会提到的threadIndex, warpIdx, laneIndex会比较混淆.那么我用下图来说明.
![ID.png](ID.png)

### B.22.1. Synopsis
```C++
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```
`T` 可以是 `int、unsigned int、long、unsigned long、long long、unsigned long long、float 或 double`。 包含 `cuda_fp16.h` 头文件后，`T` 也可以是 `__half 或 __half2`。 同样，包含 cuda_bf16.h 头文件后，T 也可以是 `__nv_bfloat16 或 __nv_bfloat162`。

### B.22.2. Description
`__shfl_sync()` 内在函数允许在 warp 内的线程之间交换变量，而无需使用共享内存。 交换同时发生在 warp 中的所有活动线程（并以`mask`命名），根据类型移动每个线程 4 或 8 个字节的数据。

warp 中的线程称为通道(lanes)，并且可能具有介于 0 和 warpSize-1（包括）之间的索引。 支持四种源通道(source-lane)寻址模式：

`__shfl_sync()`

从索引通道直接复制

`__shfl_up_sync()`

从相对于调用者 ID 较低的通道复制

`__shfl_down_sync()`

从相对于调用者具有更高 ID 的通道复制

`__shfl_xor_sync()`

基于自身通道 ID 的按位`异或`从通道复制

线程只能从积极参与 `__shfl_sync()` 命令的另一个线程读取数据。 如果目标线程处于非活动状态，则检索到的值未定义。

所有 `__shfl_sync()` 内在函数都采用一个可选的宽度参数，该参数会改变内在函数的行为。 `width` 的值必须是 2 的幂； 如果 `width` 不是 2 的幂，或者是大于 `warpSize` 的数字，则结果未定义。

`__shfl_sync()` 返回由 `srcLane` 给定 ID 的线程持有的 `var` 的值。 如果 `width` 小于 `warpSize`，则 warp 的每个子部分都表现为一个单独的实体，其起始逻辑通道 ID 为 0。如果 `srcLane` 超出范围 [0:width-1]，则返回的值对应于通过 `srcLane` srcLane modulo width所持有的 `var` 的值 （即在同一部分内）。
#### 作者添加:这里原本中说的有点绕,我还是用图来说明比较好.注意下面四个图均由作者制作,如果有问题,仅仅是作者水平问题-_-!.
![shfl.png](shfl.png)

`__shfl_up_sync()` 通过从调用者的通道 ID 中减去 delta 来计算源通道 ID。 返回由生成的通道 ID 保存的 `var` 的值：实际上， `var` 通过 `delta` 通道向上移动。 如果宽度小于 `warpSize`，则warp的每个子部分都表现为一个单独的实体，起始逻辑通道 ID 为 0。源通道索引不会环绕宽度值，因此实际上较低的 `delta` 通道将保持不变。
![shfl_up.png](shfl_up.png)

`__shfl_down_sync()` 通过将 delta 加调用者的通道 ID 来计算源通道 ID。 返回由生成的通道 ID 保存的 `var` 的值：这具有将 `var` 向下移动 `delta` 通道的效果。 如果 `width` 小于 warpSize，则 warp 的每个子部分都表现为一个单独的实体，起始逻辑通道 ID 为 0。至于 `__shfl_up_sync()`，源通道的 ID 号不会环绕宽度值，因此 upper delta lanes将保持不变。
![shfl_down.png](shfl_down.png)

`__shfl_xor_sync()` 通过对调用者的通道 ID 与 `laneMask` 执行按位异或来计算源通道 ID：返回结果通道 ID 所持有的 `var` 的值。 如果宽度小于warpSize，那么每组宽度连续的线程都能够访问早期线程组中的元素，但是如果它们尝试访问后面线程组中的元素，则将返回他们自己的`var`值。 这种模式实现了一种蝶式寻址模式，例如用于树规约和广播。
![shufl_xor.png](shufl_xor.png)

新的 `*_sync shfl` 内部函数采用一个掩码，指示参与调用的线程。 必须为每个参与线程设置一个表示线程通道 ID 的位，以确保它们在硬件执行内部函数之前正确收敛。 掩码中命名的所有非退出线程必须使用相同的掩码执行相同的内在函数，否则结果未定义。

### B.22.3. Notes
线程只能从积极参与 __shfl_sync() 命令的另一个线程读取数据。 如果目标线程处于非活动状态，则检索到的值未定义。

宽度必须是 2 的幂（即 2、4、8、16 或 32）。 未指定其他值的结果。

### B.22.4. Examples
#### B.22.4.1. Broadcast of a single value across a warp
```C++
#include <stdio.h>

__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    value = __shfl_sync(0xffffffff, value, 0);   // Synchronize all threads in warp, and get "value" from lane 0
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}

int main() {
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
```

B.22.4.2. Inclusive plus-scan across sub-partitions of 8 threads
```C++
#include <stdio.h>

__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan4<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
```

#### B.22.4.3. Reduction across a warp
```C++
#include <stdio.h>

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
```

## B.23. Nanosleep Function
### B.23.1. Synopsis
```C++
T __nanosleep(unsigned ns);
```
### B.23.2. Description
`__nanosleep(ns)` 将线程挂起大约接近延迟 ns 的睡眠持续时间，以纳秒为单位指定。

它受计算能力 7.0 或更高版本的支持。

## B.23.3. Example
以下代码实现了一个具有指数回退的互斥锁。
```C++
__device__ void mutex_lock(unsigned int *mutex) {
    unsigned int ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_unlock(unsigned int *mutex) {
    atomicExch(mutex, 0);
}
```

## B.24. Warp matrix functions
C++ warp矩阵运算利用Tensor Cores来加速 `D=A*B+C` 形式的矩阵问题。 计算能力 7.0 或更高版本的设备的混合精度浮点数据支持这些操作。 这需要一个warp中所有线程的合作。 此外，仅当条件在整个 [warp](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) 中的计算结果相同时，才允许在条件代码中执行这些操作，否则代码执行可能会挂起。

### B.24.1. Description
以下所有函数和类型都在命名空间 `nvcuda::wmma` 中定义。 Sub-byte操作被视为预览版，即它们的数据结构和 API 可能会发生变化，并且可能与未来版本不兼容。 这个额外的功能在 nvcuda::wmma::experimental 命名空间中定义。
```C++
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

`fragment`:

包含矩阵的一部分的重载类，分布在warp中的所有线程中。 矩阵元素到`fragment`内部存储的映射是未指定的，并且在未来的架构中可能会发生变化。

只允许模板参数的某些组合。 第一个模板参数指定片段将如何参与矩阵运算。 可接受的使用值是：
* `matrix_a` 当`fragment` 用作第一个被乘数时，A
* `matrix_b` 当`fragment`用作第二个被乘数时，B
* 当`fragment`用作源或目标累加器（分别为 C 或 D）时的累加器。

`m、n 和 k` 大小描述了参与乘法累加操作的warp-wide矩阵块的形状。 每个tile的尺寸取决于它的作用。 对于 `matrix_a`，图块的尺寸为 `m x k`； 对于 `matrix_b`，维度是 `k x n`，累加器块是 `m x n`。

对于被乘数，数据类型 `T` 可以是 `double、float、__half、__nv_bfloat16、char 或 unsigned char`，对于累加器，可以是 `double、float、int 或 __half`。 如[元素类型和矩阵大小](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes)中所述，支持累加器和被乘数类型的有限组合。 必须为 `matrix_a` 和 `matrix_b` 片段指定 `Layout` 参数。 `row_major` 或 `col_major` 分别表示矩阵***行或列***中的元素在内存中是连续的。 累加器矩阵的 `Layout` 参数应保留默认值 `void`。 仅当按如下所述加载或存储累加器时才指定行或列布局。


`load_matrix_sync`:

等到所有warp通道(lanes)都到达 `load_matrix_sync`，然后从内存中加载矩阵片段 `a`。 `mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。 `ldm` 描述连续行（对于行主序）或列（对于列主序）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于浮点元素类型必须是 4 的倍数。 （即，两种情况下都是 16 字节的倍数）。 如果`fragment`是累加器，则布局参数必须指定为 `mem_row_major` 或 `mem_col_major`。 对于 `matrix_a` 和 `matrix_b` 片段，`Layout`是从`fragment`的`Layout`参数中推断出来的。 a 的 `mptr、ldm、layout` 和所有模板参数的值对于 warp 中的所有线程必须相同。 这个函数必须被warp中的所有线程调用，否则结果是未定义的。

`store_matrix_sync`:

等到所有warp通道都到达 `store_matrix_sync`，然后将矩阵片段 a 存储到内存中。 `mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。 `ldm` 描述连续行（对于行主序）或列（对于列主序）之间的元素跨度，对于` __half` 元素类型必须是 8 的倍数，对于浮点元素类型必须是 4 的倍数。 （即，两种情况下都是 16 字节的倍数）。 输出矩阵的布局必须指定为 `mem_row_major` 或 `mem_col_major`。 a 的 `mptr、ldm、layout` 和所有模板参数的值对于 warp 中的所有线程必须相同。

`fill_fragment`:

用常量 v 填充矩阵片段。由于未指定矩阵元素到每个片段的映射，因此该函数通常由 warp 中的所有线程调用，并具有共同的 v 值。

`mma_sync`:

等到所有`warp lanes`都到达`mma_sync`，然后执行warp同步的矩阵乘法累加操作`D=A*B+C`。 还支持原位(in-place)操作，`C=A*B+C`。 对于 warp 中的所有线程，每个矩阵片段的 `satf` 和模板参数的值必须相同。 此外，模板参数 `m、n 和 k` 必须在片段 `A、B、C 和 D` 之间匹配。该函数必须由 warp 中的所有线程调用，否则结果未定义。

如果 `satf`（饱和到有限值--saturate to finite value）模式为真，则以下附加数值属性适用于目标累加器：
* 如果元素结果为+Infinity，则相应的累加器将包含+MAX_NORM
* 如果元素结果为 -Infinity，则相应的累加器将包含 -MAX_NORM
* 如果元素结果为 NaN，则对应的累加器将包含 +0

由于未指定矩阵元素到每个线程片段的映射，因此必须在调用 `store_matrix_sync` 后从内存（共享或全局）访问单个矩阵元素。 在 warp 中的所有线程将对所有片段元素统一应用元素操作的特殊情况下，可以使用以下`fragment`类成员实现直接元素访问。

```C++
enum fragment<Use, m, n, k, T, Layout>::num_elements;
T fragment<Use, m, n, k, T, Layout>::x[num_elements];
```

例如，以下代码将累加器矩阵缩小一半。
```C++
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
float alpha = 0.5f; // Same value for all threads in warp
/*...*/
for(int t=0; t<frag.num_elements; t++)
frag.x[t] *= alpha; 
```

### B.24.2. Alternate Floating Point
Tensor Core 支持在具有 8.0 及更高计算能力的设备上进行替代类型的浮点运算。

`__nv_bfloat16`:

此数据格式是另一种 `fp16 `格式，其范围与 `f32` 相同，但精度降低（7 位）。 您可以直接将此数据格式与 `cuda_bf16.h` 中提供的 `__nv_bfloat16` 类型一起使用。 具有 `__nv_bfloat16` 数据类型的矩阵片段需要与浮点类型的累加器组合。 支持的形状和操作与 `__half` 相同。

`tf32`:

这种数据格式是 `Tensor Cores` 支持的特殊浮点格式，范围与 f32 相同，但精度降低（>=10 位）。这种格式的内部布局是实现定义的。为了在 `WMMA` 操作中使用这种浮点格式，输入矩阵必须手动转换为 `tf32` 精度。

为了便于转换，提供了一个新的内联函数 `__float_to_tf32`。虽然内联函数的输入和输出参数是浮点类型，但输出将是 `tf32`。这个新精度仅适用于张量核心，如果与其他浮点类型操作混合使用，结果的精度和范围将是未定义的。

一旦输入矩阵（`matrix_a` 或 `matrix_b`）被转换为 `tf32` 精度，具有`precision::tf32` 精度的片段和`load_matrix_sync` 的`float` 数据类型的组合将利用此新功能。两个累加器片段都必须具有浮点数据类型。唯一支持的矩阵大小是 `16x16x8 (m-n-k)`。

片段的元素表示为浮点数，因此从 `element_type<T>` 到 `storage_element_type<T>` 的映射是：
```C++
precision::tf32 -> float
```

### B.24.3. Double Precision
`Tensor Core` 支持计算能力 8.0 及更高版本的设备上的双精度浮点运算。 要使用这个新功能，必须使用具有 `double` 类型的片段。 `mma_sync` 操作将使用 `.rn`（四舍五入到最接近的偶数）舍入修饰符执行。

### B.24.4. Sub-byte Operations
 














