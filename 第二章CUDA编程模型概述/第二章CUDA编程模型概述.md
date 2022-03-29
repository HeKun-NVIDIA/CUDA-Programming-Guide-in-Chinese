# 2.编程模型
#### 本章通过概述CUDA编程模型是如何在c++中公开的，来介绍CUDA的主要概念。
#### [编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中给出了对 CUDA C++ 的广泛描述。

#### 本章和下一章中使用的向量加法示例的完整代码可以在 vectorAdd [CUDA示例](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition)中找到。

## 2.1 内核
#### CUDA C++ 通过允许程序员定义称为kernel的 C++ 函数来扩展 C++，当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。
#### 使用 `__global__` 声明说明符定义内核，并使用新的 `<<<...>>>` 执行配置语法指定内核调用的 CUDA 线程数（请参阅 [C++ 语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)）。 每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问。

#### 作为说明，以下示例代码使用内置变量 `threadIdx` 将两个大小为 N 的向量 A 和 B 相加，并将结果存储到向量 C 中：
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