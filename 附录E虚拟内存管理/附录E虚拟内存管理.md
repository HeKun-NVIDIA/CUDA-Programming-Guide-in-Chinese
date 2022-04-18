# 附录E虚拟内存管理

## E.1. Introduction

[虚拟内存管理 API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) 为应用程序提供了一种直接管理统一虚拟地址空间的方法，该空间由 CUDA 提供，用于将物理内存映射到 GPU 可访问的虚拟地址。在 CUDA 10.2 中引入的这些 API 还提供了一种与其他进程和图形 API（如 OpenGL 和 Vulkan）进行互操作的新方法，并提供了用户可以调整以适应其应用程序的更新内存属性。

从历史上看，CUDA 编程模型中的内存分配调用（例如 `cudaMalloc`）返回了一个指向 GPU 内存的内存地址。这样获得的地址可以与任何 CUDA API 一起使用，也可以在设备内核中使用。但是，分配的内存无法根据用户的内存需求调整大小。为了增加分配的大小，用户必须显式分配更大的缓冲区，从初始分配中复制数据，释放它，然后继续跟踪新分配的地址。这通常会导致应用程序的性能降低和峰值内存利用率更高。本质上，用户有一个类似 `malloc` 的接口来分配 GPU 内存，但没有相应的 `realloc` 来补充它。虚拟内存管理 API 将地址和内存的概念解耦，并允许应用程序分别处理它们。 API 允许应用程序在他们认为合适的时候从虚拟地址范围映射和取消映射内存。

在通过 `cudaEnablePeerAccess` 启用对等设备访问内存分配的情况下，所有过去和未来的用户分配都映射到目标对等设备。这导致用户无意中支付了将所有 `cudaMalloc` 分配映射到对等设备的运行时成本。然而，在大多数情况下，应用程序通过仅与另一个设备共享少量分配进行通信，并且并非所有分配都需要映射到所有设备。使用虚拟内存管理，应用程序可以专门选择某些分配可从目标设备访问。

CUDA 虚拟内存管理 API 向用户提供细粒度控制，以管理应用程序中的 GPU 内存。它提供的 API 允许用户：

* 将分配在不同设备上的内存放入一个连续的 VA 范围内。
* 使用平台特定机制执行内存共享的进程间通信。
* 在支持它们的设备上选择更新的内存类型。

为了分配内存，虚拟内存管理编程模型公开了以下功能：

* 分配物理内存。
* 保留 VA 范围。
* 将分配的内存映射到 VA 范围。
* 控制映射范围的访问权限。

请注意，本节中描述的 API 套件需要支持 UVA 的系统。

## E.2. Query for support

在尝试使用虚拟内存管理 API 之前，应用程序必须确保他们希望使用的设备支持 CUDA 虚拟内存管理。 以下代码示例显示了查询虚拟内存管理支持：

```C++
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management 
}
   
```

## E.3. Allocating Physical Memory
