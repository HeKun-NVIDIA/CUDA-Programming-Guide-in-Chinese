# 1.CUDA简介
## 1.1 我们为什么要使用GPU
GPU（Graphics Processing Unit）在相同的价格和功率范围内，比CPU提供更高的指令吞吐量和内存带宽。许多应用程序利用这些更高的能力，使得自己在 GPU 上比在 CPU 上运行得更快 (参见[GPU应用程序](https://www.nvidia.com/object/gpu-applications.html)) 。其他计算设备，如FPGA，也非常节能，但提供的编程灵活性要比GPU少得多。

GPU和CPU之间的主要区别在于设计思想的不同。CPU的设计初衷是为了实现在执行一系列操作时达到尽可能高的性能，其中每个操作称之为一个thread，同时可能只能实现其中数十个线程的并行化，GPU的设计初衷是为了实现在在并行执行数千个线程时达到尽可能高的性能（通过分摊较慢的单线程程序以实现更高的吞吐量）。

为了能够实现更高强度的并行计算，GPU将更多的晶体管用于数据计算而不是数据缓存或控制流。
下图显示了 CPU 与 GPU 的芯片资源分布示例。

![The GPU Devotes More Transistors to Data Processing](gpu-devotes-more-transistors-to-data-processing.png)

一般来说，应用程序有并行和串行部分，所以系统可以利用GPU和CPU的混搭来获得更高的整体性能。对于并行度高的程序也可以利用GPU的大规模并行特性来实现比CPU更高的性能。

## 1.2 CUDA®：通用并行计算平台和程序模型
2006 年 11 月，NVIDIA® 推出了 CUDA®，这是一种通用并行计算平台和程序模型，它利用 NVIDIA GPU 中的并行计算引擎以比 CPU 更有效的方式解决许多复杂的计算问题。

CUDA的软件环境允许开发者使用C++等高级编程语言开发程序。 如下图所示，支持其他语言、应用程序编程接口或基于指令的方法，例如 FORTRAN、DirectCompute、OpenACC。

![gpu-computing-applications.png](gpu-computing-applications.png)


## 1.3 A Scalable Programming Model
多核 CPU 和超多核 (manycore) GPU 的出现，意味着主流处理器进入并行时代。当下开发应用程序的挑战在于能够利用不断增加的处理器核数实现对于程序并行性透明地扩展，例如 3D 图像应用可以透明地拓展其并行性来适应内核数量不同的 GPUs 硬件。

CUDA并行程序模型主要为克服这一挑战而设计，其对于程序员具有较小的学习难度，因为其使用了标准编程语言。

其核心是三个关键抽象——线程组的层次结构、共享内存和屏障同步——它们只是作为最小的语言扩展集向程序员公开。

这些抽象提供了细粒度的数据并行性和线程并行性，并将嵌套在粗粒度的数据并行和任务并行中。它们指导程序员将主问题拆解为可以线程块独立并行解决的粗粒度子问题，同时每个子问题可以被进一步细分为更小的组成部分，其可以被每个线程块中的线程通过并行合作的方式解决。

这种拆解通过运行线程在解决子问题时使用合作机制，保留了语言的表达能力，同时也为系统提供了自动的可拓展性。实际上，每个线程块可以被异步或同步地调度给 GPU 上任意一个多处理器 (Multiprocessors)。故 CUDA 程序可以被执行在具有任意 kernel 数据的 GPU 中，如下图所示，同时在运行时阶段，系统只需要给出物理多处理器地个数。

这种可扩展的程序模型允许 GPU 架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：高性能发烧友 GeForce GPU ，专业的 Quadro 和 Tesla 计算产品 (有关所有支持 CUDA 的 GPU 的列表，请参阅[支持 CUDA 的 GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)）。

![automatic-scalability.png](automatic-scalability.png)

注意：GPU 是围绕一系列流式多处理器 (SM: Streaming Multiprocessors) 构建的（有关详细信息，请参[阅硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)）。 多线程程序被划分为彼此独立执行的线程块，因此具有更多多处理器的 GPU 将比具有更少多处理器的 GPU 在更短的时间内完成程序执行。
