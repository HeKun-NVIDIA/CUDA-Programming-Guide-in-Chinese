# 附录I C++ 语言支持
如使用 [NVCC 编译中所述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)，使用 nvcc 编译的 CUDA 源文件可以包含主机代码和设备代码的混合。 CUDA 前端编译器旨在模拟主机编译器对 C++ 输入代码的行为。 输入源代码根据 C++ ISO/IEC 14882:2003、C++ ISO/IEC 14882:2011、C++ ISO/IEC 14882:2014 或 C++ ISO/IEC 14882:2017 规范进行处理，CUDA 前端编译器旨在模拟 任何主机编译器与 ISO 规范的差异。 此外，支持的语言使用本[文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_15) 中描述的特定于 CUDA 的结构进行了扩展，并受到下面描述的限制。

[C++11 语言特性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp11-language-features)、[C++14](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp14-language-features) 语言特性和 [C++17](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp17-language-features) 语言特性分别为 C++11、C++14 和 C++17 特性提供支持矩阵。 [限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrictions)列出了语言限制。 [多态函数包装器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#polymorphic-function-wrappers)和[扩展 Lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda) 描述了其他特性。 [代码示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#code-samples)提供代码示例。

## I.1. C++11 Language Features
下表列出了已被 C++11 标准接受的新语言功能。 “Proposal”列提供了描述该功能的 ISO C++ 委员会提案的链接，而“Available in nvcc (device code)”列表示包含此功能实现的第一个 nvcc 版本（如果已实现） ) 用于设备代码。

<div class="tablenoborder"><a name="cpp11-language-features__cpp11-language-features-support-matrix" shape="rect">
                              <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="cpp11-language-features__cpp11-language-features-support-matrix" class="table" frame="border" border="1" rules="all">
                              <caption><span class="tablecap">Table 12. C++11 Language Features</span></caption>
                              <thead class="thead" align="left">
                                 <tr class="row" valign="middle">
                                    <th class="entry" align="left" valign="middle" width="71.42857142857143%" id="d117e28759" rowspan="1" colspan="1">Language Feature</th>
                                    <th class="entry" align="center" valign="middle" width="14.285714285714285%" id="d117e28762" rowspan="1" colspan="1">C++11 Proposal</th>
                                    <th class="entry" align="center" valign="middle" width="14.285714285714285%" id="d117e28765" rowspan="1" colspan="1">Available in nvcc (device code)</th>
                                 </tr>
                              </thead>
                              <tbody class="tbody">
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Rvalue references</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html" target="_blank" shape="rect">N2118</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">
                                       &nbsp;&nbsp;&nbsp;&nbsp;Rvalue references for <samp class="ph codeph">*this</samp></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm" target="_blank" shape="rect">N2439</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Initialization of class objects by rvalues</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html" target="_blank" shape="rect">N1610</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Non-static data member initializers</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm" target="_blank" shape="rect">N2756</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Variadic templates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf" target="_blank" shape="rect">N2242</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">&nbsp;&nbsp;&nbsp;&nbsp;Extending variadic template template parameters</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf" target="_blank" shape="rect">N2555</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Initializer lists</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm" target="_blank" shape="rect">N2672</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Static assertions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html" target="_blank" shape="rect">N1720</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1"><samp class="ph codeph">auto</samp>-typed variables 
                                    </td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf" target="_blank" shape="rect">N1984</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">
                                       &nbsp;&nbsp;&nbsp;&nbsp;Multi-declarator <samp class="ph codeph">auto</samp></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf" target="_blank" shape="rect">N1737</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">&nbsp;&nbsp;&nbsp;&nbsp;Removal of auto as a storage-class specifier</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm" target="_blank" shape="rect">N2546</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">&nbsp;&nbsp;&nbsp;&nbsp;New function declarator syntax</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm" target="_blank" shape="rect">N2541</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Lambda expressions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf" target="_blank" shape="rect">N2927</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Declared type of an expression</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf" target="_blank" shape="rect">N2343</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">&nbsp;&nbsp;&nbsp;&nbsp;Incomplete return types</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf" target="_blank" shape="rect">N3276</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Right angle brackets</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html" target="_blank" shape="rect">N1757</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Default template arguments for function templates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226" target="_blank" shape="rect">DR226</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Solving the SFINAE problem for expressions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html" target="_blank" shape="rect">DR339</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Alias templates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf" target="_blank" shape="rect">N2258</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Extern templates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm" target="_blank" shape="rect">N1987</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Null pointer constant</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf" target="_blank" shape="rect">N2431</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Strongly-typed enums</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf" target="_blank" shape="rect">N2347</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Forward declarations for enums</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf" target="_blank" shape="rect">N2764</a><br clear="none"></br><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206" target="_blank" shape="rect">DR1206</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Standardized attribute syntax</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf" target="_blank" shape="rect">N2761</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Generalized constant expressions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf" target="_blank" shape="rect">N2235</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Alignment support</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf" target="_blank" shape="rect">N2341</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Conditionally-support behavior</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf" target="_blank" shape="rect">N1627</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Changing undefined behavior into diagnosable errors</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf" target="_blank" shape="rect">N1727</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Delegating constructors</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf" target="_blank" shape="rect">N1986</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Inheriting constructors</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm" target="_blank" shape="rect">N2540</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Explicit conversion operators</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf" target="_blank" shape="rect">N2437</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">New character types</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html" target="_blank" shape="rect">N2249</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Unicode string literals</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm" target="_blank" shape="rect">N2442</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Raw string literals</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm" target="_blank" shape="rect">N2442</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Universal character names in literals</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html" target="_blank" shape="rect">N2170</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">User-defined literals</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf" target="_blank" shape="rect">N2765</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Standard Layout Types</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm" target="_blank" shape="rect">N2342</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Defaulted functions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm" target="_blank" shape="rect">N2346</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Deleted functions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm" target="_blank" shape="rect">N2346</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Extended friend declarations</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf" target="_blank" shape="rect">N1791</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">
                                       Extending <samp class="ph codeph">sizeof</samp></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html" target="_blank" shape="rect">N2253</a><br clear="none"></br><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850" target="_blank" shape="rect">DR850</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Inline namespaces</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm" target="_blank" shape="rect">N2535</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Unrestricted unions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf" target="_blank" shape="rect">N2544</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Local and unnamed types as template arguments</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm" target="_blank" shape="rect">N2657</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Range-based for</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html" target="_blank" shape="rect">N2930</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Explicit virtual overrides</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm" target="_blank" shape="rect">N2928</a><br clear="none"></br><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm" target="_blank" shape="rect">N3206</a><br clear="none"></br><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm" target="_blank" shape="rect">N3272</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Minimal support for garbage collection and reachability-based leak detection</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm" target="_blank" shape="rect">N2670</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">
                                       N/A (see <a class="xref" href="index.html#restrictions" shape="rect">Restrictions</a>)
                                    </td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Allowing move constructors to throw [noexcept]</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html" target="_blank" shape="rect">N3050</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Defining move special member functions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html" target="_blank" shape="rect">N3053</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" colspan="3" align="center" valign="middle" headers="d117e28759 d117e28762 d117e28765" rowspan="1"><strong class="ph b">Concurrency</strong></td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Sequence points</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html" target="_blank" shape="rect">N2239</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Atomic operations</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html" target="_blank" shape="rect">N2427</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Strong Compare and Exchange</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html" target="_blank" shape="rect">N2748</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Bidirectional Fences</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm" target="_blank" shape="rect">N2752</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Memory model</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm" target="_blank" shape="rect">N2429</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Data-dependency ordering: atomics and memory model</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm" target="_blank" shape="rect">N2664</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Propagating exceptions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html" target="_blank" shape="rect">N2179</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Allow atomics use in signal handlers</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm" target="_blank" shape="rect">N2547</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Thread-local storage</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm" target="_blank" shape="rect">N2659</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Dynamic initialization and destruction with concurrency</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm" target="_blank" shape="rect">N2660</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" colspan="3" align="center" valign="middle" headers="d117e28759 d117e28762 d117e28765" rowspan="1"><strong class="ph b">C99 Features in C++11</strong></td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1"><samp class="ph codeph">__func__</samp> predefined identifier
                                    </td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm" target="_blank" shape="rect">N2340</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">C99 preprocessor</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm" target="_blank" shape="rect">N1653</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1"><samp class="ph codeph">long long</samp></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf" target="_blank" shape="rect">N1811</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">7.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e28759" rowspan="1" colspan="1">Extended integral types</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28762" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf" target="_blank" shape="rect">N1988</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e28765" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                              </tbody>
                           </table>
                        </div>


## I.2. C++14 Language Features

下表列出了已被 C++14 标准接受的新语言功能。
<div class="tablenoborder"><a name="cpp14-language-features__cpp14-language-features-support-matrix" shape="rect">
                              <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="cpp14-language-features__cpp14-language-features-support-matrix" class="table" frame="border" border="1" rules="all">
                              <caption><span class="tablecap">Table 13. C++14 Language Features</span></caption>
                              <thead class="thead" align="left">
                                 <tr class="row" valign="middle">
                                    <th class="entry" align="left" valign="middle" width="71.42857142857143%" id="d117e29817" rowspan="1" colspan="1">Language Feature</th>
                                    <th class="entry" align="center" valign="middle" width="14.285714285714285%" id="d117e29820" rowspan="1" colspan="1">C++14 Proposal</th>
                                    <th class="entry" align="center" valign="middle" width="14.285714285714285%" id="d117e29823" rowspan="1" colspan="1">Available in nvcc (device code)</th>
                                 </tr>
                              </thead>
                              <tbody class="tbody">
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Tweak to certain C++ contextual conversions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf" target="_blank" shape="rect">N3323</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Binary literals</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf" target="_blank" shape="rect">N3472</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Functions with deduced return type</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/N3638.html" target="_blank" shape="rect">N3638</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Generalized lambda capture (init-capture)</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/N3648.html" target="_blank" shape="rect">N3648</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Generic (polymorphic) lambda expressions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/N3649.html" target="_blank" shape="rect">N3649</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Variable templates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/N3651.pdf" target="_blank" shape="rect">N3651</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Relaxing requirements on constexpr functions</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/N3652.html" target="_blank" shape="rect">N3652</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Member initializers and aggregates</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html" target="_blank" shape="rect">N3653</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Clarifying memory allocation</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html" target="_blank" shape="rect">N3664</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Sized deallocation</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="https://isocpp.org/files/papers/n3778.html" target="_blank" shape="rect">N3778</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">&nbsp;</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1"><samp class="ph codeph">[[deprecated]]</samp> attribute
                                    </td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html" target="_blank" shape="rect">N3760</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                                 <tr class="row" valign="middle">
                                    <td class="entry" align="left" valign="middle" width="71.42857142857143%" headers="d117e29817" rowspan="1" colspan="1">Single-quotation-mark as a digit separator</td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29820" rowspan="1" colspan="1"><a class="xref" href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf" target="_blank" shape="rect">N3781</a></td>
                                    <td class="entry" align="center" valign="middle" width="14.285714285714285%" headers="d117e29823" rowspan="1" colspan="1">9.0</td>
                                 </tr>
                              </tbody>
                           </table>
                        </div>

## I.3. C++17 Language Features

nvcc 版本 11.0 及更高版本支持所有 C++17 语言功能，但受[此处](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp17)描述的限制的约束。


## I.4. Restrictions

### I.4.1. Host Compiler Extensions
设备代码不支持主机编译器特定的语言扩展。

`_Complex`类型仅在主机代码中受支持。

当与支持它的主机编译器一起编译时，设备代码中支持 `__int128` 类型。

`__float128` 类型仅在 64 位 x86 Linux 平台上的主机代码中受支持。 `__float128` 类型的常量表达式可以由编译器以较低精度的浮点表示形式处理。

### I.4.2. Preprocessor Symbols

#### I.4.2.1. __CUDA_ARCH__

1. 以下实体的类型签名不应取决于是否定义了 `__CUDA_ARCH__`，或者取决于 `__CUDA_ARCH__` 的特定值：
    * `__global__` 函数和函数模板
    * `__device__` 和 `__constant__` 变量
    * 纹理和表面

    例子：
```C++
#if !defined(__CUDA_ARCH__)
typedef int mytype;
#else
typedef double mytype;
#endif

__device__ mytype xxx;         // error: xxx's type depends on __CUDA_ARCH__
__global__ void foo(mytype in, // error: foo's type depends on __CUDA_ARCH__
                    mytype *ptr)
{
  *ptr = in;
}
```

2. 如果 `__global__` 函数模板被实例化并从主机启动，则无论是否定义了 `__CUDA_ARCH__` 以及无论 `__CUDA_ARCH__` 的值如何，都必须使用相同的模板参数实例化该函数模板。

    例子：
```C++
__device__ int result;
template <typename T>
__global__ void kern(T in)
{
  result = in;
}

__host__ __device__ void foo(void)
{
#if !defined(__CUDA_ARCH__)
  kern<<<1,1>>>(1);      // error: "kern<int>" instantiation only
                         // when __CUDA_ARCH__ is undefined!
#endif
}

int main(void)
{
  foo();
  cudaDeviceSynchronize();
  return 0;
}

```

3. 在单独编译模式下，是否存在具有外部链接的函数或变量的定义不应取决于是否定义了 `__CUDA_ARCH__` 或 `__CUDA_ARCH__16` 的特定值。
    
    例子：
```C++
#if !defined(__CUDA_ARCH__)
void foo(void) { }                  // error: The definition of foo()
                                    // is only present when __CUDA_ARCH__
                                    // is undefined
#endif
```

4. 在单独的编译中， `__CUDA_ARCH__` 不得在头文件中使用，这样不同的对象可能包含不同的行为。 或者，必须保证所有对象都将针对相同的 compute_arch 进行编译。 如果在头文件中定义了弱函数或模板函数，并且其行为取决于 `__CUDA_ARCH__`，那么如果为不同的计算架构编译对象，则对象中该函数的实例可能会发生冲突。
   
    例如，如果 a.h 包含：

```C++
template<typename T>
__device__ T* getptr(void)
{
#if __CUDA_ARCH__ == 200
  return NULL; /* no address */
#else
  __shared__ T arr[256];
  return arr;
#endif
}
```

然后，如果 a.cu 和 b.cu 都包含 a.h 并为同一类型实例化 `getptr`，并且 b.cu 需要一个非 NULL 地址，则编译：
```
nvcc –arch=compute_20 –dc a.cu
nvcc –arch=compute_30 –dc b.cu
nvcc –arch=sm_30 a.o b.o
```
在链接时只使用一个版本的 `getptr`，因此行为将取决于选择哪个版本。 为避免这种情况，必须为相同的计算架构编译 a.cu 和 b.cu，或者 `__CUDA_ARCH__` 不应在共享头函数中使用。

编译器不保证将为上述不受支持的 `__CUDA_ARCH__` 使用生成诊断。

### I.4.3. Qualifiers

#### I.4.3.1. Device Memory Space Specifiers

`__device__`、`__shared__`、`__managed__` 和 `__constant__` 内存空间说明符不允许用于：

* 类、结构和联合数据成员，
* 形式参数，
* 在主机上执行的函数中的非外部变量声明。

`__device__`、`__constant__` 和 `__managed__` 内存空间说明符不允许用于在设备上执行的函数中既不是外部也不是静态的变量声明。

`__device__`、`__constant__`、`__managed__` 或 `__shared__` 变量定义不能具有包含非空构造函数或非空析构函数的类的类型。一个类类型的构造函数在翻译单元中的某个点被认为是空的，如果它是一个普通的构造函数或者它满足以下所有条件：

* 构造函数已定义。
* 构造函数没有参数，初始化列表是空的，函数体是一个空的复合语句。
* 它的类没有虚函数，没有虚基类，也没有非静态数据成员初始化器。
* 其类的所有基类的默认构造函数都可以认为是空的。
* 对于其类的所有属于类类型（或其数组）的非静态数据成员，默认构造函数可以被认为是空的。

一个类的析构函数在翻译单元中的某个点被认为是空的，如果它是一个普通的析构函数或者它满足以下所有条件：

* 已定义析构函数。
* 析构函数体是一个空的复合语句。
* 它的类没有虚函数，也没有虚基类。
* 其类的所有基类的析构函数都可以认为是空的。
* 对于其类的所有属于类类型（或其数组）的非静态数据成员，析构函数可以被认为是空的。

在整个程序编译模式下编译时（有关此模式的说明，请参见 nvcc 用户手册），`__device__`、`__shared__`、`__managed__` 和 `__constant__` 变量不能使用 extern 关键字定义为外部变量。 唯一的例外是动态分配的 `__shared__` 变量，如 [`__shared__`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared) 中所述。

在单独编译模式下编译时（有关此模式的说明，请参阅 nvcc 用户手册），可以使用 extern 关键字将 `__device__`、`__shared__`、`__managed__` 和 `__constant__` 变量定义为外部变量。 当 nvlink 找不到外部变量的定义时（除非它是动态分配的 `__shared__` 变量），它会产生错误。

#### I.4.3.2. `__managed__` Memory Space Specifier

用 `__managed__` 内存空间说明符标记的变量（“managed--托管”变量）具有以下限制：

* 托管变量的地址不是常量表达式。
* 托管变量不应具有 const 限定类型。
* 托管变量不应具有引用类型。
* 当 CUDA 运行时可能不处于有效状态时，不应使用托管变量的地址或值，包括以下情况：
    * 在具有静态或线程本地存储持续时间的对象的静态/动态初始化或销毁中。
    * 在调用 exit() 之后执行的代码中（例如，一个标有 gcc 的“`__attribute__`((destructor))”的函数）。
    * 在 CUDA 运行时可能未初始化时执行的代码中（例如，标有 gcc 的“`__attribute__`((constructor))”的函数）。
* 托管变量不能用作 `decltype()` 表达式的未加括号的 id 表达式参数。
* 托管变量具有与为动态分配的托管内存指定的相同的连贯性和一致性行为。
* 当包含托管变量的 CUDA 程序在具有多个 GPU 的执行平台上运行时，变量仅分配一次，而不是每个 GPU。
* 在主机上执行的函数中不允许使用没有外部链接的托管变量声明。
* 在设备上执行的函数中不允许使用没有外部或静态链接的托管变量声明。
  
以下是托管变量的合法和非法使用示例
```C++
__device__ __managed__ int xxx = 10;         // OK

int *ptr = &xxx;                             // error: use of managed variable 
                                             // (xxx) in static initialization
struct S1_t {
  int field;
  S1_t(void) : field(xxx) { };
};
struct S2_t {
  ~S2_t(void) { xxx = 10; }
};

S1_t temp1;                                 // error: use of managed variable 
                                            // (xxx) in dynamic initialization

S2_t temp2;                                 // error: use of managed variable
                                            // (xxx) in the destructor of 
                                            // object with static storage 
                                            // duration

__device__ __managed__ const int yyy = 10;  // error: const qualified type

__device__ __managed__ int &zzz = xxx;      // error: reference type

template <int *addr> struct S3_t { };
S3_t<&xxx> temp;                            // error: address of managed 
                                            // variable(xxx) not a 
                                            // constant expression

__global__ void kern(int *ptr)
{
  assert(ptr == &xxx);                      // OK
  xxx = 20;                                 // OK
}
int main(void) 
{
  int *ptr = &xxx;                          // OK
  kern<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
  xxx++;                                    // OK
  decltype(xxx) qqq;                        // error: managed variable(xxx) used
                                            // as unparenthized argument to
                                            // decltype
                                            
  decltype((xxx)) zzz = yyy;                // OK
}
```

#### I.4.3.3. Volatile Qualifier

编译器可以自由优化对全局或共享内存的读取和写入（例如，通过将全局读取缓存到寄存器或 L1 缓存中），只要它尊重内存围栏函数（[Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)）的内存排序语义和内存可见性语义 同步函数（[Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)）。

可以使用 `volatile` 关键字禁用这些优化：如果将位于全局或共享内存中的变量声明为 `volatile`，编译器假定它的值可以随时被另一个线程更改或使用，因此对该变量的任何引用都会编译为 实际的内存读取或写入指令。

### I.4.4. Pointers

取消引用指向在主机上执行的代码中的全局或共享内存的指针，或在设备上执行的代码中指向主机内存的指针会导致未定义的行为，最常见的是segmentation fault和应用程序终止。

获取 `__device__`、`__shared__` 或 `__constant__` 变量的地址获得的地址只能在设备代码中使用。 设备内存中描述的通过 `cudaGetSymbolAddress()` 获得的 `__device__` 或 `__constant__` 变量的地址只能在主机代码中使用。

### I.4.5. Operators

#### I.4.5.1. Assignment Operator
`__constant__` 变量只能通过运行时函数（设备内存）从主机代码分配； 它们不能从设备代码中分配。

`__shared__` 变量不能将初始化作为其声明的一部分。

不允许为内置变量中定义的任何内置变量赋值。

#### I.4.5.2. Address Operator
不允许使用内置变量中定义的任何内置变量的地址。

### I.4.6. Run Time Type Information (RTTI)
主机代码支持以下与 RTTI 相关的功能，但设备代码不支持。
* `typeid` operator
* `std::type_info`
* `dynamic_cast` operator

### I.4.7. Exception Handling
异常处理仅在主机代码中受支持，但在设备代码中不支持。

`__global__` 函数不支持异常规范。

### I.4.8. Standard Library
除非另有说明，标准库仅在主机代码中受支持，而不在设备代码中受支持。

### I.4.9. Functions

#### I.4.9.1. External Linkage
仅当函数在与设备代码相同的编译单元中定义时，才允许在某些设备代码中调用使用 `extern` 限定符声明的函数，即单个文件或通过可重定位设备代码和 nvlink 链接在一起的多个文件。

#### I.4.9.2. Implicitly-declared and explicitly-defaulted functions
令 F 表示一个在其第一个声明中隐式声明或显式默认的函数 或 F 的执行空间说明符 (`__host__`, `__device__`) 是调用它的所有函数的执行空间说明符的并集（请注意， `__global__` 调用者将被视为 `__device__` 调用者进行此分析）。 例如：

```C++
class Base {
  int x;
public:  
  __host__ __device__ Base(void) : x(10) {}
};

class Derived : public Base {
  int y;
};

class Other: public Base {
  int z;
};

__device__ void foo(void)
{
  Derived D1;
  Other D2;
}

__host__ void bar(void)
{
  Other D3;
}
```

这里，隐式声明的构造函数“`Derived::Derived`”将被视为 `__device__` 函数，因为它仅从 `__device__` 函数“foo”调用。 隐式声明的构造函数 "`Other::Other`" 将被视为 `__host__ __device__` 函数，因为它是从 `__device__` 函数 "foo" 和 `__host__` 函数 "bar" 调用的。
此外，如果 F 是一个虚拟析构函数，则被 F 覆盖的每个虚拟析构函数 D 的执行空间都被添加到 F 的执行空间集合中，如果 D 不是隐式定义的，或者是显式默认的声明而不是它的声明 第一次声明。

例如：
```C++
struct Base1 { virtual __host__ __device__ ~Base1() { } };
struct Derived1 : Base1 { }; // implicitly-declared virtual destructor
                             // ~Derived1 has __host__ __device__ 
                             // execution space specifiers

struct Base2 { virtual __device__ ~Base2(); };
__device__ Base2::~Base2() = default;
struct Derived2 : Base2 { }; // implicitly-declared virtual destructor
                             // ~Derived2 has __device__ execution 
                             // space specifiers 
```

#### I.4.9.3. Function Parameters
`__global__` 函数参数通过常量内存传递给设备，并且限制为 4 KB。

`__global__` 函数不能有可变数量的参数。

`__global__` 函数参数不能通过引用传递。

在单独编译模式下，如果 `__device__` 或 `__global__` 函数在特定翻译单元中被 `ODR` 使用，则该函数的参数和返回类型在该翻译单元中必须是完整的。

```C++
//first.cu:
struct S;
__device__ void foo(S); // error: type 'S' is incomplete
__device__ auto *ptr = foo;

int main() { }

//second.cu:
struct S { int x; };
__device__ void foo(S) { }

//compiler invocation
$nvcc -std=c++14 -rdc=true first.cu second.cu -o first
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o', first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```

##### I.4.9.3.1. __global__ Function Argument Processing
当从设备代码启动 `__global__` 函数时，每个参数都必须是可简单复制和可简单销毁的。

当从主机代码启动 `__global__` 函数时，每个参数类型都可以是不可复制或不可销毁的，但对此类类型的处理不遵循标准 C++ 模型，如下所述。 用户代码必须确保此工作流程不会影响程序的正确性。 工作流在两个方面与标准 C++ 不同：

   1. Memcpy instead of copy constructor invocation;  
   从主机代码降低 `__global__` 函数启动时，编译器会生成存根函数，这些函数按值复制参数一次或多次，然后最终使用 `memcpy` 将参数复制到设备上的 `__global__` 函数的参数内存中。 即使参数是不可复制的，也会发生这种情况，因此可能会破坏复制构造函数具有副作用的程序。  
   例子：

```C++
#include <cassert>
struct S {
 int x;
 int *ptr;
 __host__ __device__ S() { }
 __host__ __device__ S(const S &) { ptr = &x; }
};

__global__ void foo(S in) {
 // this assert may fail, because the compiler
 // generated code will memcpy the contents of "in"
 // from host to kernel parameter memory, so the
 // "in.ptr" is not initialized to "&in.x" because
 // the copy constructor is skipped.
 assert(in.ptr == &in.x);
}

int main() {
  S tmp;
  foo<<<1,1>>>(tmp);
  cudaDeviceSynchronize();
}
```

```C++
#include <cassert>

__managed__ int counter;
struct S1 {
S1() { }
S1(const S1 &) { ++counter; }
};

__global__ void foo(S1) {

/* this assertion may fail, because
   the compiler generates stub
   functions on the host for a kernel
   launch, and they may copy the
   argument by value more than once.
*/
assert(counter == 1);
}

int main() {
S1 V;
foo<<<1,1>>>(V);
cudaDeviceSynchronize();
}
```

   2. Destructor may be invoked before the __global__ function has finished;          
   内核启动与主机执行是异步的。 因此，如果 `__global__` 函数参数具有非平凡的析构函数，则析构函数甚至可以在 `__global__` 函数完成执行之前在宿主代码中执行。 这可能会破坏析构函数具有副作用的程序。
   示例:

```C++
struct S {
 int *ptr;
 S() : ptr(nullptr) { }
 S(const S &) { cudaMallocManaged(&ptr, sizeof(int)); }
 ~S() { cudaFree(ptr); }
};

__global__ void foo(S in) {
 
  //error: This store may write to memory that has already been
  //       freed (see below).
  *(in.ptr) = 4;
 
}

int main() {
 S V;
 
 /* The object 'V' is first copied by value to a compiler-generated
  * stub function that does the kernel launch, and the stub function
  * bitwise copies the contents of the argument to kernel parameter
  * memory.
  * However, GPU kernel execution is asynchronous with host
  * execution. 
  * As a result, S::~S() will execute when the stub function   returns, releasing allocated memory, even though the kernel may not have finished execution.
  */
 foo<<<1,1>>>(V);
 cudaDeviceSynchronize();
}
```

#### I.4.9.4. Static Variables within Function
在函数 F 的直接或嵌套块范围内，静态变量 V 的声明中允许使用可变内存空间说明符，其中：
* F 是一个 `__global__` 或 `__device__`-only 函数。
* F 是一个 `__host__ __device__` 函数，`__CUDA_ARCH__` 定义为 17。

如果 V 的声明中没有显式的内存空间说明符，则在设备编译期间假定隐式 `__device__` 说明符。

V 具有与在命名空间范围内声明的具有相同内存空间说明符的变量相同的初始化限制，例如 `__device__` 变量不能有“非空”构造函数（请参阅设备内存空间说明符）。

函数范围静态变量的合法和非法使用示例如下所示。
```C++
struct S1_t {
  int x;
};

struct S2_t {
  int x;
  __device__ S2_t(void) { x = 10; }
};

struct S3_t {
  int x;
  __device__ S3_t(int p) : x(p) { }
};

__device__ void f1() {
  static int i1;              // OK, implicit __device__ memory space specifier
  static int i2 = 11;         // OK, implicit __device__ memory space specifier
  static __managed__ int m1;  // OK
  static __device__ int d1;   // OK
  static __constant__ int c1; // OK
  
  static S1_t i3;             // OK, implicit __device__ memory space specifier
  static S1_t i4 = {22};      // OK, implicit __device__ memory space specifier

  static __shared__ int i5;   // OK

  int x = 33;
  static int i6 = x;          // error: dynamic initialization is not allowed
  static S1_t i7 = {x};       // error: dynamic initialization is not allowed

  static S2_t i8;             // error: dynamic initialization is not allowed
  static S3_t i9(44);         // error: dynamic initialization is not allowed
}

__host__ __device__ void f2() {
  static int i1;              // OK, implicit __device__ memory space specifier
                              // during device compilation.
#ifdef __CUDA_ARCH__
  static __device__ int d1;   // OK, declaration is only visible during device
                              // compilation  (__CUDA_ARCH__ is defined)
#else
  static int d0;              // OK, declaration is only visible during host
                              // compilation (__CUDA_ARCH__ is not defined)
#endif  

  static __device__ int d2;   // error: __device__ variable inside
                              // a host function during host compilation
                              // i.e. when __CUDA_ARCH__ is not defined

  static __shared__ int i2;  // error: __shared__ variable inside
                             // a host function during host compilation
                             // i.e. when __CUDA_ARCH__ is not defined
}
```

#### I.4.9.5. Function Pointers
在主机代码中获取的 `__global__` 函数的地址不能在设备代码中使用（例如，启动内核）。 同样，在设备代码中获取的 `__global__` 函数的地址不能在主机代码中使用。

不允许在主机代码中获取 `__device__` 函数的地址。

#### I.4.9.6. Function Recursion
`__global__` 函数不支持递归。

#### I.4.9.7. Friend Functions
`__global__` 函数或函数模板不能在友元声明中定义。

例子：
```C++
struct S1_t {
  friend __global__ 
  void foo1(void);  // OK: not a definition
  template<typename T>
  friend __global__ 
  void foo2(void); // OK: not a definition
  
  friend __global__ 
  void foo3(void) { } // error: definition in friend declaration
  
  template<typename T>
  friend __global__ 
  void foo4(void) { } // error: definition in friend declaration
};
```

#### I.4.9.8. Operator Function
运算符函数不能是 `__global__` 函数。


### I.4.10. Classes
I.4.10.1. 数据成员
不支持静态数据成员，除了那些也是 const 限定的（请参阅 [Const 限定变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#const-variables)）。

#### I.4.10.2. 函数成员
静态成员函数不能是 `__global__` 函数。

#### I.4.10.3. 虚函数
当派生类中的函数覆盖基类中的虚函数时，被覆盖函数和覆盖函数上的执行空间说明符（即 `__host__、__device__`）必须匹配。

不允许将具有虚函数的类的对象作为参数传递给 `__global__` 函数。

如果在主机代码中创建对象，则在设备代码中为该对象调用虚函数具有未定义的行为。

如果在设备代码中创建了一个对象，则在主机代码中为该对象调用虚函数具有未定义的行为。

使用 Microsoft 主机编译器时，请参阅特定于 Windows 的其他限制。

例子：
```C++

struct S1 { virtual __host__ __device__ void foo() { } };

__managed__ S1 *ptr1, *ptr2;

__managed__ __align__(16) char buf1[128];
__global__ void kern() { 
  ptr1->foo();     // error: virtual function call on a object
                   //        created in host code.
  ptr2 = new(buf1) S1();
}

int main(void) {
  void *buf;
  cudaMallocManaged(&buf, sizeof(S1), cudaMemAttachGlobal);
  ptr1 = new (buf) S1();
  kern<<<1,1>>>();
  cudaDeviceSynchronize();
  ptr2->foo();  // error: virtual function call on an object
                //        created in device code.
}
```

#### I.4.10.4.  Virtual Base Classes
不允许将派生自虚拟基类的类的对象作为参数传递给 `__global__` 函数。

使用 Microsoft 主机编译器时，请参阅特定于 Windows 的其他限制。

#### I.4.10.5. Anonymous Unions
命名空间范围匿名联合的成员变量不能在 `__global__` 或 `__device__` 函数中引用。

#### I.4.10.6. 特定于 Windows 的
CUDA 编译器遵循 IA64 ABI 进行类布局，而 Microsoft 主机编译器则不遵循。 令 T 表示指向成员类型的指针，或满足以下任一条件的类类型：
* T has virtual functions.
* T has a virtual base class.
* T has multiple inheritance with more than one direct or indirect empty base class.
* All direct and indirect base classes B of T are empty and the type of the first field F of T uses B in its definition, such that B is laid out at offset 0 in the definition of F.

让 `C` 表示 `T` 或以 `T` 作为字段类型或基类类型的类类型。 CUDA 编译器计算类布局和大小的方式可能不同于 `C` 类型的 Microsoft 主机编译器。
只要类型 `C` 专门用于主机或设备代码，程序就应该可以正常工作。

在主机和设备代码之间传递 `C` 类型的对象具有未定义的行为，例如，作为 `__global__` 函数的参数或通过 `cudaMemcpy*()` 调用。

如果在主机代码中创建对象，则访问 `C` 类型的对象或设备代码中的任何子对象，或调用设备代码中的成员函数具有未定义的行为。

如果对象是在设备代码中创建的，则访问 `C `类型的对象或主机代码中的任何子对象，或调用主机代码中的成员函数具有未定义的行为。

### I.4.11. Templates
类型或模板不能在 `__global__` 函数模板实例化或 `__device__`/`__constant__` 变量实例化的类型、非类型或模板模板参数中使用，如果：
* 类型或模板在 `__host__` 或 `__host__ __device__` 中定义。
* 类型或模板是具有私有或受保护访问的类成员，其父类未在 `__device__` 或 `__global__` 函数中定义。
* 该类型未命名。
*该类型由上述任何类型复合而成。
例子：
```C++
template <typename T>
__global__ void myKernel(void) { }

class myClass {
private:
    struct inner_t { }; 
public:
    static void launch(void) 
    {
       // error: inner_t is used in template argument
       // but it is private
       myKernel<inner_t><<<1,1>>>();
    }
};

// C++14 only
template <typename T> __device__ T d1;

template <typename T1, typename T2> __device__ T1 d2;

void fn() {
  struct S1_t { };
  // error (C++14 only): S1_t is local to the function fn
  d1<S1_t> = {};

  auto lam1 = [] { };
  // error (C++14 only): a closure type cannot be used for
  // instantiating a variable template
  d2<int, decltype(lam1)> = 10;
}

```

### I.4.12. Trigraphs and Digraphs
任何平台都不支持三元组。 Windows 不支持有向图。

### I.4.13. Const-qualified variables
让“V”表示名称空间范围变量或具有 `const` 限定类型且没有执行空间注释的类静态成员变量（例如，`__device__`、`__constant__`、`__shared__`）。 V 被认为是主机代码变量。

V 的值可以直接在设备代码中使用，如果
* V 在使用点之前已经用常量表达式初始化，
* V 的类型不是 volatile 限定的，并且
* 它具有以下类型之一：
    * 内置浮点类型，除非将 Microsoft 编译器用作主机编译器，
    * 内置整型。

设备源代码不能包含对 V 的引用或获取 V 的地址。

例子：
```C++
const int xxx = 10;
struct S1_t {  static const int yyy = 20; };
    
extern const int zzz;
const float www = 5.0;
__device__ void foo(void) {
  int local1[xxx];          // OK
  int local2[S1_t::yyy];    // OK
      
  int val1 = xxx;           // OK
    					
  int val2 = S1_t::yyy;     // OK
    					
  int val3 = zzz;           // error: zzz not initialized with constant 
                            // expression at the point of use.
  
  const int &val3 = xxx;    // error: reference to host variable  
  const int *val4 = &xxx;   // error: address of host variable
  const float val5 = www;   // OK except when the Microsoft compiler is used as
                            // the host compiler.
}
const int zzz = 20;

```

### I.4.14. Long Double
设备代码不支持使用 long double 类型。

### I.4.15. Deprecation Annotation

`nvcc` 支持在使用 `gcc、clang、xlC、icc` 或 `pgcc` 主机编译器时使用 `deprecated` 属性，以及在使用 `cl.exe` 主机编译器时使用 `deprecated declspec`。当启用 C++14 时，它还支持 `[[deprecated]]` 标准属性。当定义 `__CUDA_ARCH__` 时（即在设备编译阶段），CUDA 前端编译器将为从 `__device__`、`__global__` 或 `__host__ __device__` 函数的主体内对已弃用实体的引用生成弃用诊断。对不推荐使用的实体的其他引用将由主机编译器处理，例如，来自 `__host__` 函数中的引用。

CUDA 前端编译器不支持各种主机编译器支持的`#pragma` gcc 诊断或`#pragma` 警告机制。因此，CUDA 前端编译器生成的弃用诊断不受这些 `pragma` 的影响，但主机编译器生成的诊断会受到影响。要抑制设备代码的警告，用户可以使用 NVIDIA 特定的 pragma `#pragma nv_diag_suppress`。 nvcc 标志 `-Wno-deprecated-declarations` 可用于禁止所有弃用警告，标志 `-Werror=deprecated-declarations` 可用于将弃用警告转换为错误。

### I.4.16. Noreturn Annotation
`nvcc` 支持在使用 `gcc、clang、xlC、icc 或 pgcc` 主机编译器时使用 `noreturn` 属性，并在使用 `cl.exe` 主机编译器时使用 `noreturn` `declspec`。 当启用 C++11 时，它还支持 `[[noreturn]]` 标准属性。

`属性/declspec` 可以在主机和设备代码中使用。

### I.4.17. [[likely]] / [[unlikely]] Standard Attributes

所有支持 C++ 标准属性语法的配置都接受这些属性。 这些属性可用于向设备编译器优化器提示与不包含该语句的任何替代路径相比，该语句是否更有可能被执行。

例子：
```C++
__device__ int foo(int x) {

 if (i < 10) [[likely]] { // the 'if' block will likely be entered
  return 4; 
 }
 if (i < 20) [[unlikely]] { // the 'if' block will not likely be entered
  return 1;
 }
 return 0;
}

```
如果在 `__CUDA_ARCH__` 未定义时在主机代码中使用这些属性，则它们将出现在主机编译器解析的代码中，如果不支持这些属性，则可能会生成警告。 例如，`clang11` 主机编译器将生成`“unknown attribute”`警告。

# I.4.18. const and pure GNU Attributes

当使用也支持这些属性的语言和主机编译器时，主机和设备功能都支持这些属性，例如 使用 `g++` 主机编译器。

对于使用 `pure` 属性注释的设备函数，设备代码优化器假定该函数不会更改调用者函数（例如内存）可见的任何可变状态。

对于使用 `const` 属性注释的设备函数，设备代码优化器假定该函数不会访问或更改调用者函数可见的任何可变状态（例如内存）。

例子：
```C++
__attribute__((const)) __device__ int get(int in);

__device__ int doit(int in) {
int sum = 0;

//because 'get' is marked with 'const' attribute
//device code optimizer can recognize that the
//second call to get() can be commoned out.
sum = get(in);
sum += get(in);

return sum;
}
```

### I.4.19. Intel Host Compiler Specific
CUDA 前端编译器解析器无法识别英特尔编译器（例如 icc）支持的某些内在函数。 因此，当使用 Intel 编译器作为主机编译器时，`nvcc` 将在预处理期间启用宏 `__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`。 此宏允许在相关头文件中显式声明英特尔编译器内部函数，从而允许 `nvcc` 支持在主机代码中使用[此类函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_21)。

### I.4.20. C++11 Features

`nvcc` 也支持主机编译器默认启用的 C++11 功能，但须遵守本文档中描述的限制。 此外，使用 `-std=c++11` 标志调用 `nvcc` 会打开所有 C++11 功能，还会使用相应的 C++11 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_21)调用主机预处理器、编译器和链接器。

#### I.4.20.1. Lambda Expressions

与 lambda 表达式关联的闭包类的所有[成员函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_22)的执行空间说明符由编译器派生如下。 如 C++11 标准中所述，编译器在包含 lambda 表达式的最小块范围、类范围或命名空间范围内创建闭包类型。 计算封闭闭包类型的最内层函数作用域，并将相应函数的执行空间说明符分配给闭包类成员函数。 如果没有封闭函数范围，则执行空间说明符为 `__host__`。

lambda 表达式和计算的执行空间说明符的示例如下所示（在注释中）。

```C++
auto globalVar = [] { return 0; }; // __host__ 
    
void f1(void) {
  auto l1 = [] { return 1; };      // __host__
}
    
__device__ void f2(void) {
  auto l2 = [] { return 2; };      // __device__
}
    
__host__ __device__ void f3(void) {
  auto l3 = [] { return 3; };      // __host__ __device__
}
    
__device__ void f4(int (*fp)() = [] { return 4; } /* __host__ */) {
}
    
__global__ void f5(void) {
  auto l5 = [] { return 5; };      // __device__
}
    
__device__ void f6(void) {
  struct S1_t {
    static void helper(int (*fp)() = [] {return 6; } /* __device__ */) {
    }
  };
}

```

lambda 表达式的闭包类型不能用于 `__global__` 函数模板实例化的类型或非类型参数，除非 lambda 在 `__device__` 或 `__global__` 函数中定义。

例子：
```C++
template <typename T>
__global__ void foo(T in) { };
    
template <typename T>
struct S1_t { };
    
void bar(void) {
  auto temp1 = [] { };
      
  foo<<<1,1>>>(temp1);                    // error: lambda closure type used in
                                          // template type argument
  foo<<<1,1>>>( S1_t<decltype(temp1)>()); // error: lambda closure type used in 
                                          // template type argument
}

```

#### I.4.20.2. std::initializer_list
默认情况下，CUDA 编译器将隐式认为 `std::initializer_list` 的成员函数具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码调用它们。 nvcc 标志 `--no-host-device-initializer-list` 将禁用此行为； `std::initializer_list` 的成员函数将被视为 `__host__` 函数，并且不能直接从设备代码调用。

例子：

```C++
#include <initializer_list>
    
__device__ int foo(std::initializer_list<int> in);
    
__device__ void bar(void)
  {
    foo({4,5,6});   // (a) initializer list containing only 
                    // constant expressions.
    
    int i = 4;
    foo({i,5,6});   // (b) initializer list with at least one 
                    // non-constant element.
                    // This form may have better performance than (a). 
  }

```

#### I.4.20.3. Rvalue references
默认情况下，CUDA 编译器将隐式认为 `std::move` 和 `std::forward` 函数模板具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码调用它们。 nvcc 标志 `--no-host-device-move-forward` 将禁用此行为； `std::move` 和 `std::forward` 将被视为 `__host__` 函数，不能直接从设备代码调用。

#### I.4.20.4. Constexpr functions and function templates

默认情况下，不能从执行空间不兼容的函数中调用 `constexpr` [函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_23). 实验性 nvcc 标志 `--expt-relaxed-constexpr` 消除了[此限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_24). 当指定此标志时，主机代码可以调用 `__device__ constexpr` 函数和设备 代码可以调用 `__host__ constexpr` 函数。 当指定了 `--expt-relaxed-constexpr` 时，nvcc 将定义宏 `__CUDACC_RELAXED_CONSTEXPR__`。 请注意，即使相应的模板用关键字 `constexpr` 标记（C++11 标准节 `[dcl.constexpr.p6]`），函数模板实例化也可能不是 `constexpr` 函数。

#### I.4.20.5. Constexpr variables
让“`V`”表示命名空间范围变量或已标记为 `constexpr` 且没有执行空间注释的类静态成员变量（例如，`__device__`、`__constant__`、`__shared__`）。 V 被认为是主机代码变量。

如果 V 是除 `long double` 以外的[标量类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_25) 并且该类型不是 `volatile` 限定的，则 V 的值可以直接在设备代码中使用。 此外，如果 V 是非标量类型，则 V 的标量元素可以在 `constexpr __device__` 或 `__host__ __device__` 函数中使用，如果对函数的调用是[常量表达式](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_26). 设备源代码不能包含对 V 的引用 或取 V 的地址。

例子：
```C++
constexpr int xxx = 10;
constexpr int yyy = xxx + 4;
struct S1_t { static constexpr int qqq = 100; };

constexpr int host_arr[] = { 1, 2, 3};
constexpr __device__ int get(int idx) { return host_arr[idx]; } 
  
__device__ int foo(int idx) {
  int v1 = xxx + yyy + S1_t::qqq;  // OK
  const int &v2 = xxx;             // error: reference to host constexpr 
                                   // variable
  const int *v3 = &xxx;            // error: address of host constexpr 
                                   // variable
  const int &v4 = S1_t::qqq;       // error: reference to host constexpr 
                                   // variable
  const int *v5 = &S1_t::qqq;      // error: address of host constexpr 
                                   // variable
                                   
  v1 += get(2);                    // OK: 'get(2)' is a constant 
                                   // expression.
  v1 += get(idx);                  // error: 'get(idx)' is not a constant 
                                   // expression
  v1 += host_arr[2];               // error: 'host_arr' does not have 
                                   // scalar type.
  return v1;
}

```

#### I.4.20.6. Inline namespaces
对于输入的CUDA翻译单元，CUDA编译器可以调用主机编译器来编译翻译单元内的主机代码。 在传递给主机编译器的代码中，如果输入的 CUDA 翻译单元包含以下任何实体的定义，CUDA 编译器将注入额外的编译器生成的代码：
* `__global__` 函数或函数模板实例化
* `__device__`, `__constant__`
* 具有表面或纹理类型的变量

编译器生成的代码包含对已定义实体的引用。 如果实体是在内联命名空间中定义的，而另一个具有相同名称和类型签名的实体在封闭命名空间中定义，则主机编译器可能会认为此引用不明确，主机编译将失败。
可以通过对内联命名空间中定义的此类实体使用唯一名称来避免此限制。

例子：
```C++
__device__ int Gvar;
inline namespace N1 {
  __device__ int Gvar;  
}

// <-- CUDA compiler inserts a reference to "Gvar" at this point in the 
// translation unit. This reference will be considered ambiguous by the 
// host compiler and compilation will fail.
```

```C++
inline namespace N1 {
  namespace N2 {
    __device__ int Gvar;
  }
}

namespace N2 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts reference to "::N2::Gvar" at this point in 
// the translation unit. This reference will be considered ambiguous by 
// the host compiler and compilation will fail.
```

##### I.4.20.6.1. Inline unnamed namespaces

不能在内联未命名命名空间内的命名空间范围内声明以下实体：
* `__managed__`、`__device__`、`__shared__` 和 `__constant__` 变量
* `__global__` 函数和函数模板
* 具有表面或纹理类型的变量

例子：
```C++
inline namespace {
  namespace N2 {
    template <typename T>
    __global__ void foo(void);            // error
    
    __global__ void bar(void) { }         // error
    
    template <>
    __global__ void foo<int>(void) { }    // error
      
    __device__ int x1b;                   // error
    __constant__ int x2b;                 // error
    __shared__ int x3b;                   // error 
	
    texture<int> q2;                      // error
    surface<int> s2;                      // error
  }
};
```

#### I.4.20.7. thread_local

设备代码中不允许使用 `thread_local` 存储说明符。

#### I.4.20.8. __global__ functions and function templates

如果在 `__global__` 函数模板实例化的模板参数中使用与 lambda 表达式关联的闭包类型，则 lambda 表达式必须在 `__device__` 或 `__global__` 函数的直接或嵌套块范围内定义，或者必须是扩展 lambda。

例子：
```C++
template <typename T>
__global__ void kernel(T in) { }

__device__ void foo_device(void)
{
  // All kernel instantiations in this function
  // are valid, since the lambdas are defined inside
  // a __device__ function.
  
  kernel<<<1,1>>>( [] __device__ { } );
  kernel<<<1,1>>>( [] __host__ __device__ { } );
  kernel<<<1,1>>>( []  { } );
}

auto lam1 = [] { };

auto lam2 = [] __host__ __device__ { };

void foo_host(void)
{
   // OK: instantiated with closure type of an extended __device__ lambda
   kernel<<<1,1>>>( [] __device__ { } );
   
   // OK: instantiated with closure type of an extended __host__ __device__ 
   // lambda
   kernel<<<1,1>>>( [] __host__ __device__ { } );
 
   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( []  { } );
   
   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam1);
   
   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam2);
}
```

`__global__` 函数或函数模板不能声明为 `constexpr`。

`__global__` 函数或函数模板不能有 `std::initializer_list` 或 `va_list` 类型的参数。

`__global__` 函数不能有右值引用类型的参数。

可变参数 `__global__` 函数模板具有以下限制：
* 只允许一个包参数。
* pack 参数必须在模板参数列表中最后列出。

例子：
```C++
// ok
template <template <typename...> class Wrapper, typename... Pack>
__global__ void foo1(Wrapper<Pack...>);
    
// error: pack parameter is not last in parameter list
template <typename... Pack, template <typename...> class Wrapper>
__global__ void foo2(Wrapper<Pack...>);

// error: multiple parameter packs
template <typename... Pack1, int...Pack2, template<typename...> class Wrapper1, 
          template<int...> class Wrapper2>
__global__ void foo3(Wrapper1<Pack1...>, Wrapper2<Pack2...>);
```

#### I.4.20.9. __managed__ and __shared__ variables
`__managed__` 和 `__shared__` 变量不能用关键字 `constexpr` 标记。

#### I.4.20.10. Defaulted functions
CUDA 编译器会忽略在第一个声明中显式默认的函数上的执行空间说明符。 相反，CUDA 编译器将推断执行空间说明符，如[隐式声明和显式默认函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compiler-generated-functions)中所述。

如果函数是显式默认的，则不会忽略执行空间说明符，但不会在其第一次声明时忽略。

例子：

```C++
struct S1 {
  // warning: __host__ annotation is ignored on a function that 
  //          is explicitly-defaulted on its first declaration
  __host__ S1() = default;
};

__device__ void foo1() { 
  //note: __device__ execution space is derived for S1::S1 
  //       based on implicit call from within __device__ function 
  //       foo1
  S1 s1;    
}

struct S2 {
  __host__ S2();
};

//note: S2::S2 is not defaulted on its first declaration, and 
//      its execution space is fixed to __host__  based on its 
//      first declaration.
S2::S2() = default;  

__device__ void foo2() {
   // error: call from __device__ function 'foo2' to 
   //        __host__ function 'S2::S2'
   S2 s2;  
}
```

### I.4.21. C++14 Features

nvcc 也支持主机编译器默认启用的 C++14 功能。 传递 nvcc `-std=c++14` 标志打开所有 C++14 功能，并使用相应的 C++14 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_27) 调用主机预处理器、编译器和链接器。本节描述了对受支持的 C++ 14 的限制特点。

#### I.4.21.1. Functions with deduced return type

`__global__` 函数不能有推导的返回类型。

如果 `__device__` 函数推导出返回类型，CUDA 前端编译器将在调用主机编译器之前将函数声明更改为具有 void 返回类型。 这可能会导致在主机代码中自省 `__device__` 函数的推导返回类型时出现问题。 因此，CUDA 编译器将发出编译时错误，用于在设备函数体之外引用此类推导的返回类型，除非在 `__CUDA_ARCH__` 未定义时引用不存在。

例子：

```C++
__device__ auto fn1(int x) {
  return x;
}

__device__ decltype(auto) fn2(int x) {
  return x;
}

__device__ void device_fn1() {
  // OK
  int (*p1)(int) = fn1;
}

// error: referenced outside device function bodies
decltype(fn1(10)) g1;

void host_fn1() {
  // error: referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // error: referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// error: referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct S1_t { };

// error: referenced outside device function bodies
struct S1_derived_t : S1_t<decltype(fn1)> { };
```

#### I.4.21.2. Variable templates
使用 Microsoft 主机编译器时，`__device__/__constant__` 变量模板不能具有 `const` 限定类型。

例子：
```C++
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ const T d1(2);

int *const x = nullptr;
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ T *const d2(x);

// OK
template <typename T>
__device__ const T *d3;

__device__ void fn() {
  int t1 = d1<int>;

  int *const t2 = d2<int>;

  const int *t3 = d3<int>;
}
```

### I.4.22. C++17 Features
nvcc 也支持主机编译器默认启用的 C++17 功能。 传递 nvcc `-std=c++17` 标志会打开所有 C++17 功能，并使用相应的 C++17 [选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_28)调用主机预处理器、编译器和链接器。本节描述对支持的 C++ 17的限制特点。

#### I.4.22.1. Inline Variable
如果代码在整个程序编译模式下使用 nvcc 编译，则使用 `__device__` 或 `__constant__` 或 `__managed__` 内存空间说明符声明的命名空间范围内联变量必须具有内部链接。

例子：

```C++
inline __device__ int xxx; //error when compiled with nvcc in
                                    //whole program compilation mode.
                                    //ok when compiled with nvcc in
                                    //separate compilation mode.

inline __shared__ int yyy0; // ok.

static inline __device__ int yyy; // ok: internal linkage
namespace {
inline __device__ int zzz; // ok: internal linkage
}

```
使用 g++ 主机编译器时，使用 `__managed__` 内存空间说明符声明的内联变量可能对调试器不可见。

#### I.4.22.2. Structured Binding
不能使用可变内存空间说明符声明结构化绑定。

例子：
```C++
struct S { int x; int y; };
__device__ auto [a1, b1] = S{4,5}; // error

```

## I.5. Polymorphic Function Wrappers

在 `nvfunctional` 头文件中提供了一个多态函数包装类模板 `nvstd::function`。 此类模板的实例可用于存储、复制和调用任何可调用目标，例如 lambda 表达式。 `nvstd::function` 可以在主机和设备代码中使用。

例子：

```C++
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ __device__ int foo_hd () { return 2; }
__host__ int foo_h() { return 3; }

__global__ void kernel(int *result) {
  nvstd::function<int()> fn1 = foo_d;  
  nvstd::function<int()> fn2 = foo_hd;
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}

__host__ __device__ void hostdevice_func(int *result) {
  nvstd::function<int()> fn1 = foo_hd;  
  nvstd::function<int()> fn2 =  []() { return 10; };

  *result = fn1() + fn2();
}

__host__ void host_func(int *result) {
  nvstd::function<int()> fn1 = foo_h;  
  nvstd::function<int()> fn2 = foo_hd;  
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}

```

主机代码中的 `nvstd::function` 实例不能用 `__device__` 函数的地址或 `operator()` 为 `__device__` 函数的函子初始化。 设备代码中的 `nvstd::function` 实例不能用 `__host__` 函数的地址或 `operator()` 为 `__host__` 函数的仿函数初始化。

`nvstd::function` 实例不能在运行时从主机代码传递到设备代码（反之亦然）。 如果 `__global__` 函数是从主机代码启动的，则 `nvstd::function` 不能用于 `__global__` 函数的参数类型。

例子：
```C++
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ int foo_h() { return 3; }
auto lam_h = [] { return 0; };

__global__ void k(void) {
  // error: initialized with address of __host__ function 
  nvstd::function<int()> fn1 = foo_h;  

  // error: initialized with address of functor with
  // __host__ operator() function 
  nvstd::function<int()> fn2 = lam_h;
}

__global__ void kern(nvstd::function<int()> f1) { }

void foo(void) {
  // error: initialized with address of __device__ function 
  nvstd::function<int()> fn1 = foo_d;  

  auto lam_d = [=] __device__ { return 1; };

  // error: initialized with address of functor with
  // __device__ operator() function 
  nvstd::function<int()> fn2 = lam_d;

  // error: passing nvstd::function from host to device
  kern<<<1,1>>>(fn2);
}

```

`nvstd::function` 在 `nvfunctional` 头文件中定义如下：

```C++
namespace nvstd {
  template <class _RetType, class ..._ArgTypes>
  class function<_RetType(_ArgTypes...)> 
  {
    public:
      // constructors
      __device__ __host__  function() noexcept;
      __device__ __host__  function(nullptr_t) noexcept;
      __device__ __host__  function(const function &);
      __device__ __host__  function(function &&);

      template<class _F>
      __device__ __host__  function(_F);

      // destructor
      __device__ __host__  ~function();

      // assignment operators
      __device__ __host__  function& operator=(const function&);
      __device__ __host__  function& operator=(function&&);
      __device__ __host__  function& operator=(nullptr_t);
      __device__ __host__  function& operator=(_F&&);

      // swap
      __device__ __host__  void swap(function&) noexcept;

      // function capacity
      __device__ __host__  explicit operator bool() const noexcept;

      // function invocation
      __device__ _RetType operator()(_ArgTypes...) const;
  };

  // null pointer comparisons
  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  // specialized algorithms
  template <class _R, class... _ArgTypes>
  __device__ __host__
  void swap(function<_R(_ArgTypes...)>&, function<_R(_ArgTypes...)>&);
}

```

## I.6. Extended Lambdas

nvcc 标志 '`--extended-lambda`' 允许在 lambda [表达式](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_29)中显式执行空间注释。执行空间注释应该出现在 '`lambda-introducer`' 之后和可选的 '`lambda-declarator`' 之前。当指定了“`--extended-lambda`”标志时，nvcc 将定义宏 `__CUDACC_EXTENDED_LAMBDA__`。

'`extended __device__ lambda`' 是一个用 '`__device__`' 显式注释的 lambda 表达式，并在 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块范围内定义。

'`extended __host__ __device__ lambda`' 是一个用 '`__host__`' 和 '`__device__`' 显式注释的 lambda 表达式，并在 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块范围内定义。

“`extended lambda`”表示扩展的 `__device__` lambda 或扩展的 `__host__ __device__` lambda。扩展的 lambda 可用于 [`__global__` 函数模板实例化](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp11-global)的类型参数。

如果未明确指定执行空间注释，则它们是根据包含与 lambda 关联的闭包类的范围计算的，如 C++11 支持部分所述。执行空间注释应用于与 lambda 关联的闭包类的所有方法。

例子：
```C++
void foo_host(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };
  
  // extended __device__ lambda
  auto lam2 = [] __device__ { };
  
  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };
  
  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__host__ __device__ void foo_host_device(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };
  
  // extended __device__ lambda
  auto lam2 = [] __device__ { };
  
  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };
  
  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__device__ void foo_device(void) {
  // none of the lambdas within this function are extended lambdas, 
  // because the enclosing function is not a __host__ or __host__ __device__
  // function.
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __host__ { };
}

// lam1 and lam2 are not extended lambdas because they are not defined
// within a __host__ or __host__ __device__ function.
auto lam1 = [] { };
auto lam2 = [] __host__ __device__ { };

```

### I.6.1. Extended Lambda Type Traits

编译器提供类型特征来在编译时检测扩展 lambda 的闭包类型：

`__nv_is_extended_device_lambda_closure_type(type)`：如果 'type' 是为扩展的 `__device__` lambda 创建的闭包类，则 `trait` 为真，否则为假。

`__nv_is_extended_host_device_lambda_closure_type(type)`：如果 'type' 是为扩展的 `__host__ __device__` lambda 创建的闭包类，则 `trait` 为真，否则为假。

这些特征可以在所有编译模式中使用，[无论是启用 lambda 还是扩展 lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_30)。

例子：
```C++
#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

auto lam0 = [] __host__ __device__ { };

void foo(void) {
  auto lam1 = [] { }; 
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };

  // lam0 is not an extended lambda (since defined outside function scope)
  static_assert(!IS_D_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam0)), "");

  // lam1 is not an extended lambda (since no execution space annotations)
  static_assert(!IS_D_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam1)), "");

  // lam2 is an extended __device__ lambda
  static_assert(IS_D_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam2)), "");

  // lam3 is an extended __host__ __device__ lambda
  static_assert(!IS_D_LAMBDA(decltype(lam3)), "");
  static_assert(IS_HD_LAMBDA(decltype(lam3)), "");
}

```

### I.6.2. Extended Lambda Restrictions

在调用主机编译器之前，CUDA 编译器将用命名空间范围内定义的占位符类型的实例替换扩展的 lambda 表达式。占位符类型的模板参数需要获取包含原始扩展 lambda 表达式的函数的地址。这是正确执行任何模板参数涉及扩展 lambda 的闭包类型的 `__global__` 函数模板所必需的。封闭函数计算如下。

根据定义，扩展 lambda 存在于 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块范围内。如果此函数不是 lambda 表达式的 `operator()`，则将其视为扩展 lambda 的封闭函数。否则，扩展 lambda 定义在一个或多个封闭 lambda 表达式的 `operator()` 的直接或嵌套块范围内。如果最外层的这种 lambda 表达式定义在函数 F 的直接或嵌套块范围内，则 F 是计算的封闭函数，否则封闭函数不存在。

例子：

```C++
void foo(void) {
  // enclosing function for lam1 is "foo"
  auto lam1 = [] __device__ { };
  
  auto lam2 = [] {
     auto lam3 = [] {
        // enclosing function for lam4 is "foo"
        auto lam4 = [] __host__ __device__ { };
     };
  };
}

auto lam6 = [] {
  // enclosing function for lam7 does not exist
  auto lam7 = [] __host__ __device__ { };
};

```

以下是对扩展 lambda 的限制：

扩展 lambda 不能在另一个扩展 lambda 表达式中定义。
例子：
```C++
void foo(void) {
  auto lam1 = [] __host__ __device__  {
    // error: extended lambda defined within another extended lambda
    auto lam2 = [] __host__ __device__ { };
  };
}

```
不能在通用 lambda 表达式中定义扩展 lambda。
例子：
```C++
void foo(void) {
  auto lam1 = [] (auto) {
    // error: extended lambda defined within a generic lambda
    auto lam2 = [] __host__ __device__ { };
  };
}

```
如果扩展 lambda 定义在一个或多个嵌套 lambda 表达式的直接或嵌套块范围内，则最外层的此类 lambda 表达式必须定义在函数的直接或嵌套块范围内。
例子：
```C++

auto lam1 = []  {
  // error: outer enclosing lambda is not defined within a
  // non-lambda-operator() function. 
  auto lam2 = [] __host__ __device__ { };
};
```

必须命名扩展 lambda 的封闭函数，并且可以获取其地址。 如果封闭函数是类成员，则必须满足以下条件：
* 包含成员函数的所有类都必须有一个名称。
* 成员函数在其父类中不得具有私有或受保护的访问权限。
* 所有封闭类在其各自的父类中不得具有私有或受保护的访问权限。
  
例子：
```C++
void foo(void) {
  // OK
  auto lam1 = [] __device__ { return 0; };
  {
    // OK
    auto lam2 = [] __device__ { return 0; };
    // OK
    auto lam3 = [] __device__ __host__ { return 0; };
  }
}

struct S1_t {
  S1_t(void) {
    // Error: cannot take address of enclosing function
    auto lam4 = [] __device__ { return 0; }; 
  }
};

class C0_t {
  void foo(void) { 
    // Error: enclosing function has private access in parent class
    auto temp1 = [] __device__ { return 10; };
  }
  struct S2_t {
    void foo(void) {
      // Error: enclosing class S2_t has private access in its 
      // parent class
      auto temp1 = [] __device__ { return 10; };
    }
  };
};

```
必须可以在定义扩展 lambda 的位置明确地获取封闭例程的地址。 这在某些情况下可能不可行，例如 当类 `typedef` 隐藏同名的模板类型参数时。
例子：
```C++
template <typename> struct A {
  typedef void Bar;
  void test();
};

template<> struct A<void> { };

template <typename Bar>
void A<Bar>::test() {
  /* In code sent to host compiler, nvcc will inject an
     address expression here, of the form: 
     (void (A< Bar> ::*)(void))(&A::test))
 
     However, the class typedef 'Bar' (to void) shadows the
     template argument 'Bar', causing the address 
     expression in A<int>::test to actually refer to:
     (void (A< void> ::*)(void))(&A::test))
    
     ..which doesn't take the address of the enclosing
     routine 'A<int>::test' correctly.
  */
  auto lam1 = [] __host__ __device__ { return 4; };
}

int main() {
  A<int> xxx;
  xxx.test();
}

```

不能在函数本地的类中定义扩展 lambda。
例子：
```C++
void foo(void) {
  struct S1_t {
    void bar(void) {
      // Error: bar is member of a class that is local to a function.
      auto lam4 = [] __host__ __device__ { return 0; }; 
    }
  };
}

```
扩展 lambda 的封闭函数不能推导出返回类型。
例子：
```C++
auto foo(void) {
  // Error: the return type of foo is deduced.
  auto lam1 = [] __host__ __device__ { return 0; }; 
}

```

`__host__ __device__` 扩展 lambda 不能是通用 lambda。
例子：
```C++
void foo(void) {
  // Error: __host__ __device__ extended lambdas cannot be
  // generic lambdas.
  auto lam1 = [] __host__ __device__ (auto i) { return i; };

  // Error: __host__ __device__ extended lambdas cannot be
  // generic lambdas.
  auto lam2 = [] __host__ __device__ (auto ...i) {
               return sizeof...(i);
              };
}

```

如果封闭函数是函数模板或成员函数模板的实例化，或函数是类模板的成员，则模板必须满足以下约束：
* 模板最多只能有一个可变参数，并且必须在模板参数列表中最后列出。
* 模板参数必须命名。
* 模板实例化参数类型不能涉及函数本地的类型（扩展 lambda 的闭包类型除外），或者是私有或受保护的类成员。
  
例子
```C++

template <typename T>
__global__ void kern(T in) { in(); }

template <typename... T>
struct foo {};

template < template <typename...> class T, typename... P1, 
          typename... P2>
void bar1(const T<P1...>, const T<P2...>) {
  // Error: enclosing function has multiple parameter packs
  auto lam1 =  [] __device__ { return 10; };
}

template < template <typename...> class T, typename... P1, 
          typename T2>
void bar2(const T<P1...>, T2) {
  // Error: for enclosing function, the
  // parameter pack is not last in the template parameter list.
  auto lam1 =  [] __device__ { return 10; };
}

template <typename T, T>
void bar3(void) {
  // Error: for enclosing function, the second template
  // parameter is not named.
  auto lam1 =  [] __device__ { return 10; };
}

int main() {
  foo<char, int, float> f1;
  foo<char, int> f2;
  bar1(f1, f2);
  bar2(f1, 10);
  bar3<int, 10>();
}
```
```C++

template <typename T>
__global__ void kern(T in) { in(); }

template <typename T>
void bar4(void) {
  auto lam1 =  [] __device__ { return 10; };
  kern<<<1,1>>>(lam1);
}

struct C1_t { struct S1_t { }; friend int main(void); };
int main() {
  struct S1_t { };
  // Error: enclosing function for device lambda in bar4
  // is instantiated with a type local to main.
  bar4<S1_t>();

  // Error: enclosing function for device lambda in bar4
  // is instantiated with a type that is a private member
  // of a class.
  bar4<C1_t::S1_t>();
}
```

对于 Visual Studio 主机编译器，封闭函数必须具有外部链接。存在限制是因为此主机编译器不支持使用非外部链接函数的地址作为模板参数，而 CUDA 编译器转换需要它来支持扩展的 lambda。

对于 Visual Studio 主机编译器，不应在“if-constexpr”块的主体内定义扩展 lambda。

扩展的 lambda 对捕获的变量有以下限制：
* 在发送到宿主编译器的代码中，变量可以通过值传递给一系列辅助函数，然后用于直接初始化用于表示[扩展 lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_31)的闭包类型的类型的字段。
* 变量只能按值捕获。
* 如果数组维数大于 7，则无法捕获数组类型的变量。
* 对于数组类型的变量，在发送到宿主编译器的代码中，首先对闭包类型的数组字段进行默认初始化，然后将数组字段的每个元素从捕获的数组变量的相应元素中复制分配。因此，数组元素类型在宿主代码中必须是默认可构造和可复制分配的。
* 无法捕获作为可变参数包元素的函数参数。
* 捕获的变量的类型不能涉及函数本地的类型（扩展 lambda 的闭包类型除外），或者是私有或受保护的类成员。
* 对于 `__host__ __device__` 扩展 lambda，在 lambda 表达式的 `operator()` 的返回或参数类型中使用的类型不能涉及函数本地的类型（扩展 lambda 的闭包类型除外），或者是私有或受保护的类成员.
* `__host__ __device__` 扩展 lambdas 不支持初始化捕获。 `__device__` 扩展 lambda 支持初始化捕获，除非初始化捕获是数组类型或 `std::initializer_list` 类型。
* 扩展 lambda 的函数调用运算符不是 constexpr。扩展 lambda 的闭包类型不是文字类型。 constexpr 说明符不能用于扩展 lambda 的声明。
* 一个变量不能被隐式地捕获在一个词法嵌套在扩展 lambda 内的 if-constexpr 块中，除非它已经在 if-constexpr 块之外早先被隐式捕获或出现在扩展 lambda 的显式捕获列表中（参见下面的示例）。

例子

```C++
void foo(void) {
  // OK: an init-capture is allowed for an
  // extended __device__ lambda.
  auto lam1 = [x = 1] __device__ () { return x; };

  // Error: an init-capture is not allowed for
  // an extended __host__ __device__ lambda.
  auto lam2 = [x = 1] __host__ __device__ () { return x; };

  int a = 1;
  // Error: an extended __device__ lambda cannot capture
  // variables by reference.
  auto lam3 = [&a] __device__ () { return a; };

  // Error: by-reference capture is not allowed
  // for an extended __device__ lambda.
  auto lam4 = [&x = a] __device__ () { return x; };

  struct S1_t { };
  S1_t s1;
  // Error: a type local to a function cannot be used in the type
  // of a captured variable.
  auto lam6 = [s1] __device__ () { };

  // Error: an init-capture cannot be of type std::initializer_list.
  auto lam7 = [x = {11}] __device__ () { };

  std::initializer_list<int> b = {11,22,33};
  // Error: an init-capture cannot be of type std::initializer_list.
  auto lam8 = [x = b] __device__ () { }; 
 
  // Error scenario (lam9) and supported scenarios (lam10, lam11)
  // for capture within 'if-constexpr' block 
  int yyy = 4;
  auto lam9 = [=] __device__ {  
    int result = 0;
    if constexpr(false) {
      //Error: An extended __device__ lambda cannot first-capture 
      //      'yyy' in constexpr-if context
      result += yyy;
    }
    return result;
  };

  auto lam10 = [yyy] __device__ {  
    int result = 0;
    if constexpr(false) {
      //OK: 'yyy' already listed in explicit capture list for the extended lambda
      result += yyy;
    }
    return result;
  };

  auto lam11 = [=] __device__ {  
    int result = yyy;
    if constexpr(false) {
      //OK: 'yyy' already implicit captured outside the 'if-constexpr' block
      result += yyy;
    }
    return result;
  };
}
```

解析函数时，CUDA 编译器为该函数中的每个扩展 lambda 分配一个计数器值。 此计数器值用于传递给主机编译器的替代命名类型。 因此，是否在函数中定义扩展 lambda 不应取决于 `__CUDA_ARCH__` 的特定值，或 `__CUDA_ARCH__` 未定义。

例子

```C++
template <typename T>
__global__ void kernel(T in) { in(); }

__host__ __device__ void foo(void) {
  // Error: the number and relative declaration
  // order of extended lambdas depends on
  // __CUDA_ARCH__
#if defined(__CUDA_ARCH__)
  auto lam1 = [] __device__ { return 0; };
  auto lam1b = [] __host___ __device__ { return 10; };
#endif
  auto lam2 = [] __device__ { return 4; };
  kernel<<<1,1>>>(lam2);
}
```

如上所述，CUDA 编译器将主机函数中定义的 `__device__` 扩展 lambda 替换为命名空间范围中定义的占位符类型。 此占位符类型未定义与原始 lambda 声明等效的 operator() 函数。 因此，尝试确定 operator() 函数的返回类型或参数类型可能在宿主代码中无法正常工作，因为宿主编译器处理的代码在语义上与 CUDA 编译器处理的输入代码不同。 但是，可以在设备代码中内省 operator() 函数的返回类型或参数类型。 请注意，此限制不适用于 `__host__ __device__` 扩展 lambda。

例子

```C++
#include <type_traits>

void foo(void) {
  auto lam1 = [] __device__ { return 10; };

  // Error: attempt to extract the return type
  // of a __device__ lambda in host code
  std::result_of<decltype(lam1)()>::type xx1 = 1;


  auto lam2 = [] __host__ __device__  { return 10; };

  // OK : lam2 represents a __host__ __device__ extended lambda
  std::result_of<decltype(lam2)()>::type xx2 = 1;
}
```

如果由扩展 lambda 表示的仿函数对象从主机传递到设备代码（例如，作为 `__global__` 函数的参数），则 lambda 表达式主体中捕获变量的任何表达式都必须保持不变，无论 `__CUDA_ARCH__` 是否 定义宏，以及宏是否具有特定值。 出现这个限制是因为 lambda 的闭包类布局取决于编译器处理 lambda 表达式时遇到捕获的变量的顺序； 如果闭包类布局在设备和主机编译中不同，则程序可能执行不正确。

例子

```C++
__device__ int result;
   
template <typename T>
__global__ void kernel(T in) { result = in(); }
   
void foo(void) {
  int x1 = 1;
  auto lam1 = [=] __host__ __device__ { 
    // Error: "x1" is only captured when __CUDA_ARCH__ is defined.
#ifdef __CUDA_ARCH__
    return x1 + 1;
#else	
    return 10; 
#endif       
  };
  kernel<<<1,1>>>(lam1);
}
```
如前所述，CUDA 编译器将扩展的 `__device__` lambda 表达式替换为发送到主机编译器的代码中的占位符类型的实例。 此占位符类型未在主机代码中定义指向函数的转换运算符，但在设备代码中提供了转换运算符。 请注意，此限制不适用于 `__host__ __device__` 扩展 lambda。


例子

```C++
template <typename T>
__global__ void kern(T in) {
  int (*fp)(double) = in;

  // OK: conversion in device code is supported
  fp(0);
  auto lam1 = [](double) { return 1; };

  // OK: conversion in device code is supported
  fp = lam1;
  fp(0);
}

void foo(void) {
  auto lam_d = [] __device__ (double) { return 1; };
  auto lam_hd = [] __host__ __device__ (double) { return 1; };
  kern<<<1,1>>>(lam_d);
  kern<<<1,1>>>(lam_hd);
  
  // OK : conversion for __host__ __device__ lambda is supported
  // in host code
  int (*fp)(double) = lam_hd;
  
  // Error: conversion for __device__ lambda is not supported in
  // host code.
  int (*fp2)(double) = lam_d;
}
```


如前所述，CUDA 编译器将扩展的 `__device__` 或 `__host__ __device__` lambda 表达式替换为发送到主机编译器的代码中的占位符类型的实例。 此占位符类型可以定义 C++ 特殊成员函数（例如构造函数、析构函数）。 因此，在 CUDA 前端编译器与主机编译器中，一些标准 C++ 类型特征可能会为扩展 lambda 的闭包类型返回不同的结果。 以下类型特征受到影响：`std::is_trivially_copyable、std::is_trivially_constructible、std::is_trivially_copy_constructible、std::is_trivially_move_constructible、std::is_trivially_destructible`。

必须注意这些类型特征的结果不用于 `__global__` 函数模板实例化或 `__device__` / `__constant__` / `__managed__` 变量模板实例化。

例子

```C++
template <bool b>
void __global__ foo() { printf("hi"); }

template <typename T>
void dolaunch() {

// ERROR: this kernel launch may fail, because CUDA frontend compiler
// and host compiler may disagree on the result of
// std::is_trivially_copyable() trait on the closure type of the 
// extended lambda
foo<std::is_trivially_copyable<T>::value><<<1,1>>>();
cudaDeviceSynchronize();
}

int main() {
int x = 0;
auto lam1 = [=] __host__ __device__ () { return x; };
dolaunch<decltype(lam1)>();
}
```
CUDA 编译器将为 1-12 中描述的部分情况生成编译器诊断； 不会为案例 13-17 生成诊断，但主机编译器可能无法编译生成的代码。

### I.6.3. Notes on __host__ __device__ lambdas

与 `__device__` lambdas 不同，`__host__ __device__` lambdas 可以从主机代码中调用。如前所述，CUDA 编译器将主机代码中定义的扩展 lambda 表达式替换为命名占位符类型的实例。扩展的 `__host__ __device__` lambda 的占位符类型通过间接函数[调用](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fntarg_32)原始 lambda 的 operator()。

间接函数调用的存在可能会导致主机编译器对扩展的 `__host__ __device__` lambda 的优化程度低于仅隐式或显式 `__host__` 的 lambda。在后一种情况下，宿主编译器可以轻松地将 lambda 的主体内联到调用上下文中。但是在扩展 `__host__ __device__` lambda 的情况下，主机编译器会遇到间接函数调用，并且可能无法轻松内联原始 `__host__ __device__` lambda 主体。

### I.6.4. *this Capture By Value
当在非静态类成员函数中定义 lambda，并且 lambda 的主体引用类成员变量时，C++11/C++14 规则要求类的 this 指针按值捕获，而不是引用的成员变量。如果 lambda 是在主机函数中定义的扩展 `__device__` 或 `__host__ __device__` lambda，并且 lambda 在 GPU 上执行，如果 this 指针指向主机内存，则在 GPU 上访问引用的成员变量将导致运行时错误。

例子：
```C++
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t { 
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };
  
  void doit(void) {
    
    auto lam1 = [=] __device__ { 
       // reference to "xxx" causes 
       // the 'this' pointer (S1_t*) to be captured by value
       return xxx + 1; 
      
    };
    
    // Kernel launch fails at run time because 'this->xxx'
    // is not accessible from the GPU
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```

C++17 通过添加新的“*this”捕获模式解决了这个问题。 在这种模式下，编译器复制由“*this”表示的对象，而不是按值捕获指针 this。 此处更详细地描述了“*this”捕获模式：http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html。

当使用 `--extended-lambda` nvcc 标志时，CUDA 编译器支持 `__device__` 和 `__global__` 函数中定义的 lambdas 以及主机代码中定义的扩展 `__device__` lambdas 的“*this”捕获模式。

这是修改为使用“*this”捕获模式的上述示例：

```C++
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t { 
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };
  
  void doit(void) {
    
    // note the "*this" capture specification
    auto lam1 = [=, *this] __device__ { 
      
       // reference to "xxx" causes 
       // the object denoted by '*this' to be captured by
       // value, and the GPU code will access copy_of_star_this->xxx
       return xxx + 1; 
      
    };
    
    // Kernel launch succeeds
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```

主机代码中定义的未注释 lambda 或扩展的 `__host__` `__device__` lambda 不允许使用“*this”捕获模式。 支持和不支持的用法示例：
```C++
struct S1_t { 
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };
  
  void host_func(void) {
    
    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };
    
    // Error: use in an extended __host__ __device__ lambda
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };
    
    // Error: use in an unannotated lambda in host function
    auto lam3 = [=, *this]  { return xxx; };
  }
  
  __device__ void device_func(void) {
    
    // OK: use in a lambda defined in a __device__ function
    auto lam1 = [=, *this] __device__ { return xxx; };
    
    // OK: use in a lambda defined in a __device__ function
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };
    
    // OK: use in a lambda defined in a __device__ function
    auto lam3 = [=, *this]  { return xxx; };
  }
  
   __host__ __device__ void host_device_func(void) {
    
    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };
    
    // Error: use in an extended __host__ __device__ lambda
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };
    
    // Error: use in an unannotated lambda in a __host__ __device__ function
    auto lam3 = [=, *this]  { return xxx; };
  }
};
```

### I.6.5. Additional Notes

`ADL Lookup`：如前所述，CUDA 编译器将在调用宿主编译器之前将扩展的 lambda 表达式替换为占位符类型的实例。 占位符类型的一个模板参数使用包含原始 lambda 表达式的函数的地址。 对于参数类型涉及扩展 lambda 表达式的闭包类型的任何主机函数调用，这可能会导致其他命名空间参与参数相关查找 (ADL)。 这可能会导致主机编译器选择不正确的函数。
例子：

```C++
namespace N1 {
  struct S1_t { };
  template <typename T>  void foo(T);
};
 
namespace N2 {
  template <typename T> int foo(T);
 
  template <typename T>  void doit(T in) {     foo(in);  }
}
 
void bar(N1::S1_t in) {
  /* extended __device__ lambda. In the code sent to the host compiler, this 
     is replaced with the placeholder type instantiation expression
     ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::S1_t in),(&bar),1> > { }'
   
     As a result, the namespace 'N1' participates in ADL lookup of the 
     call to "foo" in the body of N2::doit, causing ambiguity.
  */
  auto lam1 = [=] __device__ { };
  N2::doit(lam1);
}
```

在上面的示例中，CUDA 编译器将扩展 lambda 替换为涉及 N1 命名空间的占位符类型。 结果，命名空间 N1 参与了对 `N2::doit` 主体中的 `foo(in)` 的 ADL 查找，并且主机编译失败，因为找到了多个重载候选 `N1::foo` 和 `N2::foo`。

## I.7. Code Samples

### I.7.1. Data Aggregation Class

```C++
class PixelRGBA {
public:
    __device__ PixelRGBA(): r_(0), g_(0), b_(0), a_(0) { }
    
    __device__ PixelRGBA(unsigned char r, unsigned char g,
                         unsigned char b, unsigned char a = 255):
                         r_(r), g_(g), b_(b), a_(a) { }
    
private:
    unsigned char r_, g_, b_, a_;
    
    friend PixelRGBA operator+(const PixelRGBA&, const PixelRGBA&);
};

__device__ 
PixelRGBA operator+(const PixelRGBA& p1, const PixelRGBA& p2)
{
    return PixelRGBA(p1.r_ + p2.r_, p1.g_ + p2.g_, 
                     p1.b_ + p2.b_, p1.a_ + p2.a_);
}

__device__ void func(void)
{
    PixelRGBA p1, p2;
    // ...      // Initialization of p1 and p2 here
    PixelRGBA p3 = p1 + p2;
}
```

### I.7.2. Derived Class
```C++
__device__ void* operator new(size_t bytes, MemoryPool& p);
__device__ void operator delete(void*, MemoryPool& p);
class Shape {
public:
    __device__ Shape(void) { }
    __device__ void putThis(PrintBuffer *p) const;
    __device__ virtual void Draw(PrintBuffer *p) const {
         p->put("Shapeless"); 
    }
    __device__ virtual ~Shape() {}
};
class Point : public Shape {
public:
    __device__ Point() : x(0), y(0) {}
    __device__ Point(int ix, int iy) : x(ix), y(iy) { }
    __device__ void PutCoord(PrintBuffer *p) const;
    __device__ void Draw(PrintBuffer *p) const;
    __device__ ~Point() {}
private:
    int x, y;
};
__device__ Shape* GetPointObj(MemoryPool& pool)
{
    Shape* shape = new(pool) Point(rand(-20,10), rand(-100,-20));
    return shape;
}
```

### I.7.3. Class Template
```C++
template <class T>
class myValues {
    T values[MAX_VALUES];
public:
    __device__ myValues(T clear) { ... }
    __device__ void setValue(int Idx, T value) { ... }
    __device__ void putToMemory(T* valueLocation) { ... }
};

template <class T>
void __global__ useValues(T* memoryBuffer) {
    myValues<T> myLocation(0);
    ...
}

__device__ void* buffer;

int main()
{
    ...
    useValues<int><<<blocks, threads>>>(buffer);
    ...
}
```
### I.7.4. Function Template
```C++
template <typename T> 
__device__ bool func(T x) 
{
   ...
   return (...);
}

template <> 
__device__ bool func<int>(T x) // Specialization
{
   return true;
}

// Explicit argument specification
bool result = func<double>(0.5);

// Implicit argument deduction
int x = 1;
bool result = func(x);
```

### I.7.5. Functor Class
```C++
class Add {
public:
    __device__  float operator() (float a, float b) const
    {
        return a + b;
    }
};

class Sub {
public:
    __device__  float operator() (float a, float b) const
    {
        return a - b;
    }
};

// Device code
template<class O> __global__ 
void VectorOperation(const float * A, const float * B, float * C,
                     unsigned int N, O op)
{
    unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (iElement < N)
        C[iElement] = op(A[iElement], B[iElement]);
}

// Host code
int main()
{
    ...
    VectorOperation<<<blocks, threads>>>(v1, v2, v3, N, Add());
    ...
}
```










