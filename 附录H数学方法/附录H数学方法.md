# 附录H 数学方法
参考手册列出了设备代码中支持的 C/C++ 标准库数学函数的所有函数及其描述，以及所有内部函数（仅在设备代码中支持）。

本附录在适用时提供了其中一些功能的准确性信息。它使用 ULP 进行量化。有关最后位置单元 (ULP: Unit in the Last Place, 上面是直译的,这里可以理解为最小精度单元) 定义的更多信息，请参阅 Jean-Michel Muller's paper On the definition of ulp(x), RR-5504, LIP RR-2005-09, INRIA, LIP. 2005, pp.16 at https://hal.inria.fr/inria-00070503/document


设备代码中支持的数学函数不设置全局 `errno` 变量，也不报告任何浮点异常来指示错误；因此，如果需要错误诊断机制，用户应该对函数的输入和输出实施额外的筛选。用户负责指针参数的有效性。用户不得将未初始化的参数传递给数学函数，因为这可能导致未定义的行为：函数在用户程序中内联，因此受到编译器优化的影响。

## H.1. Standard Functions
本节中的函数可用于主机和设备代码。

本节指定每个函数在设备上执行时的错误范围，以及在主机不提供函数的情况下在主机上执行时的错误范围。

错误界限是从广泛但并非详尽的测试中生成的，因此它们不是保证界限。

### Single-Precision Floating-Point Functions

加法和乘法符合 IEEE 标准，因此最大误差为 0.5 ulp。

将单精度浮点操作数舍入为整数的推荐方法是 `rintf()`，而不是 `roundf()`。 原因是 `roundf()` 映射到设备上的 4 条指令序列，而 `rintf()` 映射到单个指令。 `truncf()`、`ceilf()` 和 `floorf()` 也都映射到一条指令。



<div class="tablenoborder"><a name="standard-functions__single-precision-stdlib" shape="rect">
                                 <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="standard-functions__single-precision-stdlib" class="table" frame="border" border="1" rules="all">
                                 <caption><span class="tablecap">Table 7. Single-Precision Mathematical Standard Library Functions with
                                       Maximum ULP Error</span>. <span class="desc tabledesc">The maximum error is stated as the absolute value of the
                                       difference in ulps between a correctly rounded single-precision
                                       result and the result returned by the CUDA library function.</span></caption>
                                 <thead class="thead" align="left">
                                    <tr class="row">
                                       <th class="entry" valign="top" width="40%" id="d117e25637" rowspan="1" colspan="1">Function</th>
                                       <th class="entry" valign="top" width="60%" id="d117e25640" rowspan="1" colspan="1">Maximum ulp error</th>
                                    </tr>
                                 </thead>
                                 <tbody class="tbody">
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">x+y</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">0 (IEEE-754 round-to-nearest-even)</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">x*y</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">0 (IEEE-754 round-to-nearest-even)</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">x/y</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">0 for compute capability ≥
                                             2 when compiled with <samp class="ph codeph">-prec-div=true</samp></p>
                                          <p class="p">2 (full range), otherwise</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">1/x</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">0 for compute capability ≥
                                             2 when compiled with <samp class="ph codeph">-prec-div=true</samp></p>
                                          <p class="p">1 (full range), otherwise</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1">
                                          <p class="p"><samp class="ph codeph">rsqrtf(x)</samp></p>
                                          <p class="p"><samp class="ph codeph">1/sqrtf(x)</samp></p>
                                       </td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">2 (full range)</p>
                                          <p class="p">Applies to <samp class="ph codeph">1/sqrtf(x)</samp> only when it is
                                             converted to <samp class="ph codeph">rsqrtf(x)</samp> by the compiler.
                                          </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sqrtf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">0 when compiled with <samp class="ph codeph">-prec-sqrt=true</samp></p>
                                          <p class="p">Otherwise 1 for compute capability ≥
                                             5.2
                                          </p>
                                          <p class="p">and 3 for older architectures</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">cbrtf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rcbrtf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">hypotf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rhypotf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">norm3df(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rnorm3df(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">norm4df(x,y,z,t)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rnorm4df(x,y,z,t)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">normf(dim,arr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> An error bound can't be provided because a fast algorithm is used with accuracy loss due to round-off </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rnormf(dim,arr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1"> An error bound can't be provided because a fast algorithm is used with accuracy loss due to round-off </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">expf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">exp2f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">exp10f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">expm1f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">logf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">log2f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">log10f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">log1pf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sinf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">cosf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">tanf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sincosf(x,sptr,cptr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sinpif(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">cospif(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sincospif(x,sptr,cptr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">asinf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">acosf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">atanf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">atan2f(y,x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">sinhf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">coshf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">tanhf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">asinhf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">acoshf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">atanhf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">3 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">powf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">9 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">erff(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">erfcf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">erfinvf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">erfcinvf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">erfcxf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">normcdff(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">5 (full range)</td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">normcdfinvf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">5 (full range)</td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">lgammaf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">6 (outside interval -10.001 ... -2.264; larger
                                          inside)
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">tgammaf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">11 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">fmaf(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">frexpf(x,exp)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">ldexpf(x,exp)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">scalbnf(x,n)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">scalblnf(x,l)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">logbf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">ilogbf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">j0f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">9 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 2.2 x
                                             10<sup class="ph sup">-6</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">j1f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">9 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 2.2 x
                                             10<sup class="ph sup">-6</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">jnf(n,x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          For n = 128, the maximum absolute error is 2.2 x 10<sup class="ph sup">-6</sup></td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">y0f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">9 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 2.2 x
                                             10<sup class="ph sup">-6</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">y1f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">9 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 2.2 x
                                             10<sup class="ph sup">-6</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">ynf(n,x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">
                                          <p class="p">ceil(2 + 2.5n) for |x| &lt; n</p>
                                          <p class="p">otherwise, the maximum absolute error is 2.2 x
                                             10<sup class="ph sup">-6</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">cyl_bessel_i0f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">6 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">cyl_bessel_i1f(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">6 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">fmodf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">remainderf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">remquof(x,y,iptr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">modff(x,iptr)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">fdimf(x,y)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">truncf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">roundf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">rintf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">nearbyintf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">ceilf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">floorf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">lrintf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">lroundf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">llrintf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="40%" headers="d117e25637" rowspan="1" colspan="1"><samp class="ph codeph">llroundf(x)</samp></td>
                                       <td class="entry" valign="top" width="60%" headers="d117e25640" rowspan="1" colspan="1">0 (full range) </td>
                                    </tr>
                                 </tbody>
                              </table>
                           </div>


### Double-Precision Floating-Point Functions
将双精度浮点操作数舍入为整数的推荐方法是 `rint()`，而不是 `round()`。 原因是 `round()` 映射到设备上的 5 条指令序列，而 `rint()` 映射到单个指令。 `trunc()、ceil() 和 floor()` 也都映射到一条指令。


<table cellpadding="4" cellspacing="0" summary="" class="table" frame="border" border="1" rules="all">
                                 <caption><span class="tablecap"></span>. <span class="desc tabledesc"></span></caption>
                                 <thead class="thead" align="left">
                                    <tr class="row">
                                       <th class="entry" valign="top" width="51.690821256038646%" id="d117e26777" rowspan="1" colspan="1">Function</th>
                                       <th class="entry" valign="top" width="48.30917874396135%" id="d117e26780" rowspan="1" colspan="1">Maximum ulp error</th>
                                    </tr>
                                 </thead>
                                 <tbody class="tbody">
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">x+y</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 0 (IEEE-754 round-to-nearest-even) </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">x*y</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 0 (IEEE-754 round-to-nearest-even) </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">x/y</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 0 (IEEE-754 round-to-nearest-even) </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">1/x</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p">0 (IEEE-754 round-to-nearest-even)</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sqrt(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (IEEE-754 round-to-nearest-even) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rsqrt(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 1 (full range) </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cbrt(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rcbrt(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">hypot(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rhypot(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">norm3d(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rnorm3d(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">norm4d(x,y,z,t)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rnorm4d(x,y,z,t)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">norm(dim,arr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> An error bound can't be provided because a fast algorithm is used with accuracy loss due to round-off </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rnorm(dim,arr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> An error bound can't be provided because a fast algorithm is used with accuracy loss due to round-off </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">exp(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">exp2(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">exp10(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">expm1(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">log(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">log2(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">log10(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">log1p(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sin(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cos(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">tan(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sincos(x,sptr,cptr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sinpi(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cospi(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sincospi(x,sptr,cptr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range)</td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">asin(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">acos(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">atan(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">atan2(y,x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">sinh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cosh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">tanh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 1 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">asinh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">acosh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">atanh(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">pow(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">erf(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 2 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">erfc(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 5 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">erfinv(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 5 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">erfcinv(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 6 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">erfcx(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 4 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">normcdf(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 5 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">normcdfinv(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 8 (full range)</td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">lgamma(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 4 (outside interval -11.0001 ... -2.2637; larger
                                          inside)
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">tgamma(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 8 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">fma(x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (IEEE-754 round-to-nearest-even) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">frexp(x,exp)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">ldexp(x,exp)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">scalbn(x,n)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">scalbln(x,l)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">logb(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">ilogb(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">j0(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 7 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 5 x
                                             10<sup class="ph sup">-12</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">j1(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 7 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 5 x
                                             10<sup class="ph sup">-12</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">jn(n,x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          For n = 128, the maximum absolute error is 5 x 10<sup class="ph sup">-12</sup></td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">y0(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 7 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 5 x
                                             10<sup class="ph sup">-12</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">y1(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p"> 7 for |x| &lt; 8</p>
                                          <p class="p">otherwise, the maximum absolute error is 5 x
                                             10<sup class="ph sup">-12</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">yn(n,x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1">
                                          <p class="p">For |x| &gt; 1.5n, the maximum absolute error is 5 x
                                             10<sup class="ph sup">-12</sup></p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cyl_bessel_i0(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 6 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">cyl_bessel_i1(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 6 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">fmod(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">remainder(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">remquo(x,y,iptr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">modf(x,iptr)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">fdim(x,y)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">trunc(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">round(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">rint(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">nearbyint(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">ceil(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">floor(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">lrint(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">lround(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">llrint(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="51.690821256038646%" headers="d117e26777" rowspan="1" colspan="1"><samp class="ph codeph">llround(x)</samp></td>
                                       <td class="entry" valign="top" width="48.30917874396135%" headers="d117e26780" rowspan="1" colspan="1"> 0 (full range) </td>
                                    </tr>
                                 </tbody>
                              </table>


## H.2. Intrinsic Functions
本节中的函数只能在设备代码中使用。

在这些函数中，有一些标准函数的精度较低但速度更快的版本。它们具有相同的名称，前缀为 __（例如 __sinf(x)）。 它们更快，因为它们映射到更少的本机指令。 编译器有一个选项 (-use_fast_math)，它强制下表 中的每个函数编译为其内在对应项。 除了降低受影响函数的准确性外，还可能导致特殊情况处理的一些差异。 一种更健壮的方法是通过调用内联函数来选择性地替换数学函数调用，仅在性能增益值得考虑的情况下以及可以容忍更改的属性（例如降低的准确性和不同的特殊情况处理）的情况下。

<div class="tablenoborder"><a name="intrinsic-functions__functions-affected-use-fast-math" shape="rect">
                              <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="intrinsic-functions__functions-affected-use-fast-math" class="table" frame="border" border="1" rules="all">
                              <caption><span class="tablecap">Table 9. Functions Affected by -use_fast_math</span></caption>
                              <thead class="thead" align="left">
                                 <tr class="row">
                                    <th class="entry" valign="top" width="50%" id="d117e27876" rowspan="1" colspan="1">Operator/Function</th>
                                    <th class="entry" valign="top" width="50%" id="d117e27879" rowspan="1" colspan="1">Device Function</th>
                                 </tr>
                              </thead>
                              <tbody class="tbody">
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">x/y</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1">
                                       <p class="p"><samp class="ph codeph">__fdividef(x,y)</samp></p>
                                    </td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">sinf(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1">
                                       <p class="p"><samp class="ph codeph">__sinf(x)</samp></p>
                                    </td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">cosf(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1">
                                       <p class="p"><samp class="ph codeph">__cosf(x)</samp></p>
                                    </td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">tanf(x) </samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__tanf(x)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">sincosf(x,sptr,cptr)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__sincosf(x,sptr,cptr)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">logf(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1">
                                       <p class="p"><samp class="ph codeph">__logf(x)</samp></p>
                                    </td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">log2f(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__log2f(x)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">log10f(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__log10f(x)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">expf(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__expf(x)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">exp10f(x)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__exp10f(x)</samp></td>
                                 </tr>
                                 <tr class="row">
                                    <td class="entry" valign="top" width="50%" headers="d117e27876" rowspan="1" colspan="1"><samp class="ph codeph">powf(x,y)</samp></td>
                                    <td class="entry" valign="top" width="50%" headers="d117e27879" rowspan="1" colspan="1"><samp class="ph codeph">__powf(x,y)</samp></td>
                                 </tr>
                              </tbody>
                           </table>
                        </div>


### Single-Precision Floating-Point Functions
`__fadd_[rn,rz,ru,rd]()` 和 `__fmul_[rn,rz,ru,rd]()` 映射到编译器从不合并到 `FMAD` 中的加法和乘法运算。相比之下，由“*”和“+”运算符生成的加法和乘法将经常组合到 FMAD 中。

以 `_rn` 为后缀的函数使用舍入到最接近的偶数舍入模式运行。

以 `_rz` 为后缀的函数使用向零舍入模式进行舍入操作。

以 `_ru` 为后缀的函数使用向上舍入（到正无穷大）舍入模式运行。

以 `_rd` 为后缀的函数使用向下舍入（到负无穷大）舍入模式进行操作。

浮点除法的准确性取决于代码是使用 `-prec-div=false` 还是 `-prec-div=true` 编译的。使用`-prec-div=false`编译代码时，正则除法/运算符和`__fdividef(x,y)`精度相同，但对于2<sup class="ph sup">126</sup> &lt; <samp class="ph codeph">|y|</samp> &lt;2<sup class="ph sup">128</sup>，`__fdividef(x,y)` 提供的结果为零，而 / 运算符提供的正确结果在下表 中规定的精度范围内。此外，对于 2<sup class="ph sup">126</sup> &lt; <samp class="ph codeph">|y|</samp> &lt;2<sup class="ph sup">128</sup>，如果 x 为无穷大，则 `__fdividef(x,y) `提供 NaN（作为无穷大乘以零的结果），而 / 运算符返回无穷大。另一方面，当使用 `-prec-div=true` 或根本没有任何 `-prec-div` 选项编译代码时， / 运算符符合 IEEE 标准，因为它的默认值为 true。

<div class="tablenoborder"><a name="intrinsic-functions__single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library" shape="rect">
                                 <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="intrinsic-functions__single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library" class="table" frame="border" border="1" rules="all">
                                 <caption><span class="tablecap"></span></caption>
                                 <thead class="thead" align="left">
                                    <tr class="row">
                                       <th class="entry" valign="top" width="50%" id="d117e28195" rowspan="1" colspan="1">Function</th>
                                       <th class="entry" valign="top" width="50%" id="d117e28198" rowspan="1" colspan="1">Error bounds</th>
                                    </tr>
                                 </thead>
                                 <tbody class="tbody">
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fadd_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">
                                          <p class="p">  IEEE-compliant. </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fsub_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">
                                          <p class="p">  IEEE-compliant. </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fmul_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">
                                          <p class="p">  IEEE-compliant. </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fmaf_[rn,rz,ru,rd](x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">
                                          <p class="p">  IEEE-compliant. </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__frcp_[rn,rz,ru,rd](x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">  IEEE-compliant. </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fsqrt_[rn,rz,ru,rd](x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">  IEEE-compliant. </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__frsqrt_rn(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1"> IEEE-compliant. </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fdiv_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">
                                          <p class="p"> IEEE-compliant. </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__fdividef(x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">|y|</samp> in [2<sup class="ph sup">-126</sup>,
                                          2<sup class="ph sup">126</sup>], the maximum ulp error is 2.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__expf(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">The maximum ulp error is <samp class="ph codeph">2 + floor(abs(1.16 *
                                             x))</samp>.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__exp10f(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">The maximum ulp error is <samp class="ph codeph">2+ floor(abs(2.95 *
                                             x))</samp>.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__logf(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">x</samp> in [0.5, 2], the maximum absolute
                                          error is 2<sup class="ph sup">-21.41</sup>, otherwise, the maximum ulp error
                                          is 3.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__log2f(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">x</samp> in [0.5, 2], the maximum absolute
                                          error is 2<sup class="ph sup">-22</sup>, otherwise, the maximum ulp error is
                                          2.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__log10f(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">x</samp> in [0.5, 2], the maximum absolute
                                          error is 2<sup class="ph sup">-24</sup>, otherwise, the maximum ulp error is
                                          3.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__sinf(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">x</samp> in [-π,π], the maximum absolute
                                          error is 2<sup class="ph sup">-21.41</sup>, and larger otherwise.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__cosf(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">For <samp class="ph codeph">x</samp> in [-π,π], the maximum absolute
                                          error is 2<sup class="ph sup">-21.19</sup>, and larger otherwise.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__sincosf(x,sptr,cptr)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">Same as <samp class="ph codeph">__sinf(x)</samp> and
                                          <samp class="ph codeph">__cosf(x)</samp>.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__tanf(x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">Derived from its implementation as <samp class="ph codeph">__sinf(x) *
                                             (1/__cosf(x))</samp>.
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28195" rowspan="1" colspan="1"><samp class="ph codeph">__powf(x, y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28198" rowspan="1" colspan="1">Derived from its implementation as <samp class="ph codeph">exp2f(y *
                                             __log2f(x))</samp>.
                                       </td>
                                    </tr>
                                 </tbody>
                              </table>
                           </div>


### Double-Precision Floating-Point Functions

`__dadd_rn()` 和 `__dmul_rn()` 映射到编译器从不合并到 FMAD 中的加法和乘法运算。 相比之下，由“*”和“+”运算符生成的加法和乘法将经常组合到 FMAD 中。

<table cellpadding="4" cellspacing="0" summary="" class="table" frame="border" border="1" rules="all">
                                 <caption><span class="tablecap">Table 11. Double-Precision Floating-Point Intrinsic Functions</span>. <span class="desc tabledesc">(Supported by the CUDA Runtime Library with Respective Error Bounds)</span></caption>
                                 <thead class="thead" align="left">
                                    <tr class="row">
                                       <th class="entry" valign="top" width="50%" id="d117e28543" rowspan="1" colspan="1">Function</th>
                                       <th class="entry" valign="top" width="50%" id="d117e28546" rowspan="1" colspan="1">Error bounds</th>
                                    </tr>
                                 </thead>
                                 <tbody class="tbody">
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__dadd_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p">IEEE-compliant.</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__dsub_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p">IEEE-compliant.</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__dmul_[rn,rz,ru,rd](x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p">IEEE-compliant.</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__fma_[rn,rz,ru,rd](x,y,z)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p">IEEE-compliant.</p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__ddiv_[rn,rz,ru,rd](x,y)(x,y)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p">IEEE-compliant.</p>
                                          <p class="p">
                                             Requires compute capability <u class="ph u">&gt;</u> 2.
                                          </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__drcp_[rn,rz,ru,rd](x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p"> IEEE-compliant.</p>
                                          <p class="p">
                                             Requires compute capability <u class="ph u">&gt;</u> 2.
                                          </p>
                                       </td>
                                    </tr>
                                    <tr class="row">
                                       <td class="entry" valign="top" width="50%" headers="d117e28543" rowspan="1" colspan="1"><samp class="ph codeph">__dsqrt_[rn,rz,ru,rd](x)</samp></td>
                                       <td class="entry" valign="top" width="50%" headers="d117e28546" rowspan="1" colspan="1">
                                          <p class="p"> IEEE-compliant.</p>
                                          <p class="p">
                                             Requires compute capability <u class="ph u">&gt;</u> 2.  
                                          </p>
                                       </td>
                                    </tr>
                                 </tbody>
                              </table>

