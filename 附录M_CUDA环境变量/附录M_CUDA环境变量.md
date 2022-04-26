# 附录M_CUDA环境变量

下表列出了 CUDA 环境变量。 与多进程服务相关的环境变量记录在 GPU 部署和管理指南的多进程服务部分。
<div class="tablenoborder"><a name="env-vars__cuda-environment-variables" shape="rect">
                           <!-- --></a><table cellpadding="4" cellspacing="0" summary="" id="env-vars__cuda-environment-variables" class="table" frame="border" border="1" rules="all">
                           <caption><span class="tablecap">Table 18. CUDA Environment Variables</span></caption>
                           <thead class="thead" align="left">
                              <tr class="row">
                                 <th class="entry" valign="top" width="30%" id="d117e36704" rowspan="1" colspan="1">Variable</th>
                                 <th class="entry" valign="top" width="20%" id="d117e36707" rowspan="1" colspan="1">Values</th>
                                 <th class="entry" valign="top" width="50%" id="d117e36710" rowspan="1" colspan="1">Description</th>
                              </tr>
                           </thead>
                           <tbody class="tbody">
                              <tr class="row">
                                 <td class="entry" colspan="3" valign="top" headers="d117e36704 d117e36707 d117e36710" rowspan="1"><strong class="ph b">Device Enumeration and Properties</strong></td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_VISIBLE_DEVICES </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">A comma-separated sequence of GPU identifiers<br clear="none"></br> MIG support:
                                    <samp class="ph codeph">MIG-&lt;GPU-UUID&gt;/&lt;GPU instance ID&gt;/&lt;compute instance ID&gt;</samp></td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">GPU identifiers are given as integer indices or as UUID strings. GPU UUID
                                    strings should follow the same format as given by <dfn class="term">nvidia-smi</dfn>, such as
                                    GPU-8932f937-d72c-4106-c12f-20bd9faed9f6. However, for convenience, abbreviated
                                    forms are allowed; simply specify enough digits from the beginning of the GPU UUID
                                    to uniquely identify that GPU in the target system. For example,
                                    CUDA_VISIBLE_DEVICES=GPU-8932f937 may be a valid way to refer to the above GPU UUID,
                                    assuming no other GPU in the system shares this prefix.<br clear="none"></br> Only the devices whose
                                    index is present in the sequence are visible to CUDA applications and they are
                                    enumerated in the order of the sequence. If one of the indices is invalid, only the
                                    devices whose index precedes the invalid index are visible to CUDA applications. For
                                    example, setting CUDA_VISIBLE_DEVICES to 2,1 causes device 0 to be invisible and
                                    device 2 to be enumerated before device 1. Setting CUDA_VISIBLE_DEVICES to 0,2,-1,1
                                    causes devices 0 and 2 to be visible and device 1 to be invisible.<br clear="none"></br> MIG format
                                    starts with MIG keyword and GPU UUID should follow the same format as given by
                                    <dfn class="term">nvidia-smi</dfn>. For example,
                                    MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2. Only single MIG instance
                                    enumeration is supported.
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_MANAGED_FORCE_DEVICE_ALLOC </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0) </td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Forces the driver to place all managed allocations in device memory.</td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_DEVICE_ORDER </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">FASTEST_FIRST, PCI_BUS_ID, (default is FASTEST_FIRST) </td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">FASTEST_FIRST causes CUDA to enumerate the available devices in fastest to
                                    slowest order using a simple heuristic. PCI_BUS_ID orders devices by PCI bus ID in
                                    ascending order.
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" colspan="3" valign="top" headers="d117e36704 d117e36707 d117e36710" rowspan="1"><strong class="ph b">Compilation</strong></td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_CACHE_DISABLE </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Disables caching (when set to 1) or enables caching (when set to 0) for
                                    just-in-time-compilation. When disabled, no binary code is added to or retrieved
                                    from the cache.
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_CACHE_PATH </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">filepath </td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Specifies the folder where the just-in-time compiler caches binary codes; the
                                    default values are: 
                                    <ul class="ul">
                                       <li class="li">on Windows, <samp class="ph codeph">%APPDATA%\NVIDIA\ComputeCache</samp></li>
                                       <li class="li">on Linux, <samp class="ph codeph">~/.nv/ComputeCache</samp></li>
                                    </ul>
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_CACHE_MAXSIZE</td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">integer (default is 268435456 (256 MiB) and maximum is 4294967296 (4
                                    GiB))
                                 </td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Specifies the size in bytes of the cache used by the just-in-time compiler.
                                    Binary codes whose size exceeds the cache size are not cached. Older binary codes
                                    are evicted from the cache to make room for newer binary codes if needed. 
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_FORCE_PTX_JIT </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">When set to 1, forces the device driver to ignore any binary code embedded in
                                    an application (see <a class="xref" href="index.html#application-compatibility" shape="rect">Application Compatibility</a>) and to just-in-time
                                    compile embedded <dfn class="term">PTX</dfn> code instead. If a kernel does not have embedded
                                    <dfn class="term">PTX</dfn> code, it will fail to load. This environment variable can be used
                                    to validate that <dfn class="term">PTX</dfn> code is embedded in an application and that its
                                    just-in-time compilation works as expected to guarantee application forward
                                    compatibility with future architectures (see <a class="xref" href="index.html#just-in-time-compilation" shape="rect">Just-in-Time Compilation</a>).
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_DISABLE_PTX_JIT </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">When set to 1, disables the just-in-time compilation of embedded
                                    <dfn class="term">PTX</dfn> code and use the compatible binary code embedded in an
                                    application (see <a class="xref" href="index.html#application-compatibility" shape="rect">Application Compatibility</a>). If a kernel does not have embedded binary code or the embedded binary was
                                    compiled for an incompatible architecture, then it will fail to load. This
                                    environment variable can be used to validate that an application has the compatible
                                    <dfn class="term">SASS</dfn> code generated for each kernel.(see <a class="xref" href="index.html#binary-compatibility" shape="rect">Binary Compatibility</a>).
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" colspan="3" valign="top" headers="d117e36704 d117e36707 d117e36710" rowspan="1"><strong class="ph b">Execution</strong></td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_LAUNCH_BLOCKING </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Disables (when set to 1) or enables (when set to 0) asynchronous kernel
                                    launches. 
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_DEVICE_MAX_CONNECTIONS </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">1 to 32 (default is 8)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Sets the number of compute and copy engine concurrent connections (work queues)
                                    from the host to each device of compute capability 3.5 and above. 
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_AUTO_BOOST </td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Overrides the autoboost behavior set by the --auto-boost-default option of
                                    nvidia-smi. If an application requests via this environment variable a behavior that
                                    is different from nvidia-smi's, its request is honored if there is no other
                                    application currently running on the same GPU that successfully requested a
                                    different behavior, otherwise it is ignored. 
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" colspan="3" valign="top" headers="d117e36704 d117e36707 d117e36710" rowspan="1"><strong class="ph b">cuda-gdb (on Linux platform)</strong></td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_DEVICE_WAITS_ON_EXCEPTION</td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">0 or 1 (default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">When set to 1, a CUDA application will halt when a device exception occurs,
                                    allowing a debugger to be attached for further debugging.
                                 </td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" colspan="3" valign="top" headers="d117e36704 d117e36707 d117e36710" rowspan="1"><strong class="ph b">MPS service (on Linux platform)</strong></td>
                              </tr>
                              <tr class="row">
                                 <td class="entry" valign="top" width="30%" headers="d117e36704" rowspan="1" colspan="1">CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT</td>
                                 <td class="entry" valign="top" width="20%" headers="d117e36707" rowspan="1" colspan="1">Percentage value (between 0 - 100, default is 0)</td>
                                 <td class="entry" valign="top" width="50%" headers="d117e36710" rowspan="1" colspan="1">Devices of compute capability 8.x allow, a portion of L2 cache to be set-aside
                                    for persisting data accesses to global memory. When using CUDA MPS service, the
                                    set-aside size can only be controlled using this environment variable, before
                                    starting CUDA MPS control daemon. I.e., the environment variable should be set
                                    before running the command <samp class="ph codeph">nvidia-cuda-mps-control -d</samp>.
                                 </td>
                              </tr>
                           </tbody>
                        </table>
                     </div>
