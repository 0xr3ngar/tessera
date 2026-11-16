# engine/plan

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="codegen">codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td><a href="comm">engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td><a href="jit">jit</a></td><td>Kernel bodies compiled for the host with Cranelift when a plan is made</td></tr>
<tr><td><a href="kernel-macro">kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td><a href="lower">lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td><a href="run">run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td><a href="sim">sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
</table>
