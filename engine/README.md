# engine

The engine crates deal with the GPU/CPU, and work entirely in preloaded raw weights
and tokens.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="engine">engine</a></td><td>Graphs on runners: a decode loop over a stepper, or one launch of a forward graph</td></tr>
<tr><td colspan="2"><a href="gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;<a href="gpu/gpukit">gpukit</a></td><td>The runtimes tessera loads at run time, GPUs and RDMA: pipelines, buffers, verbs</td></tr>
<tr><td>&emsp;<a href="gpu/platform">platform</a></td><td>Plain hardware description: platforms, limits and the per-architecture table</td></tr>
<tr><td colspan="2"><a href="ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;<a href="ir/graph">graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;<a href="ir/interp">interp</a></td><td>Runs a graph op by op on the CPU: the reference for every plan</td></tr>
<tr><td>&emsp;<a href="ir/mem">mem</a></td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td>&emsp;<a href="ir/testkit">testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;<a href="ir/trace">trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;<a href="ir/trace-derive">trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td><a href="models">models</a></td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="2"><a href="plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;<a href="plan/codegen">codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;<a href="plan/comm">engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;<a href="plan/host">host</a></td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td>&emsp;<a href="plan/jit">jit</a></td><td>Kernel bodies compiled for the host with Cranelift when a plan is made</td></tr>
<tr><td>&emsp;<a href="plan/kernel-macro">kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;<a href="plan/lower">lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;<a href="plan/run">run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;<a href="plan/sim">sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="2"><a href="rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;<a href="rules/cpu">cpu</a></td><td>CPU kernels for matmuls and attention: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;<a href="rules/kernels">kernels</a></td><td>Kernels and the lowering rules that propose them</td></tr>
<tr><td>&emsp;<a href="rules/linear">linear</a></td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="rules/rules">rules</a></td><td>Links every lowering rule crate</td></tr>
<tr><td><a href="settings">engine-settings</a></td><td>Knobs the front end sets once and the engine reads</td></tr>
</table>
