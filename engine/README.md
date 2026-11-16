# engine

The engine crates deal with the GPU/CPU, and work entirely in preloaded raw weights
and tokens.

<!-- crate table -->
Crate | Description
--- | ---
[engine](engine) | Graphs on runners: a decode loop over a stepper, or one launch of a forward graph
[gemm](gpu/gemm) | The GEMM family: tiled Metal and CUDA bodies with a shared CPU oracle
[gpukit](gpu/gpukit) | GPU compute infrastructure: pipeline compilation, specialization, caching, buffers
[graph](ir/graph) | The model graph: typed nodes, symbolic shapes, a JSON form
[interp](ir/interp) | Runs a graph op by op on the CPU: the reference for every plan
[mem](ir/mem) | Buffers in zones, and typed views onto them
[testkit](ir/testkit) | Helpers that tests share: noise, comparisons and scratch files
[trace](ir/trace) | Builds graphs by running model descriptions
[trace-derive](ir/trace-derive) | #[derive(Weights)] for model descriptions
[models](models) | Model descriptions, traced into graphs
[codegen](plan/codegen) | Generates kernel text for every dialect, and counts what it costs
[engine-comm](plan/comm) | Moves bytes between ranks over sockets or shared memory the front end opened
[jit](plan/jit) | Kernel bodies compiled for the host with Cranelift when a plan is made
[kernel-macro](plan/kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](plan/lower) | Lowers a graph to a plan of kernel launches
[run](plan/run) | Runs lowered plans on the host, Metal, CUDA or ROCm
[sim](plan/sim) | Analytic models of GPUs: what a launch should cost
[cpu](rules/cpu) | CPU kernels for matmuls and attention: short bodies over a dot primitive
[kernels](rules/kernels) | Kernels and the lowering rules that propose them
[linear](rules/linear) | Generated matmul kernels, registered as lowering rules
[engine-settings](settings) | Knobs the front end sets once and the engine reads
