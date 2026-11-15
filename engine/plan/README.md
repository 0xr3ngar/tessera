# engine/plan

<!-- crate table -->
Crate | Description
--- | ---
[codegen](codegen) | Generates kernel text for every dialect, and counts what it costs
[engine-comm](comm) | Moves bytes between ranks over sockets or shared memory the front end opened
[kernel-macro](kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](lower) | Lowers a graph to a plan of kernel launches
[run](run) | Runs lowered plans on the host, Metal, CUDA or ROCm
[sim](sim) | Analytic models of GPUs: what a launch should cost
