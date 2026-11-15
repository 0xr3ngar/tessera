# engine/plan

<!-- crate table -->
Crate | Description
--- | ---
[codegen](codegen) | Generates kernel text for every dialect, and counts what it costs
[comm](comm) | Moves bytes between ranks: the transport trait and TCP sockets
[kernel-macro](kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](lower) | Lowers a graph to a plan of kernel launches
[run](run) | Runs lowered plans on the host, Metal, CUDA or ROCm
[sim](sim) | Analytic models of GPUs: what a launch should cost
