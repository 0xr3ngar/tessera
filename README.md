# tessera

tessera is an engine to build engines.

In tessera, you define the shape of your model in code, and the engine makes an optimized engine for your
hardware, automatically.

## State

tessera is not alpha or beta. It's an experiment right now and no numbers are available.

## Supported Hardware

 - CUDA (NVIDIA)
 - ROCm (AMD)
 - Metal (Apple)
 - CPU (any)

## Supported Models

 - Qwen 0.6B
 - Embedding Gemma 2
 - spaCy

## Crates

<!-- crate table -->
Crate | Description
--- | ---
[engine](engine/engine) | Graphs on runners: a decode loop over a stepper, or one launch of a forward graph
[gemm](engine/gpu/gemm) | The GEMM family: tiled Metal and CUDA bodies with a shared CPU oracle
[gpukit](engine/gpu/gpukit) | GPU compute infrastructure: pipeline compilation, specialization, caching, buffers
[graph](engine/ir/graph) | The model graph: typed nodes, symbolic shapes, a JSON form
[interp](engine/ir/interp) | Runs a graph op by op on the CPU: the reference for every plan
[mem](engine/ir/mem) | Buffers in zones, and typed views onto them
[testkit](engine/ir/testkit) | Helpers that tests share: noise, comparisons and scratch files
[trace](engine/ir/trace) | Builds graphs by running model descriptions
[trace-derive](engine/ir/trace-derive) | #[derive(Weights)] for model descriptions
[models](engine/models) | Model descriptions, traced into graphs
[codegen](engine/plan/codegen) | Generates kernel text for every dialect, and counts what it costs
[engine-comm](engine/plan/comm) | Moves bytes between ranks over sockets or shared memory the front end opened
[jit](engine/plan/jit) | Kernel bodies compiled for the host with Cranelift when a plan is made
[kernel-macro](engine/plan/kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](engine/plan/lower) | Lowers a graph to a plan of kernel launches
[run](engine/plan/run) | Runs lowered plans on the host, Metal, CUDA or ROCm
[sim](engine/plan/sim) | Analytic models of GPUs: what a launch should cost
[cpu](engine/rules/cpu) | CPU kernels for matmuls and attention: short bodies over a dot primitive
[kernels](engine/rules/kernels) | Kernels and the lowering rules that propose them
[linear](engine/rules/linear) | Generated matmul kernels, registered as lowering rules
[engine-settings](engine/settings) | Knobs the front end sets once and the engine reads
[bundle](frontend/bundle) | A Hugging Face model directory as graph, weights, tokenizer and template
[checkpoint](frontend/checkpoint) | Reads weights from safetensors files
[frontend-comm](frontend/comm) | Joins ranks over TCP or a shared file, then hands the handles to the engine
[eval](frontend/eval) | Measures a model under a recipe: speed per batch, parity against a reference
[serve](frontend/serve) | An HTTP front end: text in, streamed text out, over the engine
[frontend-settings](frontend/settings) | Settings read once from the environment, passed down
[tasks](frontend/tasks) | Text in, answers out: generation and tagging over a loaded model
[calibrate](tools/calibrate) | Times kernels on this machine and stores what they cost
[tessera-check](tools/check) | Runs the style checks, and prints one report or writes one PR comment
[findings](tools/findings) | The JSON each check writes, for tessera-check to render
[tessera-lint](tools/lint) | The repository's code-shape rules, run by CI and the pre-commit hook
[load](tools/load) | Drives concurrent requests against tessera-serve and reports throughput
[ranks](tools/ranks) | Runs one rank of a sharded Qwen3 forward pass, joined to the others over TCP
[shape](tools/shape) | Line shapes: a line with names, numbers and strings collapsed
[tessera-simcheck](tools/simcheck) | Fails on near-copies of code, found with EmbeddingGemma 2 embeddings
[tessera-style](tools/style) | Comment and prose style rules, as part-of-speech patterns
