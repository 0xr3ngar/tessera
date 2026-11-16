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
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td colspan="2"><a href="engine"><b>engine/</b></a></td></tr>
<tr><td>&emsp;<a href="engine/engine">engine</a></td><td>Graphs on runners: a decode loop over a stepper, or one launch of a forward graph</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/gpukit">gpukit</a></td><td>GPU compute infrastructure: pipeline compilation, specialization, caching, buffers</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/graph">graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/interp">interp</a></td><td>Runs a graph op by op on the CPU: the reference for every plan</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/mem">mem</a></td><td>Buffers in zones, and typed views onto them</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/testkit">testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace">trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace-derive">trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td>&emsp;<a href="engine/models">models</a></td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/codegen">codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/comm">engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/jit">jit</a></td><td>Kernel bodies compiled for the host with Cranelift when a plan is made</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/kernel-macro">kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/lower">lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/run">run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/sim">sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/cpu">cpu</a></td><td>CPU kernels for matmuls and attention: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/kernels">kernels</a></td><td>Kernels and the lowering rules that propose them</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/linear">linear</a></td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="engine/settings">engine-settings</a></td><td>Knobs the front end sets once and the engine reads</td></tr>
<tr><td colspan="2"><a href="frontend"><b>frontend/</b></a></td></tr>
<tr><td>&emsp;<a href="frontend/bundle">bundle</a></td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td>&emsp;<a href="frontend/checkpoint">checkpoint</a></td><td>Reads weights from safetensors files</td></tr>
<tr><td>&emsp;<a href="frontend/comm">frontend-comm</a></td><td>Joins ranks over TCP or a shared file, then hands the handles to the engine</td></tr>
<tr><td>&emsp;<a href="frontend/eval">eval</a></td><td>Measures a model under a recipe: speed per batch, parity against a reference</td></tr>
<tr><td>&emsp;<a href="frontend/serve">serve</a></td><td>An HTTP front end: text in, streamed text out, over the engine</td></tr>
<tr><td>&emsp;<a href="frontend/settings">frontend-settings</a></td><td>Settings read once from the environment, passed down</td></tr>
<tr><td>&emsp;<a href="frontend/tasks">tasks</a></td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
<tr><td colspan="2"><a href="tools"><b>tools/</b></a></td></tr>
<tr><td>&emsp;<a href="tools/calibrate">calibrate</a></td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td>&emsp;<a href="tools/check">tessera-check</a></td><td>Runs the style checks, and prints one report or writes one PR comment</td></tr>
<tr><td>&emsp;<a href="tools/findings">findings</a></td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td>&emsp;<a href="tools/lint">tessera-lint</a></td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td>&emsp;<a href="tools/load">load</a></td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td>&emsp;<a href="tools/ranks">ranks</a></td><td>Runs one rank of a sharded Qwen3 forward pass, joined to the others over TCP</td></tr>
<tr><td>&emsp;<a href="tools/shape">shape</a></td><td>Line shapes: a line with names, numbers and strings collapsed</td></tr>
<tr><td>&emsp;<a href="tools/simcheck">tessera-simcheck</a></td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td>&emsp;<a href="tools/style">tessera-style</a></td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
