# tools

Build tools that run as part of CI or pre-commit hooks.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="calibrate">calibrate</a></td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td><a href="check">tessera-check</a></td><td>Runs the style checks, and prints one report or writes one PR comment</td></tr>
<tr><td><a href="findings">findings</a></td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td><a href="inspect">inspect</a></td><td>Draws a model's graph, its lowering per rank, and tokens moving through the ranks</td></tr>
<tr><td><a href="lint">tessera-lint</a></td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td><a href="load">load</a></td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td><a href="parity">parity</a></td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td><a href="ranks">ranks</a></td><td>Runs one rank of a sharded Qwen3 forward pass, joined to the others over TCP</td></tr>
<tr><td><a href="shape">shape</a></td><td>Line shapes: a line with names, numbers and strings collapsed</td></tr>
<tr><td><a href="simcheck">tessera-simcheck</a></td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td><a href="style">tessera-style</a></td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
