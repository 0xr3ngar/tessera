# frontend

The frontend is responsible for loading weights, mapping KV caches to real memory or storage,
speaking front-end protocols like HTTP, WebSockets or image formats, and translating inputs
like text and images to tokens for the engine to work with.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="bundle">bundle</a></td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td><a href="checkpoint">checkpoint</a></td><td>Reads weights from safetensors files</td></tr>
<tr><td><a href="cli">cli</a></td><td>The tessera command: open a recipe, then serve or fetch it</td></tr>
<tr><td><a href="comm">frontend-comm</a></td><td>Joins ranks over TCP or a shared file, then hands the handles to the engine</td></tr>
<tr><td><a href="eval">eval</a></td><td>Measures a model under a recipe: speed per batch, parity against a reference</td></tr>
<tr><td><a href="grammar">grammar</a></td><td>Grammars that limit which tokens a decode step may pick</td></tr>
<tr><td><a href="recipe">recipe</a></td><td>A model, its weights and how to plan and serve it, from a directory or git</td></tr>
<tr><td><a href="serve">serve</a></td><td>OpenAI-shaped HTTP routes over generation, chat and embedding</td></tr>
<tr><td><a href="settings">frontend-settings</a></td><td>Settings read once from the environment, passed down</td></tr>
<tr><td><a href="tasks">tasks</a></td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
</table>
