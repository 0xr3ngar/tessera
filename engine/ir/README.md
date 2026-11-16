# engine/ir

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="graph">graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td><a href="interp">interp</a></td><td>Runs a graph op by op on the CPU: the reference for every plan</td></tr>
<tr><td><a href="mem">mem</a></td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td><a href="testkit">testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td><a href="trace">trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td><a href="trace-derive">trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
</table>
