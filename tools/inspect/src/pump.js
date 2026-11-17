// Draws one lane per rank and walks a cursor over the launches.
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const svgNs = "http://www.w3.org/2000/svg";
  const lanes = document.getElementById("lanes");
  const info = document.getElementById("info");
  const pick = document.getElementById("lap");
  const laneHeight = 46;
  const left = 90;
  const width = 1000;
  let lap = data.laps[0];
  let order = [];
  let at = -1;
  let timer = null;
  let boxes = [];

  function make(tag, attrs, parent) {
    const node = document.createElementNS(svgNs, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function x(t) {
    const span = Math.max(lap.end - lap.start, 1e-12);
    return left + ((t - lap.start) / span) * (width - left - 10);
  }

  function draw() {
    lanes.textContent = "";
    const height = data.ranks.length * laneHeight + 30;
    const svg = make("svg", { viewBox: "0 0 " + width + " " + height, width: "100%" }, lanes);
    data.ranks.forEach(function (name, r) {
      const label = make("text", { x: 6, y: r * laneHeight + 28 }, svg);
      label.textContent = name;
    });
    order = lap.events.map(function (e, i) { return i; });
    order.sort(function (a, b) {
      const ea = lap.events[a], eb = lap.events[b];
      return ea.start - eb.start || ea.rank - eb.rank;
    });
    boxes = lap.events.map(function (e) {
      const y = e.rank * laneHeight + 10;
      const box = make("rect", {
        x: x(e.start), y: y, height: 28,
        width: Math.max(x(e.end) - x(e.start), 1.5),
        class: e.sum ? "s" : "k",
      }, svg);
      const tip = make("title", {}, box);
      tip.textContent = describe(e);
      return box;
    });
    lap.events.forEach(function (e) {
      if (!e.sum || e.rank !== 0) return;
      const mid = (x(e.start) + x(e.end)) / 2;
      make("line", {
        x1: mid, x2: mid, y1: 24, y2: (data.ranks.length - 1) * laneHeight + 24,
        class: "wire",
      }, svg);
    });
    const foot = make("text", { x: left, y: height - 6 }, svg);
    foot.textContent = lap.label + ": " + lap.rows + " rows, context " + lap.context +
      ", " + ((lap.end - lap.start) * 1e6).toFixed(2) + " µs";
    at = -1;
    info.textContent = "Press play or step.";
  }

  function describe(e) {
    const nodes = e.covers.map(function (c) { return "#" + c; }).join(" ");
    const verb = e.sum ? "sums " + (e.bytes / 1024).toFixed(1) + " KB with the other ranks"
      : "runs " + e.kernel + " (rule " + e.rule + ")";
    return data.ranks[e.rank] + " step s" + e.step + " " + verb + "\n  for nodes " + nodes +
      "\n  " + (e.start * 1e6).toFixed(2) + " to " + (e.end * 1e6).toFixed(2) + " µs";
  }

  function step() {
    if (at >= 0) boxes[order[at]].classList.remove("now");
    at += 1;
    if (at >= order.length) {
      const next = data.laps.indexOf(lap) + 1;
      if (next < data.laps.length) show(next);
      else stop();
      return;
    }
    const e = lap.events[order[at]];
    boxes[order[at]].classList.add("now");
    info.textContent = describe(e);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    document.getElementById("play").textContent = "play";
  }

  function show(i) {
    lap = data.laps[i];
    pick.value = String(i);
    draw();
  }

  data.laps.forEach(function (l, i) {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = l.label;
    pick.appendChild(option);
  });
  pick.addEventListener("change", function () { stop(); show(Number(pick.value)); });
  document.getElementById("step").addEventListener("click", function () { stop(); step(); });
  document.getElementById("restart").addEventListener("click", function () { stop(); show(0); });
  document.getElementById("play").addEventListener("click", function (ev) {
    if (timer) { stop(); return; }
    timer = setInterval(step, 120);
    ev.target.textContent = "pause";
  });
  show(0);
})();
