// Draws the chosen dot text with viz.js, then pans and zooms it.
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  const holder = document.getElementById("graph");
  const fallback = document.getElementById("fallback");
  const pick = document.getElementById("which");
  let viz = null;
  let svg = null;
  let box = null;
  let drag = null;

  function setBox(next) {
    box = next;
    svg.setAttribute("viewBox", box.join(" "));
  }

  function fit() {
    if (!svg) return;
    const size = svg.getBBox();
    setBox([size.x, size.y, size.width, size.height]);
  }

  function draw() {
    const dot = data[pick.value];
    if (!viz) {
      fallback.textContent = "viz.js did not load, so here is the dot text.\n\n" + dot;
      return;
    }
    fallback.textContent = "";
    holder.textContent = "";
    svg = viz.renderSVGElement(dot);
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    holder.appendChild(svg);
    fit();
  }

  holder.addEventListener("wheel", function (ev) {
    if (!svg) return;
    ev.preventDefault();
    const rect = holder.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;
    const scale = ev.deltaY > 0 ? 1.15 : 1 / 1.15;
    const w = box[2] * scale, h = box[3] * scale;
    setBox([box[0] + (box[2] - w) * fx, box[1] + (box[3] - h) * fy, w, h]);
  }, { passive: false });
  holder.addEventListener("mousedown", function (ev) {
    drag = { x: ev.clientX, y: ev.clientY, box: box.slice() };
  });
  window.addEventListener("mouseup", function () { drag = null; });
  window.addEventListener("mousemove", function (ev) {
    if (!drag) return;
    const rect = holder.getBoundingClientRect();
    const dx = (ev.clientX - drag.x) * drag.box[2] / rect.width;
    const dy = (ev.clientY - drag.y) * drag.box[3] / rect.height;
    setBox([drag.box[0] - dx, drag.box[1] - dy, drag.box[2], drag.box[3]]);
  });
  pick.addEventListener("change", draw);
  document.getElementById("fit").addEventListener("click", fit);

  if (typeof Viz === "undefined") {
    draw();
    return;
  }
  Viz.instance().then(function (instance) {
    viz = instance;
    draw();
  });
})();
