//! The graph page: dot text drawn by viz.js.

use crate::html::{Part, el, json, text};
use crate::lowering::head;
use serde::Serialize;

const SCRIPT: &str = include_str!("diagram.js");
/// Graphviz built for the browser, pinned.
const VIZ: &str = "https://cdn.jsdelivr.net/npm/@viz-js/viz@3.11.0/lib/viz-standalone.js";

#[derive(Serialize)]
struct Data<'a> {
    source: &'a str,
    sharded: &'a str,
}

/// Both graphs on one page, with pan and zoom.
pub fn page(model: &str, source: &str, sharded: &str) -> String {
    let title = format!("{model}: graph");
    let pick = el("select")
        .attr("id", "which")
        .kid(text("option", "sharded graph").attr("value", "sharded"))
        .kid(text("option", "source graph").attr("value", "source"));
    let controls = el("p")
        .kid(pick)
        .kid(text("button", "fit").attr("id", "fit"))
        .kid(text("a", "lowering").attr("href", "lowering.html"))
        .text(" · ")
        .kid(text("a", "tokens moving").attr("href", "pump.html"));
    let body = el("body")
        .kid(text("h1", title.clone()))
        .kid(controls)
        .kid(
            text(
                "p",
                "Drag to pan. Scroll to zoom. Click a node for its lowering.",
            )
            .attr("class", "hint"),
        )
        .kid(el("div").attr("id", "graph"))
        .kid(el("pre").attr("id", "fallback"))
        .kid(
            el("script")
                .attr("id", "data")
                .attr("type", "application/json")
                .kid(json(&Data { source, sharded })),
        )
        .kid(el("script").attr("src", VIZ))
        .kid(el("script").kid(Part::Raw(SCRIPT.into())));
    el("html").kid(head(&title)).kid(body).page()
}
