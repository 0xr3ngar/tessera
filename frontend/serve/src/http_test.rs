use crate::http::{Limits, run};
use crate::mocks::{MockGenerator, loaded};
use crate::route;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

/// A server over the mocks on a free loopback port.
fn server() -> TcpStream {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let limits = Limits {
        max_body: 64,
        ..Limits::default()
    };
    std::thread::spawn(move || run(listener, limits, move |r| route(&slot, r)));
    TcpStream::connect(address).unwrap()
}

/// Sends `request`, then reads until the server closes.
fn exchange(request: &str) -> String {
    let mut stream = server();
    stream.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    stream.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn keep_alive_stream() {
    let chat = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let request = format!(
        "GET /health HTTP/1.1\r\n\r\nPOST /v1/chat/completions HTTP/1.1\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{chat}",
        chat.len()
    );
    let out = exchange(&request);
    let (first, second) = out
        .split_once("HTTP/1.1 200 OK\r\n")
        .unwrap()
        .1
        .split_once("\r\n\r\n")
        .unwrap();
    assert!(first.contains("Content-Length: 15"));
    assert!(second.starts_with(r#"{"status":"ok"}HTTP/1.1 200 OK"#));
    assert!(second.contains("Transfer-Encoding: chunked"));
    assert!(second.ends_with("data: [DONE]\n\n\r\n0\r\n\r\n"));
}

#[test]
fn refused_requests() {
    let cases = [
        (
            "POST /tokenize HTTP/1.1\r\nContent-Length: 65\r\nExpect: 100-continue\r\n\r\n",
            "413",
        ),
        (
            "POST /tokenize HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
            "501",
        ),
        ("NOT A REQUEST\r\n\r\n", "400"),
    ];
    for (request, status) in cases {
        let out = exchange(request);
        assert!(out.starts_with(&format!("HTTP/1.1 {status} ")), "{out}");
    }
}
