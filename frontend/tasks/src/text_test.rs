use crate::{Spool, Stops};

/// Ids 1 and 2 are two halves of one character.
fn decode(ids: &[u32]) -> String {
    let mut bytes = Vec::new();
    for id in ids {
        match id {
            1 => bytes.push(0xC3),
            2 => bytes.push(0xA9),
            _ => bytes.extend_from_slice(b"ab"),
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn whole_characters() {
    let mut spool = Spool::default();
    let pieces = [7, 1, 2, 7].map(|id| spool.push(id, decode));
    assert_eq!(pieces, ["ab", "", "\u{e9}", "ab"]);
}

/// Per piece: shown text, and if a stop cut it.
fn stopped(stops: &[&str], pieces: &[&str]) -> Vec<(String, bool)> {
    let mut held = Stops::new(stops.iter().map(|s| s.to_string()).collect());
    pieces.iter().map(|p| held.push(p)).collect()
}

#[test]
fn stops_hold() {
    let shown = stopped(&["END"], &["ab", "cE", "N", "Dx"]);
    let want = [("", false), ("ab", false), ("c", false), ("", true)];
    assert_eq!(shown, want.map(|(t, c)| (t.to_string(), c)));
    let mut held = Stops::new(vec!["END".into()]);
    assert_eq!(held.push("abEN"), ("ab".to_string(), false));
    assert_eq!(held.rest(), "EN");
}
