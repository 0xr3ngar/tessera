use crate::Source;

#[test]
fn parses_sources() {
    let commit = "0123456789abcdef0123456789ABCDEF01234567";
    let git = Source::parse(&format!("git+https://h/r.git#{commit}")).unwrap();
    let url = "https://h/r.git".to_string();
    let commit = commit.to_ascii_lowercase();
    assert_eq!(git, Source::Git { url, commit });
    assert_eq!(Source::parse("here").unwrap(), Source::Dir("here".into()));
    for bad in ["git+https://h/r.git", "git+https://h/r#main", "https://h/r"] {
        assert!(Source::parse(bad).is_err(), "{bad}");
    }
}
