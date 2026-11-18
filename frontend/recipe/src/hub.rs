//! Model files from Hugging Face, kept by repository and commit.

use crate::{Error, io, pinned};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// File endings a model directory needs. Other files stay remote.
const KEEP: [&str; 5] = [".json", ".safetensors", ".jinja", ".txt", ".model"];

#[derive(Deserialize)]
struct Info {
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// `owner/name@commit`, downloaded into `cache` once.
pub(crate) fn snapshot(spec: &str, cache: &Path) -> Result<PathBuf, Error> {
    let refuse = || Error::Source(spec.into(), "hf weights are owner/name@<commit>");
    let (repo, commit) = spec.split_once('@').ok_or_else(refuse)?;
    if !pinned(commit) {
        return Err(refuse());
    }
    let dir = cache.join("hf").join(repo).join(commit);
    let done = dir.join(".complete");
    if done.is_file() {
        return Ok(dir);
    }
    let api = format!("https://huggingface.co/api/models/{repo}/revision/{commit}");
    let response = ureq::get(&api).call().map_err(http(&api))?;
    let info: Info = response.into_json().map_err(io(&dir))?;
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    for s in info.siblings {
        let name = s.rfilename;
        let wanted = !name.contains('/') && KEEP.iter().any(|k| name.ends_with(k));
        if wanted && !dir.join(&name).is_file() {
            let url = format!("https://huggingface.co/{repo}/resolve/{commit}/{name}");
            download(&url, &dir.join(&name))?;
        }
    }
    std::fs::write(&done, b"").map_err(io(&done))?;
    Ok(dir)
}

/// Streams `url` to `path` through a `.part` file.
fn download(url: &str, path: &Path) -> Result<(), Error> {
    eprintln!("fetching {url}");
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let response = ureq::get(url).call().map_err(http(url))?;
    let mut file = std::fs::File::create(&part).map_err(io(&part))?;
    std::io::copy(&mut response.into_reader(), &mut file).map_err(io(&part))?;
    std::fs::rename(&part, path).map_err(io(path))
}

fn http(url: &str) -> impl FnOnce(ureq::Error) -> Error + '_ {
    move |e| Error::Http(url.into(), e.to_string())
}
