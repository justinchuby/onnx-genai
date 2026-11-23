use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn files(dir: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read vendored Llama Chat assets") {
        let path = entry.expect("read asset entry").path();
        if path.is_dir() {
            files(&path, output);
        } else {
            output.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
        .join("vendor/llama-chat/assets");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut assets = Vec::new();
    files(&root, &mut assets);
    assets.sort();
    let mut source =
        String::from("pub(super) fn asset(path: &str) -> Option<&'static [u8]> { match path {\n");
    for path in assets {
        let relative = path
            .strip_prefix(&root)
            .expect("asset under root")
            .to_str()
            .expect("UTF-8 asset path")
            .replace('\\', "/");
        source.push_str(&format!(
            "{relative:?} => Some(include_bytes!({:?})),\n",
            path.to_str().expect("UTF-8 path")
        ));
    }
    source.push_str("_ => None, } }\n");
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"))
            .join("llama_chat_assets.rs"),
        source,
    )
    .expect("write embedded asset manifest");
}
