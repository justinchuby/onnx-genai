use std::path::{Component, Path, PathBuf};

/// Join an external-data path to its base after rejecting lexical escapes.
pub(crate) fn guarded_join(base: &Path, rel: &str) -> Result<PathBuf, &'static str> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err("absolute paths are not allowed");
    }
    for component in rel_path.components() {
        match component {
            Component::ParentDir => {
                return Err("parent-directory (`..`) traversal is not allowed");
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("absolute / rooted paths are not allowed");
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(base.join(rel_path))
}

#[cfg(test)]
mod tests {
    use super::guarded_join;
    use std::path::Path;

    #[test]
    fn rejects_parent_traversal() {
        assert_eq!(
            guarded_join(Path::new("model"), "../escape.bin"),
            Err("parent-directory (`..`) traversal is not allowed")
        );
        assert_eq!(
            guarded_join(Path::new("model"), "nested/../../escape.bin"),
            Err("parent-directory (`..`) traversal is not allowed")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_absolute_path() {
        assert_eq!(
            guarded_join(Path::new("model"), "/escape.bin"),
            Err("absolute paths are not allowed")
        );
    }

    #[cfg(windows)]
    #[test]
    fn rejects_rooted_path() {
        assert_eq!(
            guarded_join(Path::new("model"), r"\escape.bin"),
            Err("absolute / rooted paths are not allowed")
        );
    }

    #[test]
    fn allows_current_and_nested_paths() {
        assert_eq!(
            guarded_join(Path::new("model"), "./nested/weights.bin"),
            Ok(Path::new("model").join("./nested/weights.bin"))
        );
    }
}
