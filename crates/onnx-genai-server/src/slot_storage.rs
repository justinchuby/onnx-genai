//! Operator-owned capability directory for bounded, atomic slot state files.
// Reference-only builds retain configuration validation; persistence consumers require Native.
#![cfg_attr(not(feature = "native-backend"), allow(dead_code))]
use anyhow::Context;
use cap_std::fs::{Dir, OpenOptions};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

const LOCK: &str = ".nxrt-slots.lock";
const TEMP_PREFIX: &str = ".nxrt-slot-";

#[derive(Debug, Clone)]
pub(crate) struct SlotFilename(String);
impl SlotFilename {
    pub(crate) fn new(value: String) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !value.trim().is_empty()
                && value.len() <= 255
                && !value.starts_with(".nxrt-")
                && !value.chars().any(char::is_control)
                && !value.contains(['/', '\\', ':'])
                && matches!(
                    Path::new(&value).components().next(),
                    Some(std::path::Component::Normal(_))
                ),
            "invalid slot filename {value:?}; use one plain filename of at most 255 bytes without separators, control characters or the reserved .nxrt- prefix"
        );
        Ok(Self(value))
    }
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotStorageLimits {
    pub(crate) file_bytes: u64,
    pub(crate) total_bytes: u64,
    pub(crate) files: usize,
}
impl Default for SlotStorageLimits {
    fn default() -> Self {
        Self {
            file_bytes: 4 * 1024 * 1024 * 1024,
            total_bytes: 16 * 1024 * 1024 * 1024,
            files: 128,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "slot storage quota exhausted: {reason}; remove obsolete files or raise the configured slot storage bound"
)]
pub(crate) struct SlotStorageQuota {
    reason: String,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid slot state file: {0}; select a regular saved file within the configured bounds")]
pub(crate) struct SlotStorageInput(pub(crate) String);

#[derive(Debug)]
pub(crate) struct SlotStorage {
    root: Dir,
    configured_path: PathBuf,
    _process_lock: std::fs::File,
    gate: Mutex<()>,
    pub(crate) limits: SlotStorageLimits,
}
impl SlotStorage {
    pub(crate) fn open(path: &Path, limits: SlotStorageLimits) -> anyhow::Result<Self> {
        anyhow::ensure!(
            limits.file_bytes > 88 && limits.total_bytes >= limits.file_bytes && limits.files > 0,
            "invalid slot storage limits {limits:?}; require file bytes > 88, total bytes >= file bytes and a positive file count"
        );
        Dir::create_ambient_dir_all(path, cap_std::ambient_authority()).with_context(|| {
            format!(
                "create --slot-save-path '{}'; choose a writable directory",
                path.display()
            )
        })?;
        let root =
            Dir::open_ambient_dir(path, cap_std::ambient_authority()).with_context(|| {
                format!(
                    "open --slot-save-path '{}' as a capability directory; check permissions",
                    path.display()
                )
            })?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let lock = root
            .open_with(LOCK, &options)
            .context("open slot storage process lock; use a writable operator-owned directory")?
            .into_std();
        anyhow::ensure!(
            lock.metadata()?.is_file() && lock.metadata()?.len() == 0,
            "invalid slot storage lock file; use an empty regular .nxrt-slots.lock file in the operator-owned directory"
        );
        lock.try_lock().context("lock slot storage directory; another server may own it, choose a separate directory or stop that server")?;
        let result = Self {
            root,
            configured_path: path.into(),
            _process_lock: lock,
            gate: Mutex::new(()),
            limits,
        };
        result.inventory()?;
        Ok(result)
    }

    fn inventory(&self) -> anyhow::Result<(u64, usize)> {
        let mut bytes = 0_u64;
        let mut count = 0_usize;
        for entry in self
            .root
            .entries()
            .context("inspect slot storage directory; check directory read permission")?
        {
            let entry =
                entry.context("read slot storage directory entry; check filesystem health")?;
            if entry.file_name() == LOCK {
                continue;
            }
            count = count
                .checked_add(1)
                .context("slot storage entry count overflow")?;
            if count > self.limits.files {
                return Err(SlotStorageQuota {
                    reason: format!("directory contains more than {} entries", self.limits.files),
                }
                .into());
            }
            let metadata = entry
                .metadata()
                .context("read slot storage entry metadata; check filesystem health")?;
            if metadata.is_file() {
                bytes = bytes
                    .checked_add(metadata.len())
                    .context("slot storage byte count overflow")?;
            }
            if bytes > self.limits.total_bytes {
                return Err(SlotStorageQuota {
                    reason: format!("directory exceeds {} bytes", self.limits.total_bytes),
                }
                .into());
            }
        }
        Ok((bytes, count))
    }

    pub(crate) fn save<T>(
        &self,
        name: &SlotFilename,
        write: impl FnOnce(&mut SlotWriter) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let _gate=self.gate.lock().map_err(|_| anyhow::anyhow!("slot storage lock is poisoned; restart the server after resolving the storage fault"))?;
        let (used, count) = self.inventory()?;
        let exists = match self.root.symlink_metadata(name.as_str()) {
            Ok(_) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                return Err(e).context("inspect slot save destination; check directory access");
            }
        };
        if !exists && count >= self.limits.files {
            return Err(SlotStorageQuota {
                reason: format!("file count reached {}", self.limits.files),
            }
            .into());
        }
        let available = self.limits.total_bytes - used;
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random)
            .context("generate a unique slot staging filename; check system randomness")?;
        let temp = format!(
            "{TEMP_PREFIX}{}.tmp",
            random
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = self
            .root
            .open_with(&temp, &options)
            .context("create unique slot staging file; check storage space and permissions")?;
        let staged = StagedFile {
            storage: self,
            name: temp,
            published: false,
        };
        let mut writer = SlotWriter {
            file,
            written: 0,
            limit: available.min(self.limits.file_bytes),
        };
        let result = write(&mut writer).with_context(|| {
            format!(
                "save slot state filename {:?} under configured directory '{}'",
                name.as_str(),
                self.configured_path.display()
            )
        })?;
        writer
            .file
            .sync_all()
            .context("sync complete slot state file before publication; check filesystem health")?;
        drop(writer);
        staged.publish(name)?;
        Ok(result)
    }

    pub(crate) fn restore<T>(
        &self,
        name: &SlotFilename,
        read: impl FnOnce(&mut cap_std::fs::File) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let _gate = self
            .gate
            .lock()
            .map_err(|_| anyhow::anyhow!("slot storage lock is poisoned; restart the server"))?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let mut file=self.root.open_with(name.as_str(),&options)
            .with_context(||format!("open slot state filename {:?} in configured directory '{}'; save the file first and ensure it stays inside the slot directory",name.as_str(),self.configured_path.display()))?;
        let metadata = file
            .metadata()
            .context("inspect opened slot state file; check read access")?;
        if !metadata.is_file() {
            return Err(SlotStorageInput(format!(
                "filename {:?} is not a regular file",
                name.as_str()
            ))
            .into());
        }
        if metadata.len() > self.limits.file_bytes {
            return Err(SlotStorageInput(format!(
                "filename {:?} contains {} bytes, exceeding {}",
                name.as_str(),
                metadata.len(),
                self.limits.file_bytes
            ))
            .into());
        }
        read(&mut file).with_context(|| {
            format!(
                "restore slot state filename {:?}; use an intact file from the loaded model",
                name.as_str()
            )
        })
    }
}

struct StagedFile<'a> {
    storage: &'a SlotStorage,
    name: String,
    published: bool,
}
impl StagedFile<'_> {
    fn publish(mut self, target: &SlotFilename) -> anyhow::Result<()> {
        self.storage.root.rename(&self.name,&self.storage.root,target.as_str())
            .context("atomically publish slot state file; check destination access and filesystem health")?;
        self.published = true;
        Ok(())
    }
}
impl Drop for StagedFile<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _ = self.storage.root.remove_file(&self.name);
        }
    }
}

pub(crate) struct SlotWriter {
    file: cap_std::fs::File,
    written: u64,
    limit: u64,
}
impl Write for SlotWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.limit.saturating_sub(self.written) {
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                SlotStorageQuota {
                    reason: format!(
                        "atomic staging has {} bytes available",
                        self.limit.saturating_sub(self.written)
                    ),
                },
            ));
        }
        let n = self.file.write(bytes)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn name(s: &str) -> SlotFilename {
        SlotFilename::new(s.into()).unwrap()
    }
    fn storage(path: &Path) -> SlotStorage {
        SlotStorage::open(
            path,
            SlotStorageLimits {
                file_bytes: 256,
                total_bytes: 512,
                files: 2,
            },
        )
        .unwrap()
    }
    #[test]
    fn filenames_reject_escape_and_reserved_names() {
        for value in [
            "",
            " ",
            ".",
            "..",
            "../outside",
            "/absolute",
            "a/b",
            "a\\b",
            "C:disk",
            ".nxrt-slots.lock",
            "line\nfeed",
        ] {
            assert!(SlotFilename::new(value.into()).is_err(), "{value:?}");
        }
        assert!(SlotFilename::new("聊天-cache.bin".into()).is_ok());
    }
    #[test]
    fn failed_replacement_preserves_file_and_removes_staging() {
        let dir = tempfile::tempdir().unwrap();
        let store = storage(dir.path());
        store
            .save(&name("cache"), |f| {
                f.write_all(b"original")?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .save(&name("cache"), |f| -> anyhow::Result<()> {
                    f.write_all(b"partial")?;
                    anyhow::bail!("injected write failure")
                })
                .is_err()
        );
        assert_eq!(
            std::fs::read(dir.path().join("cache")).unwrap(),
            b"original"
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
        drop(store);
        let reopened = storage(dir.path());
        let content = reopened
            .restore(&name("cache"), |f| {
                let mut b = Vec::new();
                f.read_to_end(&mut b)?;
                Ok(b)
            })
            .unwrap();
        assert_eq!(content, b"original");
    }
    #[test]
    fn quotas_bound_staging_and_entry_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = storage(dir.path());
        assert!(
            store
                .save(&name("large"), |f| {
                    f.write_all(&[0; 257])?;
                    Ok(())
                })
                .is_err()
        );
        assert!(!dir.path().join("large").exists());
        for file in ["one", "two"] {
            store
                .save(&name(file), |f| {
                    f.write_all(b"saved")?;
                    Ok(())
                })
                .unwrap();
        }
        assert!(store.save(&name("third"), |_| Ok(())).is_err());
        store
            .save(&name("one"), |f| {
                f.write_all(b"replaced")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
    }
    #[test]
    fn directory_has_exclusive_process_owner() {
        let dir = tempfile::tempdir().unwrap();
        let store = storage(dir.path());
        assert!(SlotStorage::open(dir.path(), store.limits).is_err());
        drop(store);
        let _reopened = storage(dir.path());
    }
    #[cfg(unix)]
    #[test]
    fn capability_directory_stays_pinned_and_rejects_symlink_restore() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let parent = tempfile::tempdir().unwrap();
        let original = parent.path().join("root");
        let store = storage(&original);
        let moved = parent.path().join("moved");
        std::fs::rename(&original, &moved).unwrap();
        std::fs::create_dir(&original).unwrap();
        store
            .save(&name("cache"), |f| {
                f.write_all(b"owned")?;
                Ok(())
            })
            .unwrap();
        assert!(!original.join("cache").exists());
        assert_eq!(
            std::fs::metadata(moved.join("cache"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        let outside = parent.path().join("outside");
        std::fs::write(&outside, b"secret").unwrap();
        symlink(&outside, moved.join("escape")).unwrap();
        assert!(store.restore(&name("escape"), |_| Ok(())).is_err());
        assert_eq!(std::fs::read(outside).unwrap(), b"secret");
    }
}
