//! Platform mapping primitives.
//!
//! Each backend supplies the same four operations: reserve address space,
//! commit a block into part of it, release that block, and release the
//! reservation. The differences are entirely in how the OS spells "reserve
//! without committing, then commit a piece without disturbing the rest".

use crate::VirtualMemoryError;
use std::ptr::NonNull;

#[cfg(windows)]
mod imp {
    use super::{NonNull, VirtualMemoryError};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER,
        PAGE_READWRITE, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    /// Windows splits and coalesces reservations at its allocation granularity
    /// (64 KiB), not at page size, so that is the unit a placeholder can be
    /// carved into.
    pub fn granularity() -> usize {
        // SAFETY: `GetSystemInfo` fills the struct and cannot fail.
        let info = unsafe {
            let mut info: SYSTEM_INFO = std::mem::zeroed();
            GetSystemInfo(&mut info);
            info
        };
        info.dwAllocationGranularity as usize
    }

    pub fn reserve(len: usize) -> Result<NonNull<u8>, VirtualMemoryError> {
        // A placeholder reservation can later be split and have parts replaced
        // individually, which plain MEM_RESERVE cannot.
        // SAFETY: a null base asks the OS to choose the address.
        let base = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        };
        NonNull::new(base.cast::<u8>()).ok_or_else(|| last_error("VirtualAlloc2 (reserve)"))
    }

    pub unsafe fn map(
        base: NonNull<u8>,
        gap: (usize, usize),
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let target = unsafe { base.as_ptr().add(offset) };
        // Splitting is only needed when the block is a strict subset of the
        // placeholder that contains it. Asking to split a placeholder into
        // exactly itself is an error, so the two cases have to be told apart --
        // and the placeholder is bounded by already-mapped neighbours, not by
        // the reservation.
        let (gap_start, gap_len) = gap;
        if !(offset == gap_start && len == gap_len) {
            // SAFETY: `target` is inside the reservation and the length is
            // granularity-aligned.
            let split =
                unsafe { VirtualFree(target.cast(), len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
            if split == 0 {
                return Err(last_error("VirtualFree (split placeholder)"));
            }
        }
        // SAFETY: the placeholder covering `target` is now exactly `len` bytes.
        let committed = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                target.cast(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        if committed.is_null() {
            return Err(last_error("VirtualAlloc2 (replace placeholder)"));
        }
        Ok(())
    }

    pub unsafe fn unmap(
        base: NonNull<u8>,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let target = unsafe { base.as_ptr().add(offset) };
        // Preserve the placeholder so the address space stays reserved and the
        // offset can be mapped again later.
        // SAFETY: `target`/`len` came from a successful `map`.
        let freed =
            unsafe { VirtualFree(target.cast(), len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        if freed == 0 {
            return Err(last_error("VirtualFree (release block)"));
        }
        Ok(())
    }

    pub unsafe fn release(base: NonNull<u8>, _len: usize) {
        // SAFETY: `base` came from `reserve`; MEM_RELEASE requires a zero size.
        unsafe {
            VirtualFree(base.as_ptr().cast(), 0, MEM_RELEASE);
        }
    }

    fn last_error(operation: &'static str) -> VirtualMemoryError {
        let error = std::io::Error::last_os_error();
        VirtualMemoryError::Os {
            operation,
            reason: error.to_string(),
            code: error.raw_os_error().unwrap_or(0),
        }
    }

    use windows_sys::Win32::System::Memory::{MEM_REPLACE_PLACEHOLDER, PAGE_NOACCESS};
}

#[cfg(unix)]
mod imp {
    use super::{NonNull, VirtualMemoryError};

    /// On Unix a reservation can be carved at page granularity, which is far
    /// finer than Windows' 64 KiB and finer than a KV page, so virtual
    /// contiguity costs essentially no wasted memory here.
    pub fn granularity() -> usize {
        // SAFETY: `sysconf` with a valid name returns a positive value or -1.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size > 0 { size as usize } else { 4096 }
    }

    pub fn reserve(len: usize) -> Result<NonNull<u8>, VirtualMemoryError> {
        // PROT_NONE plus MAP_NORESERVE takes address space without asking the
        // kernel to account for backing store.
        // SAFETY: a null hint asks the kernel to choose the address.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(last_error("mmap (reserve)"));
        }
        NonNull::new(base.cast::<u8>()).ok_or_else(|| last_error("mmap (reserve)"))
    }

    pub unsafe fn map(
        base: NonNull<u8>,
        _gap: (usize, usize),
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let target = unsafe { base.as_ptr().add(offset) };
        // MAP_FIXED replaces exactly this span of the reservation and leaves the
        // rest untouched.
        // SAFETY: `target` is inside the reservation and page-aligned.
        let mapped = unsafe {
            libc::mmap(
                target.cast(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(last_error("mmap (commit block)"));
        }
        Ok(())
    }

    pub unsafe fn unmap(
        base: NonNull<u8>,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let target = unsafe { base.as_ptr().add(offset) };
        // Map PROT_NONE back over the span rather than munmap: munmap would
        // punch a hole out of the reservation and another allocation could take
        // the address before the range is done with it.
        // SAFETY: `target`/`len` came from a successful `map`.
        let restored = unsafe {
            libc::mmap(
                target.cast(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if restored == libc::MAP_FAILED {
            return Err(last_error("mmap (release block)"));
        }
        Ok(())
    }

    pub unsafe fn release(base: NonNull<u8>, len: usize) {
        // SAFETY: `base`/`len` describe the whole reservation.
        unsafe {
            libc::munmap(base.as_ptr().cast(), len);
        }
    }

    fn last_error(operation: &'static str) -> VirtualMemoryError {
        let error = std::io::Error::last_os_error();
        VirtualMemoryError::Os {
            operation,
            reason: error.to_string(),
            code: error.raw_os_error().unwrap_or(0),
        }
    }
}

pub(crate) use imp::{granularity, map, release, reserve, unmap};
