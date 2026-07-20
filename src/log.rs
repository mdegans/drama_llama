//! Process-global log callback for llama.cpp and ggml.
//!
//! llama.cpp installs a log sink via `llama_log_set(callback, user_data)`;
//! ggml has its own twin via `ggml_log_set`. Both are global state — there
//! is no per-context scope. This module hosts a single Rust closure
//! registered against both sinks, dispatched from a C trampoline.
//! Replacing the closure replaces the global slot; the prior closure is
//! dropped after the C side has been swapped, so an in-flight log line on
//! another thread either runs the old closure to completion or finds an
//! empty slot and is dropped.
//!
//! The closure is held behind an [`Arc`], and `trampoline` clones that
//! `Arc` out of the slot and *releases the lock before calling it*. This
//! makes the callback re-entrant-safe: a sink that itself triggers a
//! llama.cpp/ggml log line, or that calls [`set_log_callback`] /
//! [`clear_log_callback`], no longer deadlocks on the non-reentrant
//! `Mutex`. (A sink that logs unconditionally will recurse — that is the
//! caller's bug, but a visible stack overflow beats a silent hang.) The
//! cloned `Arc` keeps the closure alive for the whole call even if
//! another thread swaps or clears the slot meanwhile.
//!
//! Prefer [`set_log_callback`] (closures). [`set_log_callback_raw`] is
//! available for callers that already hold a `ggml_log_callback` function
//! pointer.

use std::ffi::{c_void, CStr};
use std::sync::{Arc, Mutex};

use llama_cpp_sys_3::{
    ggml_log_callback, ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_CONT,
    ggml_log_level_GGML_LOG_LEVEL_DEBUG, ggml_log_level_GGML_LOG_LEVEL_ERROR,
    ggml_log_level_GGML_LOG_LEVEL_INFO, ggml_log_level_GGML_LOG_LEVEL_NONE,
    ggml_log_level_GGML_LOG_LEVEL_WARN, ggml_log_set, llama_log_set,
};

/// Log severity from llama.cpp / ggml. Unknown C values map to
/// [`LogLevel::Other`] so a future llama.cpp release that grows the
/// enum can't silently drop messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// `GGML_LOG_LEVEL_NONE` — used for raw `LLAMA_LOG(...)` lines
    /// that carry no severity (e.g. continuation prints).
    None,
    Debug,
    Info,
    Warn,
    Error,
    /// `GGML_LOG_LEVEL_CONT` — continuation of the previous line.
    Cont,
    /// Unknown / future variant. Carries the raw C value.
    Other(ggml_log_level),
}

impl From<ggml_log_level> for LogLevel {
    fn from(level: ggml_log_level) -> Self {
        #[allow(non_upper_case_globals)]
        match level {
            ggml_log_level_GGML_LOG_LEVEL_NONE => LogLevel::None,
            ggml_log_level_GGML_LOG_LEVEL_DEBUG => LogLevel::Debug,
            ggml_log_level_GGML_LOG_LEVEL_INFO => LogLevel::Info,
            ggml_log_level_GGML_LOG_LEVEL_WARN => LogLevel::Warn,
            ggml_log_level_GGML_LOG_LEVEL_ERROR => LogLevel::Error,
            ggml_log_level_GGML_LOG_LEVEL_CONT => LogLevel::Cont,
            other => LogLevel::Other(other),
        }
    }
}

type LogFn = Arc<dyn Fn(LogLevel, &str) + Send + Sync + 'static>;

/// Global closure slot. Replaced by [`set_log_callback`]; cleared by
/// [`clear_log_callback`]. Read by [`trampoline`] on every log line
/// llama.cpp or ggml emits while [`trampoline`] is the registered sink.
static GLOBAL_LOG_CALLBACK: Mutex<Option<LogFn>> = Mutex::new(None);

/// C trampoline registered with both llama and ggml log sinks.
/// Dispatches to the boxed closure in [`GLOBAL_LOG_CALLBACK`]. No-op if
/// the slot is empty (logs are dropped).
unsafe extern "C" fn trampoline(
    level: ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp / ggml pass a null-terminated C string that's
    // valid for the duration of this call. We borrow into a `&str` and
    // never retain the pointer past return.
    let s = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let lvl = LogLevel::from(level);

    // Clone the `Arc` out and drop the guard *before* calling the
    // closure. Holding the lock across the call would deadlock a
    // consumer whose sink logs (re-entering here) or calls
    // set/clear_log_callback (both take this same non-reentrant lock).
    // The cloned `Arc` keeps the closure alive across the call even if
    // another thread swaps or clears the slot in the meantime.
    let cb = {
        let guard = GLOBAL_LOG_CALLBACK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map(Arc::clone)
    };
    if let Some(cb) = cb {
        // The closure is consumer-supplied and runs on whatever thread
        // llama.cpp/ggml logs from. Unwinding out of an `extern "C"` fn
        // hits Rust's abort-on-unwind shim, so a stray `.unwrap()` in
        // someone's log sink would take the process down with no
        // diagnostic beyond the panic message. Swallow it: a dropped
        // log line is a better failure mode than a dead process, and
        // losing logging is exactly the situation where you can least
        // afford to lose the process.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cb(lvl, s)
        }));
    }
}

/// Install a Rust closure as the log callback for both llama.cpp and
/// ggml. Subsequent calls replace the previous closure. The closure
/// runs on whatever thread llama.cpp / ggml emits from — bring your
/// own synchronization if you need single-threaded sink semantics.
pub fn set_log_callback<F>(f: F)
where
    F: Fn(LogLevel, &str) + Send + Sync + 'static,
{
    {
        let mut guard = GLOBAL_LOG_CALLBACK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *guard = Some(Arc::new(f));
    }
    // SAFETY: `trampoline` reads from the `'static`
    // `GLOBAL_LOG_CALLBACK` slot only — it never touches `user_data`,
    // so a null pointer is fine.
    unsafe {
        llama_log_set(Some(trampoline), std::ptr::null_mut());
        ggml_log_set(Some(trampoline), std::ptr::null_mut());
    }
}

/// Clear the log callback and restore default (stderr) output for
/// both llama.cpp and ggml.
pub fn clear_log_callback() {
    // SAFETY: passing `None` restores the C default. We swap the C
    // side *before* dropping the boxed closure so any in-flight log
    // line on another thread either runs the old closure to completion
    // or — if it lands after the swap — finds an empty slot and no-ops.
    unsafe {
        llama_log_set(None, std::ptr::null_mut());
        ggml_log_set(None, std::ptr::null_mut());
    }
    let mut guard = GLOBAL_LOG_CALLBACK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// Silence both llama.cpp and ggml log output. Installs a no-op
/// closure; idempotent. Inverse of [`restore_default_logs`].
pub fn silence_logs() {
    set_log_callback(|_, _| {});
}

/// Restore default (stderr) logging. Alias for [`clear_log_callback`].
pub fn restore_default_logs() {
    clear_log_callback();
}

/// Install a raw C callback for both llama.cpp and ggml.
///
/// Prefer [`set_log_callback`] for safe code; this exists for callers
/// that already hold a `ggml_log_callback` function pointer (e.g.
/// FFI-glue layers).
///
/// # Safety
///
/// - `callback` (if `Some`) must be safe to invoke from any thread at
///   any time until it is replaced — llama.cpp and ggml log
///   asynchronously.
/// - `user_data` must remain valid (i.e. not freed, not aliased
///   mutably) for as long as `callback` is registered. The next
///   [`clear_log_callback`] / `set_log_callback*` call unregisters it.
/// - Passing `Some(callback)` here drops any boxed Rust closure that
///   was previously installed via [`set_log_callback`].
pub unsafe fn set_log_callback_raw(
    callback: ggml_log_callback,
    user_data: *mut c_void,
) {
    // SAFETY: forwarded to the caller's contract.
    unsafe {
        llama_log_set(callback, user_data);
        ggml_log_set(callback, user_data);
    }
    // Drop any previously-installed closure — the user has taken over
    // the C-side slot, and the trampoline is no longer the registered
    // sink. Drop happens after the swap, so a racing log line either
    // ran the old trampoline (and found Some(cb)) or hits the new
    // `callback` directly.
    let mut guard = GLOBAL_LOG_CALLBACK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    /// The log callback is process-global; tests that touch it must
    /// serialize against each other or they'll trample the slot.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// Smoke: set/clear is round-trippable, and the trampoline
    /// dispatches level + message through to the installed closure.
    #[test]
    fn set_clear_and_dispatch() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let count = Arc::new(AtomicUsize::new(0));
        let last_level = Arc::new(StdMutex::new(LogLevel::None));
        let last_msg = Arc::new(StdMutex::new(String::new()));
        {
            let c = count.clone();
            let l = last_level.clone();
            let m = last_msg.clone();
            set_log_callback(move |lvl, s| {
                c.fetch_add(1, Ordering::SeqCst);
                *l.lock().unwrap() = lvl;
                *m.lock().unwrap() = s.to_owned();
            });
        }
        let msg = std::ffi::CString::new("hello").unwrap();
        unsafe {
            trampoline(
                ggml_log_level_GGML_LOG_LEVEL_WARN,
                msg.as_ptr(),
                std::ptr::null_mut(),
            );
        }
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(*last_level.lock().unwrap(), LogLevel::Warn);
        assert_eq!(*last_msg.lock().unwrap(), "hello");

        clear_log_callback();
        // Round-trip again to confirm the slot is reusable.
        set_log_callback(|_, _| {});
        clear_log_callback();
    }

    /// Regression for the reentrancy deadlock (issue #56): a sink that
    /// calls back into the log machinery — here `clear_log_callback`,
    /// which takes the same `Mutex` the trampoline used to hold across
    /// the call — must not deadlock. Before the `Arc`-clone-then-unlock
    /// fix this hung; now it completes and the test returns.
    #[test]
    fn sink_may_touch_the_lock_without_deadlock() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let ran = Arc::new(AtomicUsize::new(0));
        {
            let r = ran.clone();
            set_log_callback(move |_, _| {
                r.fetch_add(1, Ordering::SeqCst);
                // Re-enter the lock from inside the callback. The
                // trampoline must already have released it.
                clear_log_callback();
            });
        }
        let msg = std::ffi::CString::new("reentrant").unwrap();
        unsafe {
            trampoline(
                ggml_log_level_GGML_LOG_LEVEL_INFO,
                msg.as_ptr(),
                std::ptr::null_mut(),
            );
        }
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        // The callback cleared itself; a second dispatch is a no-op.
        unsafe {
            trampoline(
                ggml_log_level_GGML_LOG_LEVEL_INFO,
                msg.as_ptr(),
                std::ptr::null_mut(),
            );
        }
        assert_eq!(ran.load(Ordering::SeqCst), 1);

        clear_log_callback();
    }
}
