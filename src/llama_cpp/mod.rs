//! [`llama_cpp_sys_3`] [`Decoder`] and [`Model`] [`Backend`].
//!
//! [`Decoder`]: crate::Decoder
//! [`Model`]: crate::Model

pub mod decoder;
pub mod engine;
pub mod model;
#[cfg(feature = "mtmd")]
pub mod mtmd;
pub mod options;

pub use crate::Backend;
pub use decoder::{DecodeError, FlashAttention, LlamaCppDecoder, NewError};
pub use engine::LlamaCppEngine;
pub use model::{llama_quantize, LlamaCppModel};
#[cfg(feature = "mtmd")]
pub use mtmd::{Mtmd, MtmdParams};
pub use options::LlamaCppOptions;

/// Names of the GPU-class compute devices ggml discovered, in registry
/// order. Empty means every model will run on the CPU.
///
/// This asks ggml what it actually found rather than what the build
/// requested, and those differ in ways nothing else reports. A build with
/// `feature = "cuda"` links CUDA and still silently falls back to the CPU
/// when the driver is unusable — a kernel module that no longer matches
/// the userspace libraries (`cuInit` → `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`,
/// the state an unattended driver upgrade leaves behind until reboot) is
/// indistinguishable, from inside the process, from a box with no card.
///
/// Only `GPU` counts. `ACCEL` and `IGPU` are deliberately excluded: the
/// question this answers is "did the discrete accelerator this build was
/// compiled for actually show up", and an integrated fallback is a no.
pub fn gpu_device_names() -> Vec<String> {
    // Safety: the backend registry is initialized on first access inside
    // ggml and is read-only thereafter. `dev_get` is in range by the
    // `dev_count` bound, and `dev_name` returns a borrowed static C string
    // owned by the device, which we copy before returning.
    unsafe {
        let count = llama_cpp_sys_3::ggml_backend_dev_count();
        (0..count)
            .filter_map(|i| {
                let dev = llama_cpp_sys_3::ggml_backend_dev_get(i);
                if dev.is_null() {
                    return None;
                }
                if llama_cpp_sys_3::ggml_backend_dev_type(dev)
                    != llama_cpp_sys_3::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                {
                    return None;
                }
                let name = llama_cpp_sys_3::ggml_backend_dev_name(dev);
                if name.is_null() {
                    return None;
                }
                Some(
                    std::ffi::CStr::from_ptr(name)
                        .to_string_lossy()
                        .into_owned(),
                )
            })
            .collect()
    }
}

/// Tag for the llama-cpp [`Backend`]. Use as a type parameter for [`Engine`] or
/// [`Session`].
///
/// [`Engine`]: crate::Engine
/// [`Session`]: crate::Session
#[derive(Debug, Clone, Copy)]
pub struct LlamaCppBackend;

impl Backend for LlamaCppBackend {
    const NAME: &'static str = "llama-cpp";
    type Decoder = LlamaCppDecoder;
    type Model = LlamaCppModel;
    #[cfg(feature = "mtmd")]
    type Vision = mtmd::Mtmd;
    #[cfg(not(feature = "mtmd"))]
    type Vision = crate::NoVision;

    fn is_supported_model(name: &str, meta: &std::fs::Metadata) -> bool {
        // `<model>.mmproj.gguf` is a vision *projector* sidecar, not a
        // standalone model — it auto-loads alongside its base model and
        // fails if asked to load on its own. Exclude it so it never
        // surfaces in `/api/tags` (or gets picked as a default).
        meta.is_file()
            && name.ends_with(".gguf")
            && !name.ends_with(".mmproj.gguf")
    }

    /// Routes both llama.cpp's and ggml's sinks — they are separate
    /// globals upstream and this installs the same trampoline in each.
    fn set_log_callback<F>(f: F) -> Result<(), crate::backend::NotImplemented>
    where
        F: Fn(crate::LogLevel, &str) + Send + Sync + 'static,
    {
        crate::log::set_log_callback(f);
        Ok(())
    }

    fn clear_log_callback() -> Result<(), crate::backend::NotImplemented> {
        crate::log::clear_log_callback();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// A `cuda` build that quietly ran on the CPU is a broken box, not a
    /// passing test run.
    ///
    /// ggml falls back to the CPU whenever the CUDA driver is unusable, and
    /// the model tier stays **green** through it — correctness does not
    /// depend on the device. Timing does not give it away either: only
    /// generation-heavy tests slow down, while load-dominated ones get
    /// *faster* (no VRAM upload, and the weights are already in page
    /// cache), so the suite's wall clock can move either direction. That
    /// makes an explicit device assertion the only reliable signal.
    ///
    /// The state this catches, seen 2026-07-25: an unattended upgrade of
    /// `nvidia-driver-580-server` (580.159.03 → 580.173.02) left the old
    /// kernel module loaded against new userspace libraries. `nvidia-smi`
    /// failed outright, `cuInit` returned 804
    /// (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`), ggml found zero CUDA
    /// devices — and the whole model tier passed on the CPU. Reboot fixes
    /// it; nothing in CI said so.
    ///
    /// Set `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1` to exercise the CPU path on
    /// purpose. That is a legitimate thing to want; silently getting it is
    /// not. Note the CPU path is far slower on the generation-heavy tests
    /// (`whodunit` and friends), so expect the ignored tier to crawl.
    ///
    /// The `cuda` gate is on the **body**, not on `#[test]`, so the test is
    /// listed in every configuration and `nextest list` returns the same
    /// count everywhere. That keeps README.md's hand-counted numbers
    /// config-independent — gating the attribute instead makes the badge
    /// mean one thing under `llama-cpp` and another under `llama-cpp-cpu`,
    /// which is how #82 went red. Same rule for any future
    /// backend-specific test.
    #[test]
    fn cuda_build_has_a_gpu_device() {
        #[cfg(not(feature = "cuda"))]
        eprintln!(
            "cuda_build_has_a_gpu_device: not a `cuda` build — nothing to \
             assert"
        );

        #[cfg(feature = "cuda")]
        {
            if std::env::var_os("DRAMA_LLAMA_ALLOW_CPU_FALLBACK").is_some() {
                eprintln!(
                    "cuda_build_has_a_gpu_device: \
                     DRAMA_LLAMA_ALLOW_CPU_FALLBACK set — skipping (CPU path \
                     is intentional)"
                );
                return;
            }

            let gpus = super::gpu_device_names();
            assert!(
                !gpus.is_empty(),
                "built with `feature = \"cuda\"` but ggml found no GPU \
                 device — every model test will silently run on the CPU.\n\
                 \n\
                 Most likely the driver is unusable rather than absent. \
                 Check:\n\
                 \n\
                 \tnvidia-smi                # NVML mismatch reports here first\n\
                 \tcat /proc/driver/nvidia/version   # loaded kernel module\n\
                 \tls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*  # userspace\n\
                 \n\
                 A driver upgrade with the old module still loaded needs a \
                 reboot (or an nvidia module reload).\n\
                 \n\
                 Testing the CPU path deliberately? \
                 Set DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1."
            );
        }
    }
}
