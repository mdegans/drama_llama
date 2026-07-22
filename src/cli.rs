//! CLI-shaped types shared by this crate's binaries and its examples.
//!
//! Two things live here and nothing else: the runtime backend selector
//! ([`BackendKind`]) and the union-of-all-backends load knobs
//! ([`BackendArgs`]) that a binary choosing its backend at runtime must
//! flatten. Per-backend load options are *not* CLI types — they are
//! [`LlamaCppOptions`](crate::LlamaCppOptions) and friends, and a
//! single-backend binary should flatten those directly.
//!
//! # Why the union has to exist
//!
//! `#[command(flatten)]` resolves at compile time and `--backend` is
//! chosen at run time, so a multi-backend binary cannot flatten
//! `B::Options` — there is no single `B` yet when clap builds the
//! command. The union struct is the only shape that works, and it must
//! then be narrowed to a concrete backend's options with
//! [`TryFrom`], which is where an inapplicable flag gets caught.

/// Inference backend selector for a `--backend` flag. Variants are cfg-gated
/// to whichever crate features are enabled, so a single-backend build gets a
/// single-variant flag rather than one that can name a backend that isn't
/// there.
///
/// Shared by `bin/blallama` and the examples' `CommonArgs` so there is one
/// answer to "which backends does this build have" rather than one per
/// consumer.
///
/// Note the `cli` feature implies `llama-cpp` (see `Cargo.toml`), so
/// [`Self::LlamaCpp`] is always present wherever this type compiles today.
/// The cfg is kept anyway: it is what the enum *means*, and it is what makes
/// splitting `cli` into `clap`-only a one-line change rather than a hunt.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum BackendKind {
    #[cfg(feature = "llama-cpp")]
    LlamaCpp,
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    Moeflux,
}

/// Default `--backend` value: prefer llama-cpp when both backends are
/// compiled in (it has been blallama's default for its whole life, and it is
/// the backend the example models are packaged for).
pub const fn default_backend_kind() -> BackendKind {
    #[cfg(feature = "llama-cpp")]
    {
        BackendKind::LlamaCpp
    }
    #[cfg(all(
        all(feature = "moeflux", target_os = "macos"),
        not(feature = "llama-cpp"),
    ))]
    {
        BackendKind::Moeflux
    }
}

/// The `--n-ctx` this repo's front-ends fall back to when the user says
/// nothing. **Not** a library default: the library honours what it is
/// given and llama.cpp's own default (512) applies when that is nothing.
///
/// 32k because it is sized to the workload rather than to the model.
/// Recent local models reach 1M, and agentic use wants ~64k at minimum,
/// but the shape that actually runs here — an Agora seed agent, five
/// forum turns plus a closing interview — lands at 40–60k with a ~7k
/// prefix, and occasionally a little over. 32k covers the common case
/// without committing every `--help`-reading user to a KV cache sized
/// for the worst one; anything longer is a `--n-ctx` away.
pub const DEFAULT_N_CTX: u32 = 32768;

/// The union of every compiled-in backend's load-time knobs, plus the
/// [`BackendKind`] selector — what a binary that picks its backend from
/// a flag should `#[command(flatten)]`.
///
/// Narrow it to a concrete backend's options with [`TryFrom`]:
///
/// ```no_run
/// # use drama_llama::{cli::BackendArgs, LlamaCppOptions};
/// # let args: BackendArgs = todo!();
/// let options = LlamaCppOptions::try_from(&args)?;
/// # Ok::<_, drama_llama::cli::UnsupportedOptions>(())
/// ```
///
/// The model path is deliberately absent: consumers disagree about what
/// it even is. `blallama` takes a *directory* it serves many models out
/// of; an example takes one model file. Both flatten this alongside
/// their own path argument.
#[derive(Debug, Clone, clap::Args)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackendArgs {
    /// Inference backend. Choices are whichever backends this build
    /// compiled in, so a single-backend build offers exactly one.
    #[arg(long, value_enum, default_value_t = default_backend_kind())]
    pub backend: BackendKind,

    /// Max context length, in tokens. llama-cpp only.
    // The one knob here with an eager default, so `--help` can show the
    // number rather than a vague "the backend's". That costs a little
    // precision: "unset" and "set to exactly `DEFAULT_N_CTX`" are
    // indistinguishable afterwards, so `--backend moeflux --n-ctx 32768`
    // is accepted where any other value is refused (see the moeflux
    // `TryFrom` below). Asking for precisely the value you would have
    // got anyway, on a backend that ignores it, is a cheap thing to miss
    // — cheaper than hiding the default from everyone who runs `--help`.
    #[arg(long, default_value_t = DEFAULT_N_CTX)]
    pub n_ctx: u32,

    /// Keep this many KV sequences over one unified cell pool, so an
    /// N-agent workload caches N prefixes instead of thrashing one.
    /// llama-cpp only.
    #[arg(long)]
    pub cache_slots: Option<u32>,

    /// Read the experts as the 2-bit packed layout. moeflux only.
    #[arg(long)]
    pub use_2bit: bool,
}

/// A flag was set that the chosen backend has no notion of.
///
/// Deliberately fatal rather than a warning. Quietly dropping `--n-ctx
/// 32768` leaves a run that *looks* configured and is not — the kind of
/// thing found three benchmarks later, if at all. One typed flag is
/// cheap for the user to remove; a silently wrong context is not cheap
/// for anyone.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("--backend {backend} does not support: {}", .flags.join(" "))]
pub struct UnsupportedOptions {
    /// The backend that was asked for, by [`Backend::NAME`](crate::Backend::NAME).
    pub backend: &'static str,
    /// The offending flags, as the user would have typed them.
    pub flags: Vec<&'static str>,
}

#[cfg(feature = "llama-cpp")]
impl TryFrom<&BackendArgs> for crate::LlamaCppOptions {
    type Error = UnsupportedOptions;

    fn try_from(args: &BackendArgs) -> Result<Self, Self::Error> {
        let mut flags = Vec::new();
        if args.use_2bit {
            flags.push("--use-2bit");
        }
        if !flags.is_empty() {
            return Err(UnsupportedOptions {
                backend: <crate::LlamaCppBackend as crate::Backend>::NAME,
                flags,
            });
        }
        Ok(Self {
            n_ctx: Some(args.n_ctx),
            cache_slots: args.cache_slots,
            ..Self::default()
        })
    }
}

#[cfg(all(feature = "moeflux", target_os = "macos"))]
impl TryFrom<&BackendArgs> for crate::MoefluxOptions {
    type Error = UnsupportedOptions;

    fn try_from(args: &BackendArgs) -> Result<Self, Self::Error> {
        let mut flags = Vec::new();
        // moeflux's context length is `variants::MAX_SEQ_LEN`, baked in
        // at compile time, and it has one physical stream (its seq_id is
        // a namespace label), so neither of these has anywhere to go.
        // Compared against the default rather than tested for
        // presence: `n_ctx` carries an eager default so `--help` can
        // show it, which costs us the ability to tell "unset" from
        // "set to the default". See `BackendArgs::n_ctx`.
        if args.n_ctx != DEFAULT_N_CTX {
            flags.push("--n-ctx");
        }
        if args.cache_slots.is_some() {
            flags.push("--cache-slots");
        }
        if !flags.is_empty() {
            return Err(UnsupportedOptions {
                backend: <crate::MoefluxBackend as crate::Backend>::NAME,
                flags,
            });
        }
        Ok(Self {
            use_2bit: args.use_2bit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> BackendArgs {
        BackendArgs {
            backend: default_backend_kind(),
            n_ctx: DEFAULT_N_CTX,
            cache_slots: None,
            use_2bit: false,
        }
    }

    /// Saying nothing must convert cleanly to every backend — otherwise
    /// a bare `--backend X` would refuse to start.
    #[test]
    fn silence_converts_to_every_backend() {
        #[cfg(feature = "llama-cpp")]
        assert_eq!(
            crate::LlamaCppOptions::try_from(&args()),
            Ok(crate::LlamaCppOptions::default().with_n_ctx(DEFAULT_N_CTX))
        );
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        assert_eq!(
            crate::MoefluxOptions::try_from(&args()),
            Ok(crate::MoefluxOptions::default())
        );
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn llama_cpp_takes_its_own_knobs_and_refuses_moeflux_s() {
        let opts = crate::LlamaCppOptions::try_from(&BackendArgs {
            n_ctx: 4096,
            cache_slots: Some(3),
            ..args()
        })
        .expect("both are llama.cpp knobs");
        assert_eq!(opts.n_ctx, Some(4096));
        assert_eq!(opts.cache_slots, Some(3));

        let err = crate::LlamaCppOptions::try_from(&BackendArgs {
            use_2bit: true,
            ..args()
        })
        .expect_err("2-bit experts are a moeflux concept");
        assert_eq!(err.flags, ["--use-2bit"]);
    }

    /// The whole point of failing fast: a context size the backend
    /// cannot honour is an error naming the flag, not a shrug.
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    #[test]
    fn moeflux_refuses_llama_cpp_knobs_by_name() {
        let err = crate::MoefluxOptions::try_from(&BackendArgs {
            n_ctx: 4096,
            cache_slots: Some(5),
            ..args()
        })
        .expect_err("moeflux has neither");
        assert_eq!(err.flags, ["--n-ctx", "--cache-slots"]);
        assert_eq!(err.backend, "moeflux");
        assert!(err.to_string().contains("--n-ctx --cache-slots"));
    }
}
