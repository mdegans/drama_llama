//! [`SessionTransport`] — a [`misanthropic::Transport`] over a locally-owned
//! [`Session`], so anything generic over a transport (the upstream
//! [`Chat`](misanthropic::chat::Chat) driver, agentkit's reactor) drives
//! local inference exactly as it would the API [`Client`].
//!
//! Completions run on tokio's blocking pool, one at a time: the session is
//! `Send` but not `Sync` (decode mutates the KV cache), so it lives behind
//! an async [`Mutex`](tokio::sync::Mutex) and
//! [`max_concurrency`](misanthropic::Transport::max_concurrency) stays at
//! its default of 1. Cloning the transport clones the *handle* — every
//! clone serializes through the same session and the same KV cache, which
//! is the honest shape of one local model on one GPU.
//!
//! [`Client`]: misanthropic::Client

use std::sync::Arc;

use misanthropic::{model, response, CachedPrompt, Prompt, Quirks, Transport};

use crate::{backend::Model as _, Backend, Session, SessionError, Token};

/// A [`misanthropic::Transport`] over a locally-owned [`Session`] — see the
/// module docs for the concurrency model.
pub struct SessionTransport<B: Backend> {
    session: Arc<tokio::sync::Mutex<Session<B>>>,
    /// Snapshotted at construction: [`models`](Transport::models) must not
    /// need the lock (it may be called while a completion holds it).
    model_info: model::ModelInfo,
}

// Manual impl: `derive(Clone)` would demand `B: Clone`, which the handle
// (an `Arc` + plain data) does not actually need.
impl<B: Backend> Clone for SessionTransport<B> {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            model_info: self.model_info.clone(),
        }
    }
}

impl<B: Backend> SessionTransport<B> {
    /// Wrap `session`. The model's display name is snapshotted here as the
    /// transport's advertised [`ModelInfo`](model::ModelInfo).
    ///
    /// The session's prefix cache is switched **on**: this transport's
    /// [`quirks`](Transport::quirks) advertise breakpoint-keyed prefix
    /// reuse, which would be a lie against a session that ignores its
    /// breakpoints. (A no-op if the caller already enabled it.)
    pub fn new(session: Session<B>) -> Self {
        let session = session.with_prefix_cache(true);
        let display_name = session
            .engine()
            .model
            .display_name()
            .unwrap_or_else(|| "unknown".to_string());
        let model_info =
            model::ModelInfo::new(display_name.clone(), display_name);
        Self {
            session: Arc::new(tokio::sync::Mutex::new(session)),
            model_info,
        }
    }

    /// The shared session handle — lock it between completions to inspect
    /// usage, adjust sampling, or reseed. Completions in flight hold the
    /// lock, so this never races a decode.
    pub fn session(&self) -> Arc<tokio::sync::Mutex<Session<B>>> {
        self.session.clone()
    }

    /// [`Session::scan_text_for_specials`] through the shared handle:
    /// scan `text` for content that would tokenize to a reserved
    /// chat-framing special, returning the first offender's
    /// `(id, piece)`. For relay tools (mail, docket filings) holding a
    /// transport clone — check at send time and bounce the message
    /// back to its author as a recoverable `is_error` tool result,
    /// instead of poisoning the recipient's prompt and killing their
    /// loop at ingest. Briefly awaits the session lock (a completion
    /// in flight holds it).
    pub async fn scan_text_for_specials(
        &self,
        text: &str,
    ) -> Option<(Token, String)> {
        self.session.lock().await.scan_text_for_specials(text)
    }
}

impl<B> SessionTransport<B>
where
    B: Backend + 'static,
    Session<B>: Send,
{
    /// Run one completion on the blocking pool. The owned guard moves into
    /// the closure — the lock is held for exactly the duration of the
    /// completion, and the async caller stays cancel-safe (dropping the
    /// future before the spawn merely abandons the queued lock).
    async fn complete(
        &self,
        prompt: Prompt,
    ) -> Result<response::Message, SessionError> {
        let guard = self.session.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || {
            let mut guard = guard;
            guard.complete_response(&prompt)
        })
        .await?
    }

    /// The blallama-class quirk profile shared by both prompt-type impls:
    /// prefix reuse keys on the end-of-assistant render, and changing
    /// `output_config` does not invalidate the prefix cache.
    ///
    /// `cache_stats_unreported` stays `false`, which assumes the
    /// wrapped session has its prefix cache enabled — the session
    /// reports both cache counters as `None` when it's off.
    /// [`Self::new`] enforces the assumption (`with_prefix_cache(true)`).
    fn local_quirks() -> Quirks {
        // `Quirks` is `#[non_exhaustive]`: mutate a default, no literal.
        let mut quirks = Quirks::default();
        quirks.breakpoint_after_assistant = true;
        quirks.output_config_cache_safe = true;
        quirks
    }
}

#[async_trait::async_trait]
impl<B> Transport for SessionTransport<B>
where
    B: Backend + 'static,
    Session<B>: Send,
{
    type Error = SessionError;

    async fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<response::Message, Self::Error> {
        self.complete(prompt.clone()).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        Ok(std::iter::once(self.model_info.clone()).collect())
    }

    fn quirks(&self) -> Quirks {
        Self::local_quirks()
    }
}

#[async_trait::async_trait]
impl<B> Transport<CachedPrompt> for SessionTransport<B>
where
    B: Backend + 'static,
    Session<B>: Send,
{
    type Error = SessionError;

    async fn send(
        &self,
        prompt: &CachedPrompt,
    ) -> Result<response::Message, Self::Error> {
        // The session renders from the inner `Prompt`; the cached wrapper's
        // frozen-prefix guarantee is upheld by construction on the caller's
        // side and needs no translation here.
        self.complete(prompt.as_ref().clone()).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        Ok(std::iter::once(self.model_info.clone()).collect())
    }

    fn quirks(&self) -> Quirks {
        Self::local_quirks()
    }
}

#[cfg(feature = "llama-cpp")]
static_assertions::assert_impl_all!(
    crate::LlamaCppSession: Send
);
#[cfg(feature = "llama-cpp")]
static_assertions::assert_impl_all!(
    SessionTransport<crate::LlamaCppBackend>: Send, Sync, Clone
);
static_assertions::assert_impl_all!(
    SessionError: std::error::Error, Send, Sync
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The blallama profile, pinned: these two flags are what the upstream
    /// `Chat` driver keys its breakpoint placement on.
    #[test]
    fn quirks_profile_is_blallama_class() {
        #[cfg(feature = "llama-cpp")]
        {
            let quirks =
                SessionTransport::<crate::LlamaCppBackend>::local_quirks();
            assert!(quirks.breakpoint_after_assistant);
            assert!(quirks.output_config_cache_safe);
            assert!(!quirks.cache_markers_ignored);
            assert!(!quirks.tool_choice_not_respected);
            assert!(!quirks.cache_stats_unreported);
        }
    }

    #[cfg(feature = "llama-cpp")]
    fn model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/model.gguf")
    }

    /// `send` produces a fully-populated response, `models` advertises the
    /// session's model, and two concurrent `send`s serialize through the
    /// mutex without deadlock.
    #[cfg(feature = "llama-cpp")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "long running; requires models/model.gguf"]
    async fn send_populates_the_response_and_serializes() {
        let session = crate::LlamaCppSession::from_path_sync(model_path())
            .unwrap()
            .quiet();
        let transport = SessionTransport::new(session);

        // Two prompt-type impls exist; pin one for the call.
        let models = Transport::<Prompt>::models(&transport).await.unwrap();
        let info = models.into_iter().next().expect("one advertised model");

        let mut prompt = Prompt::default();
        // Small generation bound to keep this ignored test fast (the cap
        // now lives on the prompt, not the session).
        prompt.max_tokens = std::num::NonZeroU32::new(8).unwrap();
        prompt.messages.push(
            (misanthropic::prompt::message::Role::User, "Say hi.").into(),
        );

        let (a, b) =
            tokio::join!(transport.send(&prompt), transport.send(&prompt));
        let (a, b) = (a.unwrap(), b.unwrap());

        for response in [&a, &b] {
            assert!(!response.id.is_empty());
            assert_eq!(response.model, info.id);
            assert!(response.usage.output_tokens > 0);
            assert!(response.stop_reason.is_some());
        }
        // Distinct completions get distinct correlation ids.
        assert_ne!(a.id, b.id);
    }
}
