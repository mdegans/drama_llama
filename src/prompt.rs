use std::fmt::{Display, Formatter};

mod format;
pub use format::Format;

use crate::Model;

/// Yet another stab at a prompt struct. The intended use case is for chat. This
/// takes inspiration from the OpenAI API, but is not intended to be compatible
/// with it.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(all(test, feature = "webchat"), derive(rocket::UriDisplayQuery))]
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "webchat", derive(rocket::form::FromForm))]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
pub struct Prompt {
    /// Setting, as in set and setting. This is the context in which the
    /// interaction takes place. It may be a location, a time, a situation, or
    /// any other context that may be relevant. The composition of a universe.
    #[cfg_attr(feature = "serde", field(validate = len(..4096), default = None))]
    pub setting: Option<String>,
    /// Agent's name, e.g. "Mr. Rogers" or "GPT-5".
    #[cfg_attr(feature = "serde", field(validate = len(..64), default = "assistant"))]
    pub agent: String,
    /// Human's name, e.g. "Alice" or "Bob".
    #[cfg_attr(feature = "serde", field(validate = len(..64), default = "user"))]
    pub human: String,
    /// System's name, e.g. "System", "Narrator", or "God". Should imply
    /// authority to the Agent -- not necessarily to the Human.
    #[cfg_attr(feature = "serde", field(validate = len(..64), default = None))]
    pub system: Option<String>,
    /// Messages in the chat transcript. There must be at least two messages.
    #[cfg_attr(feature = "serde", field(validate = len(2..512)))]
    pub transcript: Vec<Message>,
}

impl Prompt {
    /// Load from a TOML file.
    #[cfg(feature = "toml")]
    pub fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
        let prompt: Prompt =
            toml::from_str(&std::fs::read_to_string(path)?).unwrap();
        Ok(prompt)
    }

    /// Format the prompt in a specific format. This does not add a BOS token so
    /// if this is desired, it must be prepended or [`Format::for_model`] must
    /// be used instead.
    pub fn format<F>(&self, format: Format, f: &mut F) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        format.format_prompt(self, None, f)
    }

    /// Format the prompt for a specific model. This adds a BOS token if the
    /// model requires it. If this is unknown, a BOS token will not be added.
    /// This is the recommended method for formatting a prompt.
    ///
    /// This will first attempt to use native formatting for the model. If a
    /// format would be unknown, it will attempt to apply a chat template using
    /// the model's metadata and `llama.cpp`. If *that* fails, it will use the
    /// [`Format::Unknown`] format.
    ///
    /// This does not add the assistant's prefix to the prompt. If this is
    /// desired, [`format_agent_prefix`] should be called after this method or
    /// [`Model::apply_chat_template`] should be used instead with the `add_ass`
    /// parameter set to `true`.
    /// 
    /// [`format_agent_prefix`]: Self::format_agent_prefix
    pub fn format_for_model<F>(
        &self,
        model: &Model,
        f: &mut F,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        let format = match Format::from_model(model) {
            Some(format) => format,
            None => match model.apply_chat_template(None, self, false) {
                Some(string) => return f.write_str(&string),
                None => Format::Unknown,
            },
        };
        format.format_prompt(self, Some(model), f)
    }

    /// Format the agent's prefix. This should be called after a format method
    /// in order to append the agent's prefix to the prompt which in turn forces
    /// the model to generate a response from the agent's perspective.
    pub fn format_agent_prefix<F>(
        &self,
        format: Format,
        f: &mut F,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        format.format_agent_prefix(f, self)
    }
}

impl Display for Prompt {
    // By default we format for foundation/unknown models.
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Format::Unknown.format_prompt(self, None, f)
    }
}

/// A message in a chat transcript.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(all(test, feature = "webchat"), derive(rocket::UriDisplayQuery))]
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "webchat", derive(rocket::form::FromForm))]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
pub struct Message {
    pub role: Role,
    #[cfg_attr(feature = "serde", field(validate = len(..4096)))]
    pub text: String,
}

/// A [`Role`] is the participant's role in a chat transcript. This is similar
/// to the OpenAI API's role.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(all(test, feature = "webchat"), derive(rocket::UriDisplayQuery))]
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "webchat", derive(rocket::form::FromFormField))]
#[cfg_attr(
    feature = "serde",
    serde(crate = "rocket::serde"),
    serde(rename_all = "snake_case")
)]
pub enum Role {
    Human,
    Agent,
    /// Superuser role. This is some authority figure that constrains the
    /// Agent's behavior. It may be a system, a narrator, or a god.
    System,
}
