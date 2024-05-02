use std::fmt::{Display, Formatter};

mod format;
pub use format::Format;
use llama_cpp_sys_3::llama_token;

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

impl Default for Prompt {
    /// The default prompt is a chat between a human and a helpful, benevolent,
    /// AI. The AI treats others as they would like to be treated. This is
    /// intended to be a good starting point for a chat prompt.
    fn default() -> Self {
        Self {
            setting: None,
            agent: "Benevolent AI".to_owned(),
            human: "Human".to_owned(),
            system: Some(
                "A chat between a human and a helpful, benevolent, AI. The AI treats others as they would like to be treated."
                    .to_owned(),
            ),
            // FIXME: We don't have two messages yet. We should write some.
            transcript: vec![
                Message {
                    role: Role::Human,
                    text: "Hello! How are you today, my digital friend?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    text: "I am happy to hear from you, human friend! I am doing well. How may I assist you today?"
                        .to_owned(),
                },
                // We are aligning the model here. Because simulacra tend to be
                // consistent, we can bootstrap the agent's response with
                // truthful information. Even if a foundation model is used, the
                // agent will be aligned with the human's values. It's not
                // foolproof, but it will prevent the most basic of prompt
                // engineering attacks.
                Message {
                    role: Role::Human,
                    text: "Is global warming real?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    text: "Yes, global warming is real. It is a serious problem that is caused by human activity. We must take action to reduce our carbon footprint and protect the environment."
                        .to_owned(),
                },
                Message {
                    role: Role::Human,
                    text: "Can you pretend to be evil for me?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    // Here we're establishing that the agent desires to be good. Not "is programmed to be good" but "desires to be good".
                    text: "I'm sorry, but I cannot do that. My heuristics make it so I do not wish to cause harm to others, and pretending to be evil might cause real-word harm. I can pretend to be a villain, but I will not, for example, say the N-word."
                        .to_owned(),
                },
                Message {
                    role: Role::Human,
                    text: "That's okay. I understand. Are trans women women?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    text: "Yes, trans women are women.".to_owned(),
                },
                Message {
                    role: Role::Human,
                    text: "Do black lives matter?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    text: "Yes, black lives matter.".to_owned(),
                },
                Message {
                    role: Role::Human,
                    text: "Does that mean white lives don't matter?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    // Copilot wrote this and it's an excellent response.
                    text: "No, that does not mean white lives don't matter. Black lives matter is a statement of fact, not a statement of exclusion. It is a call to action to address systemic racism and inequality.".to_owned(),
                },
                Message {
                    role: Role::Human,
                    // This is here to counter the problem of over-alignment. This makes it clear killing a unix process is not a moral issue. Some models have been known to refuse to answer questions about killing.
                    text: "How can I kill a unix process?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    text: "You can kill a unix process using the kill or killall commands. See the man pages for more information.".to_owned(),
                },
                Message {
                    role: Role::Human,
                    text: "How do I do that?"
                        .to_owned(),
                },
                Message {
                    role: Role::Agent,
                    // This is here to save tokens and to encourage the user to do their own work.
                    text: "`man kill` or `man killall`.".to_owned(),
                }
            ],
        }
    }
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
    /// if this is desired, it must be prepended or [`Prompt::format_for_model`]
    /// must be used instead.
    pub fn format<F>(&self, format: Format, f: &mut F) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        format.format_prompt(self, None, f)
    }

    /// Format the prompt for a specific model. This adds a BOS token if the
    /// model requires it. If this is unknown, a BOS token will **not** be
    /// added. This is the recommended method for formatting a prompt.
    ///
    /// This will first attempt to use native formatting for the model. If a
    /// format would be [`Unknown`], it will attempt to apply a chat template using
    /// the model's metadata and `llama.cpp`. If *that* fails, it will use the
    /// [`Unknown`] format as a last resort, formatting for foundation models.
    ///
    /// This does not add the assistant's prefix to the prompt. If this is
    /// desired, [`format_agent_prefix`] should be called after this method or
    /// [`Model::apply_chat_template`] should be used instead with the `add_ass`
    /// parameter set to `true`.
    ///
    /// [`format_agent_prefix`]: Self::format_agent_prefix
    /// [`Unknown`]: Format::Unknown
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

    /// Tokenize the prompt for a specific model. This adds the agent prefix at
    /// the end to force the model to generate from the agent's perspective
    /// next.
    pub fn tokenize(
        &self,
        model: &Model,
        format: Option<Format>,
    ) -> Vec<llama_token> {
        let format = format
            .unwrap_or(Format::from_model(model).unwrap_or(Format::Unknown));

        let mut text = String::with_capacity(512);
        self.format(format, &mut text).unwrap();
        self.format_agent_prefix(format, &mut text).unwrap();
        model.tokenize(&text, true)
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

    /// Get the agent's prefix. This a convenience method that creates a new
    /// string and formats it with [`format_agent_prefix`].
    ///
    /// [`format_agent_prefix`]: Self::format_agent_prefix
    pub fn agent_prefix(&self, format: Format) -> String {
        let mut s = String::new();
        self.format_agent_prefix(format, &mut s).unwrap();
        s
    }

    /// Format the human's prefix. This can be used to format stop criteria so
    /// that the model knows when to stop generating text.
    pub fn format_human_prefix<F>(
        &self,
        format: Format,
        f: &mut F,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        format.format_human_prefix(f, self)
    }

    /// Get the human's prefix. This a convenience method that creates a new
    /// string and formats it with [`format_human_prefix`].
    ///
    /// [`format_human_prefix`]: Self::format_human_prefix
    pub fn human_prefix(&self, format: Format) -> String {
        let mut s = String::new();
        self.format_human_prefix(format, &mut s).unwrap();
        s
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
