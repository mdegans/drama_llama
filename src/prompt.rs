use std::fmt::{Display, Formatter};

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
    #[cfg_attr(feature = "serde", field(validate = len(..64), default = "Assistant"))]
    pub agent: String,
    /// Human's name, e.g. "Alice" or "Bob".
    #[cfg_attr(feature = "serde", field(validate = len(..64), default = "Human"))]
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
}

impl Display for Prompt {
    // TODO: Right now this formats foundation models. For tuned models, the
    // prompt should be formatted differently.
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if let Some(setting) = &self.setting {
            write!(f, "# Context\n{}\n\n", setting)?;
        }

        write!(f, "# Transcript\n")?;
        for message in &self.transcript {
            write!(
                f,
                "{}: {}\n",
                match message.role {
                    Role::Human => &self.human,
                    Role::Agent => &self.agent,
                    Role::System => self.system.as_deref().unwrap_or("System"),
                },
                message.text
            )?;
        }

        Ok(())
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
