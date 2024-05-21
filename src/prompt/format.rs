use llama_cpp_sys_3::llama_token;

use crate::{Message, Model, Prompt, Role};

pub mod llama2 {
    pub const BOS: &str = "<s>";
    pub const EOS: &str = "</s>";
    pub const B_INST: &str = "[INST]";
    pub const E_INST: &str = "[/INST]";
    pub const B_SYS: &str = "<<SYS>>";
    pub const E_SYS: &str = "<</SYS>>";

    /// These cannot be in the transcript or prompt. They're reserved to prevent
    /// prompt injection, or at least make it more difficult.
    pub const SPECIAL_TAGS: &[&str] = &[BOS, EOS, B_INST, E_INST, B_SYS, E_SYS];
}

pub mod llama3 {
    pub const BOS: &str = "<|begin_of_text|>";
    pub const EOS: &str = "<|end_of_text|>";
    pub const B_HEADER: &str = "<|start_header_id|>";
    pub const E_HEADER: &str = "<|end_header_id|>";
    pub const EOT: &str = "<|eot_id|>";

    pub const SPECIAL_TAGS: &[&str] = &[BOS, EOS, B_HEADER, E_HEADER, EOT];
}

/// [`Format`] for a [`Prompt`].
#[derive(Debug, Clone, Copy)]
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
pub enum Format {
    /// Unknown model. We will attempt to get the format from the model.
    Unknown,
    /// LLaMA foundation model.
    LLaMA,
    /// LLaMA 2 foundation model.
    LLaMA2,
    /// LLaMA 2 chat model.
    LLaMA2Chat,
    /// LLaMA 3 foundation model.
    LLaMA3,
    /// LLaMA 3 chat model.
    LLaMA3Chat,
    /// Vicuna format. This assumes Vicuna 1.1 or later.
    Vicuna,
}

impl Format {
    pub fn from_model(model: &Model) -> Option<Self> {
        let found =
            match model.get_meta("general.name").as_ref().map(|s| s.as_str()) {
                Some("LLaMA") => Some(Self::LLaMA),
                Some("LLaMA v2") => Some(Self::LLaMA2),
                Some("LLaMA v2 Chat") => Some(Self::LLaMA2Chat),
                // The current ggml conversion of llama3 has ".." as the model name,
                // but that is likely an error. I am testing with this model:
                // https://huggingface.co/NousResearch/Meta-Llama-3-70B-GGUF As soon
                // as a better conversion is available, I will update this. Other
                // heuristics may be needed to determine the model type.
                Some("LLaMA v3") => Some(Self::LLaMA3),
                Some("LLaMA v3 Chat") => Some(Self::LLaMA3Chat),
                // TODO: download Vicuna and check the model name, also check the
                // chat models above.
                _ => None,
            };

        if found.is_some() {
            return found;
        }

        // We'll use some heuristics to determine the model type. TODO: this
        // is a bit fragile and may need to be updated.
        let vocab_size =
            model.get_meta("llama.vocab_size").unwrap_or("0".to_owned());
        let feed_forward_len = model
            .get_meta("llama.feed_forward_length")
            .unwrap_or("0".to_owned());
        let head_count = model
            .get_meta("llama.attention.head_count_kv")
            .unwrap_or("0".to_owned());
        let bos = model.bos();
        let eos = model.eos();

        match (
            vocab_size.as_str(),
            feed_forward_len.as_str(),
            head_count.as_str(),
            bos,
            eos,
        ) {
            ("128256", "28672", "8", 128000, 128001) => Some(Self::LLaMA3),
            _ => None,
        }
    }

    pub const fn stop_tokens(&self) -> Option<&'static [llama_token]> {
        match self {
            Format::LLaMA
            | Format::LLaMA2
            | Format::LLaMA2Chat
            | Format::Vicuna => Some(&[2]),
            Format::LLaMA3 => Some(&[128001]),
            Format::LLaMA3Chat => Some(&[128001, 128009]),
            Format::Unknown => None,
        }
    }

    pub fn format_setting<F>(
        &self,
        f: &mut F,
        prompt: &Prompt,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        if let Some(setting) = prompt.setting.as_ref() {
            match self {
                // Foundation model or unknown. We use `# Context` to denote the
                // setting.
                Format::Unknown
                | Format::LLaMA
                | Format::LLaMA2
                | Format::LLaMA3 => {
                    write!(f, "# Context\n{}\n\n", setting)?;
                }
                // Chat model. We're using the system instructions to denote the
                // setting. This is not following the LLaMA 3 format, but it
                // doesn't strictly need to.
                Format::LLaMA3Chat => {
                    assert!(
                        !llama3::SPECIAL_TAGS
                            .iter()
                            .any(|tag| setting.contains(tag)),
                        "Setting contains forbidden special token."
                    );
                    write!(
                        f,
                        "{}{}{}\n\n{}{}",
                        llama3::B_HEADER,
                        "system",
                        llama3::E_HEADER,
                        &setting,
                        llama3::EOT
                    )?;
                }
                // Logic is borrowed from:
                // https://github.com/meta-llama/llama/blob/main/llama/generation.py
                Format::LLaMA2Chat => {
                    assert!(
                        !llama2::SPECIAL_TAGS
                            .iter()
                            .any(|tag| setting.contains(tag)),
                        "Setting contains forbidden special token."
                    );
                    write!(
                        f,
                        "{} {}{}{} {}",
                        llama2::B_INST,
                        llama2::B_SYS,
                        setting,
                        llama2::E_INST,
                        llama2::E_SYS
                    )?;
                }
                Format::Vicuna => todo!(),
            }
        }

        Ok(())
    }

    pub fn format_transcript<F>(
        &self,
        f: &mut F,
        prompt: &Prompt,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        if matches!(
            self,
            Format::Unknown | Format::LLaMA | Format::LLaMA2 | Format::LLaMA3
        ) {
            f.write_str("# Transcript")?;
        }

        for Message { role, text } in prompt.transcript.iter() {
            match self {
                Format::Unknown
                | Format::LLaMA
                | Format::LLaMA2
                | Format::LLaMA3 => match role {
                    Role::Human => write!(f, "\n{}: {}", &prompt.human, text)?,
                    Role::Agent => write!(f, "\n{}: {}", &prompt.agent, text)?,
                    Role::System => write!(
                        f,
                        "\n{}: {}",
                        prompt.system.as_deref().unwrap_or("Narrator"),
                        text
                    )?,
                },
                Format::LLaMA2Chat => match role {
                    crate::Role::Human => write!(
                        f,
                        "\n{} {}: {} {}",
                        llama2::B_INST,
                        prompt.human,
                        text,
                        llama2::E_INST
                    )?,
                    crate::Role::Agent => {
                        write!(f, "\n{}: {}", prompt.agent, text)?
                    }
                    crate::Role::System => write!(
                        f,
                        "\n{} {}{}: {}{} {}",
                        llama2::B_INST,
                        llama2::B_SYS,
                        prompt.system.as_deref().unwrap_or("narrator"),
                        text,
                        llama2::E_SYS,
                        llama2::E_INST
                    )?,
                },
                Format::LLaMA3Chat => {
                    f.write_str(llama3::B_HEADER)?;
                    match role {
                        Role::Human => f.write_str(&prompt.human)?,
                        Role::Agent => f.write_str(&prompt.agent)?,
                        Role::System => f.write_str(
                            // TODO: experiment with narrator and system to see
                            // which better constrains the agent's behavior and
                            // this may be per-agent, which is why it's a good
                            // thing it's configurable in the `Prompt`.
                            &prompt.system.as_deref().unwrap_or("narrator"),
                        )?,
                    }
                    f.write_str(llama3::E_HEADER)?;
                    f.write_str("\n\n")?;
                    f.write_str(text)?;
                    f.write_str(llama3::EOT)?;
                }
                Format::Vicuna => todo!(),
            }
        }

        Ok(())
    }

    /// Format the agent's partial response. This is enough to force the agent
    /// to respond next.
    pub fn format_agent_prefix<F>(
        &self,
        f: &mut F,
        prompt: &Prompt,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        match self {
            Format::Unknown
            | Format::LLaMA
            | Format::LLaMA2
            | Format::LLaMA3 => {
                f.write_str("\n")?;
                f.write_str(&prompt.agent)?;
                f.write_str(":")?;
            }
            Format::LLaMA2Chat => {
                f.write_str("\n")?;
                f.write_str(&prompt.agent)?;
                f.write_str(":")?;
            }
            Format::LLaMA3Chat => {
                f.write_str(llama3::B_HEADER)?;
                f.write_str(&prompt.agent)?;
                f.write_str(llama3::E_HEADER)?;
                f.write_str("\n\n")?;
            }
            Format::Vicuna => todo!(),
        }

        Ok(())
    }

    /// Format the human's partial response. This can be used for stop criteria.
    pub fn format_human_prefix<F>(
        &self,
        f: &mut F,
        prompt: &Prompt,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        match self {
            Format::Unknown
            | Format::LLaMA
            | Format::LLaMA2
            | Format::LLaMA3 => {
                f.write_str("\n")?;
                f.write_str(&prompt.human)?;
                f.write_str(":")?;
            }
            Format::LLaMA2Chat => {
                f.write_str("\n")?;
                f.write_str(&prompt.human)?;
                f.write_str(":")?;
            }
            Format::LLaMA3Chat => {
                f.write_str(llama3::B_HEADER)?;
                f.write_str(&prompt.human)?;
                f.write_str(llama3::E_HEADER)?;
                f.write_str("\n\n")?;
            }
            Format::Vicuna => todo!(),
        }

        Ok(())
    }

    pub fn format_prompt<F>(
        &self,
        prompt: &Prompt,
        model: Option<&Model>,
        f: &mut F,
    ) -> std::fmt::Result
    where
        F: std::fmt::Write,
    {
        if let Some(model) = model {
            if model.add_bos().unwrap_or(Model::DEFAULT_ADD_BOS) {
                f.write_str(&model.token_to_piece(model.bos()))?;
            }
        }

        self.format_setting(f, prompt)?;
        self.format_transcript(f, prompt)?;

        Ok(())
    }

    pub const fn special_tags(&self) -> &'static [&'static str] {
        match self {
            Format::Unknown => &[],
            Format::LLaMA => &[],
            Format::LLaMA2 => llama2::SPECIAL_TAGS,
            Format::LLaMA2Chat => llama2::SPECIAL_TAGS,
            Format::LLaMA3 => llama3::SPECIAL_TAGS,
            Format::LLaMA3Chat => llama3::SPECIAL_TAGS,
            Format::Vicuna => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_prompt() {
        let prompt = Prompt {
            setting: Some("A random setting.".to_string()),
            agent: "Assistant".to_string(),
            human: "Human".to_string(),
            system: Some("System".to_string()),
            transcript: vec![
                Message {
                    role: Role::Human,
                    text: "Hello.".to_string(),
                },
                Message {
                    role: Role::Agent,
                    text: "Hi.".to_string(),
                },
                Message {
                    role: Role::System,
                    text: "How are you?".to_string(),
                },
            ],
        };

        let mut s = prompt.to_string();
        assert_eq!(
            &s,
            "# Context\nA random setting.\n\n# Transcript\nHuman: Hello.\nAssistant: Hi.\nSystem: How are you?",
        );

        prompt.format_agent_prefix(Format::Unknown, &mut s).unwrap();
        assert_eq!(
            &s,
            "# Context\nA random setting.\n\n# Transcript\nHuman: Hello.\nAssistant: Hi.\nSystem: How are you?\nAssistant:",
        );

        let mut s = String::new();
        prompt.format(Format::LLaMA2Chat, &mut s).unwrap();
        assert_eq!(
            &s,
            "[INST] <<SYS>>A random setting.[/INST] <</SYS>>\n[INST] Human: Hello. [/INST]\nAssistant: Hi.\n[INST] <<SYS>>System: How are you?<</SYS>> [/INST]",
        );

        prompt
            .format_agent_prefix(Format::LLaMA2Chat, &mut s)
            .unwrap();

        assert_eq!(
            &s,
            "[INST] <<SYS>>A random setting.[/INST] <</SYS>>\n[INST] Human: Hello. [/INST]\nAssistant: Hi.\n[INST] <<SYS>>System: How are you?<</SYS>> [/INST]\nAssistant:",
        );

        s.clear();
        prompt.format(Format::LLaMA3Chat, &mut s).unwrap();

        assert_eq!(
            s,
            "<|start_header_id|>system<|end_header_id|>\n\nA random setting.<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\nHello.<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\nHi.<|eot_id|><|start_header_id|>System<|end_header_id|>\n\nHow are you?<|eot_id|>",
        );

        prompt
            .format_agent_prefix(Format::LLaMA3Chat, &mut s)
            .unwrap();

        assert_eq!(
            s,
            "<|start_header_id|>system<|end_header_id|>\n\nA random setting.<|eot_id|><|start_header_id|>Human<|end_header_id|>\n\nHello.<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\nHi.<|eot_id|><|start_header_id|>System<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>Assistant<|end_header_id|>\n\n",
        );
    }
}
