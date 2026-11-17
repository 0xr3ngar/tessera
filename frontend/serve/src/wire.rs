//! OpenAI's response bodies: completions, chat and the model list.

use serde::Serialize;

/// Which route an answer is for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Chat,
}

/// The fields every body of one answer repeats.
#[derive(Clone)]
pub struct Head {
    pub kind: Kind,
    pub id: String,
    pub created: u64,
    pub model: String,
}

#[derive(Serialize)]
pub struct Body {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: [Choice; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<Said>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<Said>,
    finish_reason: Option<&'static str>,
}

/// A chat message, or a streamed part of one.
#[derive(Serialize)]
struct Said {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
pub struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

impl Head {
    fn body(&self, object: &'static str, choice: Choice, usage: Option<Usage>) -> Body {
        Body {
            id: self.id.clone(),
            object,
            created: self.created,
            model: self.model.clone(),
            choices: [choice],
            usage,
        }
    }

    /// The whole answer in one body.
    pub fn whole(&self, text: String, finish: &'static str, usage: Usage) -> Body {
        let mut choice = Choice::new(Some(finish));
        let object = match self.kind {
            Kind::Text => {
                choice.text = Some(text);
                "text_completion"
            }
            Kind::Chat => {
                choice.message = Some(Said::assistant(text));
                "chat.completion"
            }
        };
        self.body(object, choice, Some(usage))
    }

    /// One streamed piece. The last has the reason, no text.
    pub fn chunk(&self, text: String, finish: Option<&'static str>) -> Body {
        let mut choice = Choice::new(finish);
        let object = match self.kind {
            Kind::Text => {
                choice.text = Some(text);
                "text_completion"
            }
            Kind::Chat => {
                let content = (finish.is_none()).then_some(text);
                choice.delta = Some(Said {
                    role: None,
                    content,
                });
                "chat.completion.chunk"
            }
        };
        self.body(object, choice, None)
    }

    /// The first chat chunk, which names the role.
    pub fn role(&self) -> Body {
        let mut choice = Choice::new(None);
        choice.delta = Some(Said::assistant(String::new()));
        self.body("chat.completion.chunk", choice, None)
    }
}

impl Choice {
    fn new(finish: Option<&'static str>) -> Self {
        Self {
            index: 0,
            text: None,
            message: None,
            delta: None,
            finish_reason: finish,
        }
    }
}

impl Said {
    fn assistant(content: String) -> Self {
        Self {
            role: Some("assistant"),
            content: Some(content),
        }
    }
}

#[derive(Serialize)]
pub struct Models {
    object: &'static str,
    data: Vec<Model>,
}

#[derive(Serialize)]
struct Model {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

impl Models {
    /// The one model this server holds.
    pub fn one(id: String, created: u64) -> Self {
        Self {
            object: "list",
            data: vec![Model {
                id,
                object: "model",
                created,
                owned_by: "tessera",
            }],
        }
    }
}

/// A refusal, as OpenAI's clients read it.
#[derive(Serialize)]
pub struct Failed {
    error: Detail,
}

#[derive(Serialize)]
struct Detail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    code: u16,
}

impl Failed {
    pub fn new(code: u16, message: String) -> Self {
        let kind = match code {
            400..500 => "invalid_request_error",
            _ => "server_error",
        };
        Self {
            error: Detail {
                message,
                kind,
                code,
            },
        }
    }
}
