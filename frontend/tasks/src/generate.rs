//! Prompts to requests, with the model's template and stops.

use crate::Error;
use bundle::tokenize::hf::Hf;
use bundle::{Message, Template};
use engine::Request;

pub struct Generate {
    pub tokenizer: Hf,
    pub template: Option<Template>,
    /// Ids that end an answer.
    pub stop: Vec<u32>,
}

impl Generate {
    /// `prompt` as a request for `max_tokens` more.
    pub fn request(&self, prompt: &str, max_tokens: usize) -> Result<Request, Error> {
        let tokens = self.tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            return Err("an empty prompt".into());
        }
        Ok(Request {
            tokens,
            max_tokens,
            stop: self.stop.clone(),
        })
    }

    /// The chat as the template lays it out.
    pub fn chat(&self, messages: &[Message], think: bool) -> Result<String, Error> {
        let template = self
            .template
            .as_ref()
            .ok_or("the model has no chat template")?;
        if messages.is_empty() {
            return Err("no messages".into());
        }
        Ok(template.render(messages, think)?)
    }
}
