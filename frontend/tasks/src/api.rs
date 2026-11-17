//! What an HTTP front end asks of a loaded model.

use crate::Error;
use bundle::Message;

/// The served model's name, and the tokens one answer holds.
#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub context: usize,
}

/// Text to ids and back, as generation sees them.
pub trait Tokens: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// Messages to one prompt, in the model's chat layout.
pub trait Chat: Send + Sync {
    fn render(&self, messages: &[Message], think: bool) -> Result<String, Error>;
}

/// What one answer may do beyond the prompt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    pub max_tokens: usize,
    /// Strings that end the answer, left out of it.
    pub stop: Vec<String>,
}

/// Why an answer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

/// One step of an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// A token, and the text it lets out, maybe none.
    Token { id: u32, text: String },
    /// The last text, the reason, and the token count.
    Done {
        text: String,
        finish: Finish,
        tokens: usize,
    },
}

/// Takes each step. Returns false when it wants no more.
pub type Sink = Box<dyn FnMut(Step) -> bool + Send>;

/// Answers a prompt, a step at a time, into `sink`.
pub trait Generator: Send + Sync {
    fn generate(&self, prompt: Vec<u32>, params: Params, sink: Sink) -> Result<(), Error>;
}

/// Vectors for texts, and the tokens they took.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedded {
    pub vectors: Vec<Vec<f32>>,
    pub tokens: usize,
}

/// Texts to unit vectors. Calls can block.
pub trait Embed: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error>;
}
