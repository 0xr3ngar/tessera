//! The chat layout for `/v1/chat/completions`.

use sensai_bundle::Message;
use sensai_tasks::Error;
use sensai_tasks::api::Chat;

/// Each message as `role:content`, joined.
pub(super) struct MockChat;

impl Chat for MockChat {
    fn render(&self, messages: &[Message], _think: bool) -> Result<String, Error> {
        Ok(messages
            .iter()
            .map(|m| format!("{}:{}", m.role, m.content))
            .collect())
    }
}
