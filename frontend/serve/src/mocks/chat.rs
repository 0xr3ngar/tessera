//! The chat layout for `/v1/chat/completions`.

use tessera_bundle::Message;
use tessera_tasks::Error;
use tessera_tasks::api::Chat;

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
