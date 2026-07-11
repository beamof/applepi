use crate::llm::Message;

/// 短期记忆：维护 system + 会话消息。
pub struct History {
    msgs: Vec<Message>,
}

impl History {
    pub fn new(system: String) -> Self {
        Self { msgs: vec![Message::system(system)] }
    }
    pub fn add(&mut self, m: Message) {
        self.msgs.push(m);
    }
    pub fn all(&self) -> &[Message] {
        &self.msgs
    }
}
