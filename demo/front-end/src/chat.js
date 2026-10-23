import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
} from "@chatscope/chat-ui-kit-react";
import { useContext } from "react";
import { MessageContext } from "./data/message-context";

export function Chat() {
  const { messages, onSend } = useContext(MessageContext);
  return (
    <MainContainer>
      <ChatContainer>
        <MessageList>
          {messages.map((message, index) => (
            <Message key={index} model={message} />
          ))}
        </MessageList>
        <MessageInput
          autoFocus
          placeholder="Type message here"
          onSend={onSend}
          attachButton={false}
        />
      </ChatContainer>
    </MainContainer>
  );
}
