import { createContext, useEffect, useState } from "react";
import { connectBackend } from "../actions/connect-backend";
import { wsURL } from "../constants";
import { parse } from "marked";
import { sanitize } from "dompurify";

export const MessageContext = createContext({});

const ws = new WebSocket(wsURL);

export const MessageProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);

  const onReceive = (messageMd) => {
    const message = sanitize(parse(messageMd));
    setMessages((prevMessages) => [
      ...prevMessages,
      { message, direction: "incoming" },
    ]);
  };

  const onSend = (_, message) => {
    ws.send(message);
    setMessages((prevMessages) => [
      ...prevMessages,
      { message, direction: "outgoing" },
    ]);
  };

  useEffect(() => {
    connectBackend(onReceive);

    ws.onerror = function () {
      onReceive("Failed to connect to websocket at: " + wsURL);
    };

    ws.onmessage = function (event) {
      onReceive(event.data);
    };
  }, []);

  return (
    <MessageContext.Provider
      value={{ messages, onSend, onReceive }}
    >
      {children}
    </MessageContext.Provider>
  );
};
