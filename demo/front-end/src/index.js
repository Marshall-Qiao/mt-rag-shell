import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import { MessageProvider } from "./data/message-context";
import { Chat } from "./chat";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <div style={{ position: "relative", height: "100vh" }}>
    <MessageProvider>
      <Chat />
    </MessageProvider>
  </div>
);
