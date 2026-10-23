export const PROMPTS = {
  qa: "Q and A",
  completion: "Text Completion",
  agent_qa: "Agent Q and A",
}

export const serverURL = (
  process.env.REACT_APP_API_URL || "http://localhost:8080"
).replace(/\/$/, "")
export const wsURL =
  process.env.REACT_APP_WS_URL || serverURL.replace(/^http/, "ws") + "/chat"
