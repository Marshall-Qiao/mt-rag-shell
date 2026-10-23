import { serverURL } from "../constants";
const url = serverURL + "/connect";

export async function connectBackend(onReceive) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    onReceive(data.response);
  } catch (error) {
    console.error("There was a problem with the fetch operation:", error);
    onReceive("Failed to connect to sever at: " + url);
  }
}
