// This example is from the Rocket chat example. It's been modified to remove
// room functionality and to remove the username, both of which aren't needed
// for the Charlie chat example.

let messagesDiv = document.getElementById("messages");
let newMessageForm = document.getElementById("new-message");
let statusDiv = document.getElementById("status");

let messageTemplate = document.getElementById("message");
let messageField = newMessageForm.querySelector("#message");

var STATE = {
  history: [],
  connected: false,
};

// Set the connection status: `true` for connected, `false` for disconnected.
function setConnectedStatus(status) {
  STATE.connected = status;
  statusDiv.className = status ? "connected" : "reconnecting";
}

// Generate a color from a "hash" of a string. Thanks, internet.
function hashColor(str) {
  let hash = 0;
  for (var i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
    hash = hash & hash;
  }

  return `hsl(${hash % 360}, 100%, 70%)`;
}

// Add `message` from `role` to `history`. If `push`, then actually store the
// message. Finally, render the message.
function addMessage(role, text, push = false) {
  if (push) {
    STATE.history.push({ role, text });
  }

  var node = messageTemplate.content.cloneNode(true);
  node.querySelector(".message .role").textContent = role;
  node.querySelector(".message .role").style.color = hashColor(role);
  node.querySelector(".message .text").textContent = text;
  messagesDiv.appendChild(node);
}

// Subscribe to the event source at `uri` with exponential backoff reconnect.
function subscribe(uri) {
  var retryTime = 1;

  function connect(uri) {
    const events = new EventSource(uri);

    events.addEventListener("message", (ev) => {
      console.log("raw data", JSON.stringify(ev.data));
      console.log("decoded data", JSON.stringify(JSON.parse(ev.data)));
      const msg = JSON.parse(ev.data);
      if (!("text" in msg) || !("role" in msg)) return;
      addMessage(msg.role, msg.text, true);
    });

    events.addEventListener("open", () => {
      setConnectedStatus(true);
      console.log(`connected to event stream at ${uri}`);
      retryTime = 1;
    });

    events.addEventListener("error", () => {
      setConnectedStatus(false);
      events.close();

      let timeout = retryTime;
      retryTime = Math.min(64, retryTime * 2);
      console.log(`connection lost. attempting to reconnect in ${timeout}s`);
      setTimeout(() => connect(uri), (() => timeout * 1000)());
    });
  }

  connect(uri);
}

// Let's go! Initialize the world.
function init() {
  // Set up the form handler.
  newMessageForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const text = messageField.value;
    const role = "Human";
    if (!text || !role) return;

    if (STATE.connected) {
      fetch("/message", {
        method: "POST",
        body: new URLSearchParams({ role, text }),
      }).then((response) => {
        if (response.ok) messageField.value = "";
      });
    }
  });

  // Subscribe to server-sent events.
  subscribe("/events");
}

init();
