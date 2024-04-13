// This example is from the Rocket chat example. It's been modified into a
// frontend for the regurgitater example.
//
// Many thanks to Bing's Copilot for helping me with this code. I'm not a
// frontend developer, so I'm not very good at this stuff. Many bugs were
// squashed with their help.

let generationDiv = document.getElementById("generation");
let newRequestForm = document.getElementById("request");
let statusDiv = document.getElementById("status");
let pieceTemplate = document.getElementById("piece");
let inputTextField = document.getElementById("request_text");
let chunkDropdown = document.getElementById("request_chunks");
let progressBar = document.getElementById("progress_bar");
let scoresTemplate = document.getElementById("scores");
let scorebox = document.getElementById("scorebox");

// State to store status.
let STATE = {
  status: "pending",
};

// Set the connection status. The status is a string corresponding to a CSS
// class name and the response kind.
function setStatus(status) {
  STATE.status = status;
  statusDiv.className = status;
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

// A function calculating color for a piece. 0.0 is green, 1.0 is red.
function scoreColor(score) {
  return `hsl(${score * 120}, 100%, 70%)`;
}

// Add a piece to the generation.
function addPiece(piece) {
  var spanClone = pieceTemplate.content.cloneNode(true);
  spanClone.querySelector(".piece").textContent = piece;
  generationDiv.appendChild(spanClone);
}

// Color the last piece with a score.
function colorLastPiece(percent) {
  var lastPiece = generationDiv.lastElementChild;
  lastPiece.style.backgroundColor = scoreColor(percent);
}

// Get the lastElementChild of the scorebox div. If there are no children, add
// a new scores div.
function getScores() {
  if (scorebox.children.length == 0) {
    let scores = scoresTemplate.content.cloneNode(true);
    scorebox.appendChild(scores);
  }

  return scorebox.lastElementChild;
}

// Clear everything.
function clear() {
  generationDiv.innerHTML = "";
  scorebox.innerHTML = "";
  progressBar.style.width = "0.0%";
  progressBar.style.textContent = "";
}

// Disable input fields.
function disableInput(disabled) {
  inputTextField.disabled = disabled;
  chunkDropdown.disabled = disabled;
}

// Subscribe to the event source at `uri` with exponential backoff reconnect.
function subscribe(uri) {
  var retryTime = 1;

  function connect(uri) {
    const events = new EventSource(uri);

    events.addEventListener("message", (ev) => {
      console.log("raw data", JSON.stringify(ev.data));
      console.log("decoded data", JSON.stringify(JSON.parse(ev.data)));
      const res = JSON.parse(ev.data);
      if (!("content" in res) || !("kind" in res)) return;

      switch (res.kind) {
        case "piece":
          addPiece(res.content);
          break;
        case "token_unigram_score":
          colorLastPiece(res.content);
          // id is `token_unigram_score`
          let tokenUnigramScore = getScores().querySelector(
            "#token_unigram_score"
          );
          tokenUnigramScore.textContent = res.content;
          break;
        case "token_bigram_score":
          let tokenBigramScore = getScores().querySelector(
            "#token_bigram_score"
          );
          tokenBigramScore.textContent = res.content;
          break;
        case "progress":
          progressBar.style.width = res.content;
          progressBar.textContent = res.content;
          break;
        case "percent_of_tokens":
          progressBar.style.width = "0.0%";
          progressBar.style.textContent = "";
          if (generationDiv.children.length != 0) {
            addPiece("\n\n\n");
          }
          // TODO: clean this up
          addPiece("percent of tokens: " + res.content + "\n\n\n");
          var lastPiece = generationDiv.lastElementChild;
          lastPiece.style.color = "var(--text-color)";
          let scores = scoresTemplate.content.cloneNode(true);
          let percentOfTokensScore = scores.querySelector(
            "#percent_of_tokens_score"
          );
          percentOfTokensScore.textContent = res.content;
          scorebox.appendChild(scores);
          break;
        case "character_score":
          let characterScore = getScores().querySelector("#character_score");
          characterScore.textContent = res.content;
          break;
        case "unigram_score":
          let unigramScore = getScores().querySelector("#unigram_score");
          unigramScore.textContent = res.content;
          break;
        case "bigram_score":
          let bigramScore = getScores().querySelector("#bigram_score");
          bigramScore.textContent = res.content;
          break;
        case "prefix":
          addPiece(res.content);
          var lastPiece = generationDiv.lastElementChild;
          lastPiece.style.backgroundColor = "blue";
          break;
        case "ready":
          disableInput(false);
          setStatus(res.kind);
          console.log(res.content);
          break;
        case "error":
        case "fatal":
        case "busy":
        case "shutdown":
          disableInput(true);
          setStatus(res.kind);
          console.log(res.content);
          break;
        default:
          console.error(
            `unknown response kind: ${res.kind} with content: ${res.content}`
          );
      }
    });

    events.addEventListener("open", () => {
      setStatus("connected");
      // TODO: On reconnect we should check the status, but our API is very
      // simple for this example code and doesn't support this yet, nor is
      // authentication implemented. We don't even have sessions.
      clear();
      console.log(`connected to event stream at ${uri}`);
      retryTime = 1;
    });

    events.addEventListener("error", () => {
      setStatus("disconnected");
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
  newRequestForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const text = inputTextField.value;
    const mode = "jaccard";
    const chunks = chunkDropdown.value;

    if (STATE.status === "connected" || STATE.status === "ready") {
      fetch("/request", {
        method: "POST",
        body: new URLSearchParams({ mode, text, chunks }),
      }).then((response) => {
        if (response.ok) inputTextField.value = "";
      });
    }
  });

  // Subscribe to server-sent events.
  subscribe("/events");
}

init();
