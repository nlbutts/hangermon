const presenceEl = document.getElementById("presence");
const confidenceEl = document.getElementById("confidence");
const fpsEl = document.getElementById("fps");
const updatedEl = document.getElementById("updated");
const recordingStateEl = document.getElementById("recording-state");
const temperatureEl = document.getElementById("temperature");
const humidityEl = document.getElementById("humidity");
const ledSlider = document.getElementById("led-slider");
const ledValueEl = document.getElementById("led-value");
const clipBody = document.getElementById("clip-body");

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const payload = await response.json();
    presenceEl.textContent = payload.human_present ? "Human detected" : "Clear";
    presenceEl.className = payload.human_present ? "badge badge-active" : "badge badge-idle";
    confidenceEl.textContent = payload.confidence?.toFixed?.(2) ?? payload.confidence;
    fpsEl.textContent = payload.fps;
    updatedEl.textContent = payload.last_updated ? new Date(payload.last_updated * 1000).toLocaleTimeString() : "--";

    const state = payload.recording_state || "standby";
    recordingStateEl.textContent = state.toUpperCase();
    recordingStateEl.className = "state-badge " + state;

    // Sense HAT sensors
    if (payload.temperature_f !== undefined) {
      temperatureEl.textContent = payload.temperature_f.toFixed(1) + "°F / " + payload.temperature_c.toFixed(1) + "°C";
    }
    if (payload.humidity !== undefined) {
      humidityEl.textContent = payload.humidity.toFixed(1);
    }
    if (payload.led_intensity !== undefined && !ledSlider._userDragging) {
      ledSlider.value = payload.led_intensity;
      ledValueEl.textContent = payload.led_intensity;
    }
  } catch (err) {
    console.error("status error", err);
  }
}

// LED slider interaction
let ledDebounce = null;

ledSlider.addEventListener("input", () => {
  ledSlider._userDragging = true;
  ledValueEl.textContent = ledSlider.value;
  clearTimeout(ledDebounce);
  ledDebounce = setTimeout(() => {
    fetch("/api/led", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intensity: parseInt(ledSlider.value) }),
    }).finally(() => {
      ledSlider._userDragging = false;
    });
  }, 100);
});

async function refreshClips() {
  try {
    const response = await fetch("/api/clips");
    if (!response.ok) return;
    const data = await response.json();
    clipBody.innerHTML = "";
    (data.clips || []).forEach((clip) => {
      const row = document.createElement("tr");
      const ts = document.createElement("td");
      ts.textContent = new Date(clip.timestamp).toLocaleString();
      const duration = document.createElement("td");
      duration.textContent = clip.duration.toFixed(1);
      const conf = document.createElement("td");
      conf.textContent = clip.confidence.toFixed(2);
      const linkCell = document.createElement("td");
      const link = document.createElement("a");
      link.textContent = "Download";
      link.className = "btn btn-download";
      link.href = `/api/clips/${clip.relative_path ?? clip.filename}`;
      link.setAttribute("download", clip.filename);
      linkCell.appendChild(link);
      row.appendChild(ts);
      row.appendChild(duration);
      row.appendChild(conf);
      row.appendChild(linkCell);
      clipBody.appendChild(row);
    });
  } catch (err) {
    console.error("clip error", err);
  }
}

refreshStatus();
refreshClips();
setInterval(refreshStatus, 2000);
setInterval(refreshClips, 10000);
