const presenceEl = document.getElementById("presence");
const confidenceEl = document.getElementById("confidence");
const fpsEl = document.getElementById("fps");
const recordingStateEl = document.getElementById("recording-state");
const recordingOverlay = document.getElementById("recording-overlay");
const temperatureEl = document.getElementById("temperature");
const cpuTempEl = document.getElementById("cpu-temp");
const humidityEl = document.getElementById("humidity");
const ledSlider = document.getElementById("led-slider");
const ledValueEl = document.getElementById("led-value");
const clipListEl = document.getElementById("clip-list");

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + 
         ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const payload = await response.json();

    // Detections
    if (payload.human_present) {
      presenceEl.textContent = "HUMAN";
      presenceEl.className = "badge-active value";
    } else {
      presenceEl.textContent = "CLEAR";
      presenceEl.className = "badge-idle value";
    }


    confidenceEl.textContent = payload.confidence?.toFixed(2) ?? "0.00";
    fpsEl.textContent = payload.fps?.toFixed(1) ?? "0.0";

    // State
    const state = payload.recording_state || "monitoring";
    recordingStateEl.textContent = state.toUpperCase();
    recordingStateEl.className = "value state-badge " + state;

    if (state === "saving") {
      recordingOverlay.classList.remove("hidden");
    } else {
      recordingOverlay.classList.add("hidden");
    }

    // Sensors
    if (payload.temperature_f !== undefined) {
      temperatureEl.textContent = `${payload.temperature_f.toFixed(1)}°F`;
    }
    if (payload.cpu_temp !== undefined) {
      cpuTempEl.textContent = `${payload.cpu_temp.toFixed(1)}°C`;
    }
    if (payload.humidity !== undefined) {
      humidityEl.textContent = payload.humidity.toFixed(1) + "%";
    }


    // Controls
    if (payload.led_intensity !== undefined && !ledSlider._userDragging) {
      ledSlider.value = payload.led_intensity;
      ledValueEl.textContent = payload.led_intensity;
    }
  } catch (err) {
    console.error("Status fetch error", err);
  }
}

async function refreshClips() {
  try {
    const response = await fetch("/api/clips");
    if (!response.ok) return;
    const data = await response.json();
    const clips = data.clips || [];

    // Clear and rebuild
    clipListEl.innerHTML = "";

    clips.forEach(clip => {
      const card = document.createElement("a");
      card.href = `/api/clips/${clip.relative_path}`;
      card.className = "clip-card";
      card.setAttribute("download", clip.filename);

      const thumbUrl = clip.thumbnail_path 
        ? `/api/clips/${clip.thumbnail_path}` 
        : "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 9'%3E%3Crect width='16' height='9' fill='%23111'/%3E%3C/svg%3E";

      card.innerHTML = `
        <div class="clip-thumb">
          <img src="${thumbUrl}" alt="Thumbnail" loading="lazy" />
        </div>
        <div class="clip-info">
          <span class="clip-time">${formatTime(clip.timestamp)}</span>
          <div class="clip-meta">
            <span>⌛ ${clip.duration.toFixed(1)}s</span>
            <span>🎯 ${clip.confidence.toFixed(2)}</span>
          </div>
        </div>
      `;
      clipListEl.appendChild(card);
    });

  } catch (err) {
    console.error("Clips fetch error", err);
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

// Init
refreshStatus();
refreshClips();
setInterval(refreshStatus, 1500);
setInterval(refreshClips, 10000);
