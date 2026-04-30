import { AppStreamer, LogLevel, StreamType } from '@nvidia/omniverse-webrtc-streaming-library';
import './style.css';

const RAD_S_TO_RPM = 30 / Math.PI;
const RPM_TO_RAD_S = Math.PI / 30;
const OMEGA_MIN_RAD_S = 150;
const OMEGA_MAX_RAD_S = 350;
const FRAME_STEP_RAD_S = 10;
const RPM_MIN = Math.round(OMEGA_MIN_RAD_S * RAD_S_TO_RPM);  // ~1432
const RPM_MAX = Math.round(OMEGA_MAX_RAD_S * RAD_S_TO_RPM);  // ~3342
const RPM_STEP = 10;
const params = new URLSearchParams(window.location.search);
const STREAM_HOST = params.get('host')
  || (window.location.hostname && window.location.hostname !== 'localhost'
    ? window.location.hostname
    : '127.0.0.1');
const SIGNALING_PORT = Number(params.get('port') || 49100);
const MEDIA_PORT = params.has('mediaPort') ? Number(params.get('mediaPort')) : 47998;
const STREAM_WIDTH = Number(params.get('width') || 1280);
const STREAM_HEIGHT = Number(params.get('height') || 720);
const STREAM_FPS = Number(params.get('fps') || 30);

let isConnecting = false;
let isConnected = false;
let pendingSend = 0;

document.querySelector('#app').innerHTML = `
  <main id="viewport-shell">
    <section id="webrtc-stream-container" aria-label="Omniverse stream">
      <div class="stream-placeholder">
        <span class="stream-text">Omniverse WebRTC</span>
      </div>
      <video id="remote-video" autoplay muted playsinline></video>
      <audio id="remote-audio" autoplay muted></audio>
    </section>

    <section class="floating-panel" aria-label="CFD controls">
      <div class="panel-header">
        <div>
          <h1 class="title">Volume Trace</h1>
          <p id="stream-endpoint" class="endpoint">${STREAM_HOST}:${SIGNALING_PORT} / ${MEDIA_PORT}</p>
        </div>
        <span id="status-light" class="status-light idle"></span>
      </div>

      <div class="button-row">
        <button id="connect-btn" type="button">Connect</button>
        <button id="disconnect-btn" type="button" disabled>Disconnect</button>
      </div>
      <p id="stream-status" class="status-text">Ready</p>

      <div class="divider"></div>

      <div class="control-group">
        <label class="control-label" for="omegaSlider">Rotor speed</label>
        <div class="slider-container">
          <span class="slider-bound">${RPM_MIN}</span>
          <input
            type="range"
            id="omegaSlider"
            min="${RPM_MIN}"
            max="${RPM_MAX}"
            step="${RPM_STEP}"
            value="${RPM_MIN}"
            class="custom-slider"
          >
          <span class="slider-bound">${RPM_MAX}</span>
        </div>
        <div class="current-value">
          <span id="omegaValueDisplay">${RPM_MIN}</span><span class="unit">RPM</span>
        </div>
      </div>

      <div class="divider"></div>

      <div class="control-group">
        <div class="control-label">Velocity magnitude</div>
        <div class="legend-container">
          <div class="legend-bar turbo-gradient"></div>
          <div class="legend-labels">
            <span>0</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
          </div>
        </div>
      </div>
    </section>
  </main>
`;

const slider = document.getElementById('omegaSlider');
const display = document.getElementById('omegaValueDisplay');
const connectButton = document.getElementById('connect-btn');
const disconnectButton = document.getElementById('disconnect-btn');
const statusText = document.getElementById('stream-status');
const statusLight = document.getElementById('status-light');
const container = document.getElementById('webrtc-stream-container');

function rpmToRadS(rpm) {
  return rpm * RPM_TO_RAD_S;
}

function rpmToFrame(rpm) {
  return (rpmToRadS(rpm) - OMEGA_MIN_RAD_S) / FRAME_STEP_RAD_S;
}

function updateSliderBackground(value) {
  const min = Number(slider.min);
  const max = Number(slider.max);
  const percentage = ((Number(value) - min) / (max - min)) * 100;
  slider.style.setProperty('--val', `${percentage}%`);
}

function setStatus(message, tone = 'idle') {
  statusText.textContent = message;
  statusLight.className = `status-light ${tone}`;
}

function setConnectedState(connected) {
  isConnected = connected;
  isConnecting = false;
  connectButton.disabled = connected;
  disconnectButton.disabled = !connected;
  container.classList.toggle('connected', connected);
}

function currentOmegaMessage() {
  const rpm = Number(slider.value);
  const omega_rad_s = rpmToRadS(rpm);
  return {
    event_type: 'cfd.setOmega',
    payload: {
      rpm,
      omega_rad_s,
      frame: rpmToFrame(rpm),
      omega_min_rad_s: OMEGA_MIN_RAD_S,
      omega_max_rad_s: OMEGA_MAX_RAD_S,
      frame_step_rad_s: FRAME_STEP_RAD_S,
    },
  };
}

async function sendOmegaToKit() {
  if (!isConnected) {
    return;
  }
  try {
    await AppStreamer.sendMessage(currentOmegaMessage());
    setStatus(`Sent ${slider.value} RPM`, 'connected');
  } catch (error) {
    setStatus(error?.info || error?.message || 'Cannot send speed update', 'error');
  }
}

function queueOmegaSend() {
  window.clearTimeout(pendingSend);
  pendingSend = window.setTimeout(sendOmegaToKit, 80);
}

async function connectStream() {
  if (isConnecting || isConnected) {
    return;
  }

  isConnecting = true;
  connectButton.disabled = true;
  setStatus('Connecting...', 'connecting');

  const streamProps = {
    streamSource: StreamType.DIRECT,
    logLevel: LogLevel.WARN,
    streamConfig: {
      videoElementId: 'remote-video',
      audioElementId: 'remote-audio',
      authenticate: true,
      maxReconnects: 20,
      signalingServer: STREAM_HOST,
      signalingPort: SIGNALING_PORT,
      mediaServer: STREAM_HOST,
      mediaPort: MEDIA_PORT,
      nativeTouchEvents: true,
      width: STREAM_WIDTH,
      height: STREAM_HEIGHT,
      fps: STREAM_FPS,
      onUpdate: (message) => {
        console.info('stream update', message);
      },
      onStart: (message) => {
        if (message.action === 'start' && message.status === 'success') {
          setConnectedState(true);
          setStatus('Connected', 'connected');
          sendOmegaToKit();
          return;
        }
        if (message.status === 'warning') {
          setStatus(message.info || 'Stream warning', 'connecting');
          return;
        }
        if (message.status === 'error') {
          setConnectedState(false);
          setStatus(message.info || 'Stream connection failed', 'error');
        }
      },
      onStop: (message) => {
        setConnectedState(false);
        setStatus(message.info || 'Disconnected', message.status === 'error' ? 'error' : 'idle');
      },
      onTerminate: (message) => {
        setConnectedState(false);
        setStatus(message.info || 'Disconnected', message.status === 'error' ? 'error' : 'idle');
      },
    },
  };

  try {
    const result = await AppStreamer.connect(streamProps);
    if (result.status === 'inProgress') {
      setStatus('Waiting for stream...', 'connecting');
    }
  } catch (error) {
    setConnectedState(false);
    setStatus(error?.info || error?.message || 'Cannot connect', 'error');
  }
}

async function disconnectStream() {
  if (!isConnected) {
    return;
  }
  try {
    setStatus('Disconnecting...', 'connecting');
    await AppStreamer.stop();
    setConnectedState(false);
    setStatus('Ready', 'idle');
  } catch (error) {
    setStatus(error?.info || error?.message || 'Cannot disconnect', 'error');
  }
}

updateSliderBackground(slider.value);

slider.addEventListener('input', (event) => {
  display.textContent = event.target.value;
  updateSliderBackground(event.target.value);
  queueOmegaSend();
});

connectButton.addEventListener('click', connectStream);
disconnectButton.addEventListener('click', disconnectStream);

window.addEventListener('beforeunload', () => {
  if (isConnected) {
    AppStreamer.stop();
  }
});
