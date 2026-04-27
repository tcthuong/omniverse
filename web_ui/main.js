import { AppStreamer, LogLevel, StreamType } from '@nvidia/omniverse-webrtc-streaming-library';

const slider = document.getElementById('rpmSlider');
const display = document.getElementById('rpmValueDisplay');
const container = document.getElementById('webrtc-stream-container');

const STREAM_HOST = window.location.hostname && window.location.hostname !== 'localhost'
    ? window.location.hostname
    : '127.0.0.1';
const SIGNALING_PORT = 49100;

let isConnecting = false;
let isConnected = false;

function updateSliderBackground(value) {
    const min = Number(slider.min);
    const max = Number(slider.max);
    const percentage = ((Number(value) - min) / (max - min)) * 100;
    slider.style.setProperty('--val', `${percentage}%`);
}

function renderIdleState(message = 'Click Connect to start Omniverse WebRTC') {
    container.innerHTML = `
        <div class="stream-placeholder">
            <span class="stream-text">${message}</span>
        </div>
        <div style="position:absolute;top:16px;left:16px;z-index:100;display:flex;gap:8px;align-items:center;">
            <button id="connect-btn" style="padding:10px 20px;background:#76b900;color:#071000;border:none;border-radius:6px;font-weight:700;cursor:pointer;">
                Connect
            </button>
            <span id="stream-status" style="padding:9px 12px;background:rgba(0,0,0,.72);color:#d7d7d7;border-radius:6px;font-size:13px;">
                ${STREAM_HOST}:${SIGNALING_PORT}
            </span>
        </div>
    `;

    document.getElementById('connect-btn').addEventListener('click', startOmniverseStream);
}

function setStreamStatus(message, tone = 'info') {
    const status = document.getElementById('stream-status');
    if (!status) return;

    const colors = {
        info: '#d7d7d7',
        success: '#b7ff6a',
        warning: '#ffd166',
        error: '#ff8b8b',
    };

    status.style.color = colors[tone] || colors.info;
    status.textContent = message;
}

async function startOmniverseStream() {
    if (isConnecting || isConnected) return;

    isConnecting = true;
    container.innerHTML = `
        <video id="remote-video" autoplay muted playsinline style="width:100%;height:100%;object-fit:cover;background:#050505;"></video>
        <audio id="remote-audio" autoplay muted></audio>
        <div style="position:absolute;top:16px;left:16px;z-index:100;">
            <span id="stream-status" style="padding:9px 12px;background:rgba(0,0,0,.72);color:#d7d7d7;border-radius:6px;font-size:13px;">
                Connecting to ${STREAM_HOST}:${SIGNALING_PORT}...
            </span>
        </div>
    `;

    const streamProps = {
        streamSource: StreamType.DIRECT,
        streamConfig: {
            videoElementId: 'remote-video',
            audioElementId: 'remote-audio',
            authenticate: false,
            maxReconnects: 20,
            signalingServer: STREAM_HOST,
            signalingPort: SIGNALING_PORT,
            mediaServer: STREAM_HOST,
            width: 1920,
            height: 1080,
            fps: 60,
            onUpdate: (message) => {
                console.log('Stream update event:', message);
            },
            onStart: (message) => {
                console.log('Stream start event:', message);

                if (message.status === 'success') {
                    isConnected = true;
                    isConnecting = false;
                    setStreamStatus('Omniverse stream connected', 'success');
                    return;
                }

                if (message.status === 'warning') {
                    setStreamStatus(message.info || 'Stream warning', 'warning');
                    return;
                }

                if (message.status === 'error') {
                    isConnecting = false;
                    isConnected = false;
                    setStreamStatus(message.info || 'Stream connection failed', 'error');
                }
            },
            onStop: (message) => {
                console.log('Stream stop event:', message);
                isConnecting = false;
                isConnected = false;
                setStreamStatus(message.info || 'Stream stopped', message.status === 'error' ? 'error' : 'info');
            },
            onStreamStats: (message) => {
                console.log('Stream stats:', message);
            },
        }
    };

    try {
        console.log('Connecting to Omniverse stream:', streamProps);
        AppStreamer.connect(streamProps)
            .then((result) => {
                console.log('Connect request result:', result);
                if (result.status === 'inProgress') {
                    setStreamStatus('Waiting for Omniverse stream...', 'info');
                }
            })
            .catch((error) => {
                console.error('Cannot connect to Omniverse stream:', error);
                isConnecting = false;
                isConnected = false;
                setStreamStatus(error?.info || error?.message || 'Cannot connect to Omniverse stream', 'error');
            });
    } catch (error) {
        console.error('Cannot connect to Omniverse stream:', error);
        isConnecting = false;
        isConnected = false;
        setStreamStatus(error?.info || error?.message || 'Cannot connect to Omniverse stream', 'error');
    }
}

updateSliderBackground(slider.value);
renderIdleState();

slider.addEventListener('input', (event) => {
    const value = event.target.value;
    display.textContent = value;
    updateSliderBackground(value);

    console.log(`RPM changed to ${value}`);
});
