const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const predictedCharEl = document.getElementById('predicted-char');
const predictionConfidenceEl = document.getElementById('prediction-confidence');
const sentenceTextEl = document.getElementById('sentence-text');
const suggestionsEl = document.getElementById('suggestions');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

let currentSentence = "";
let latestPrediction = "?";
let lastPredictionTime = 0;
const PREDICTION_INTERVAL = 300; // ms

// --- Auto-Add State ---
let stableGesture = "?";
let gestureCount = 0;
const STABILITY_THRESHOLD = 4; // Number of cycles to 'lock in' (approx 1.2s)
let autoAddCooldown = false;

const predictionOverlay = document.getElementById('prediction-overlay');
const predictionProgress = document.getElementById('prediction-progress');

// --- Gemini Refinements ---
const btnRefine = document.getElementById('btn-refine');
const refinedOutput = document.getElementById('refined-sentence');
const refinedTextEl = document.getElementById('refined-text');
let refinedSentence = "";

// --- MediaPipe Setup ---
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});

// Manual Control
document.getElementById('btn-camera-on').onclick = () => {
    statusText.innerText = "Starting Camera...";
    camera.start().then(() => {
        statusDot.classList.add('connected');
        statusText.innerText = "Camera Connected";
    }).catch(err => {
        statusText.innerText = "Camera Error";
        console.error(err);
    });
};

document.getElementById('btn-camera-off').onclick = () => {
    camera.stop();
    statusDot.classList.remove('connected');
    statusText.innerText = "Camera Stopped";
    
    // Clear the canvas
    canvasCtx.save();
    canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.restore();
    
    predictedCharEl.innerText = "?";
    predictionConfidenceEl.innerText = "0%";
    predictionProgress.style.width = "0%";
};

// Start default (optional, keeping it off by default for 'user control')
statusText.innerText = "Camera Stopped";

// --- Logic ---

function onResults(results) {
    // Resize canvas to match video
    if (canvasElement.width !== videoElement.videoWidth) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Mirror the view
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);
    
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const now = Date.now();
        const shouldPredict = (now - lastPredictionTime > PREDICTION_INTERVAL);
        
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
            drawLandmarks(canvasCtx, landmarks, { color: '#6366f1', lineWidth: 1, radius: 4 });
        }

        if (shouldPredict) {
            predictBestGesture(results.multiHandLandmarks);
            lastPredictionTime = now;
        }
    } else {
        predictedCharEl.innerText = "?";
        predictionConfidenceEl.innerText = "0%";
        latestPrediction = "?";
        
        // Reset Auto-Add
        stableGesture = "?";
        gestureCount = 0;
        autoAddCooldown = false;
        predictionProgress.style.width = "0%";
        predictionOverlay.classList.remove('locked');
    }
    
    canvasCtx.restore();
}

async function predictBestGesture(multiLandmarks) {
    let bestRes = { label: "?", confidence: 0 };

    for (const landmarks of multiLandmarks) {
        const flattened = normalizeLandmarks(landmarks);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: flattened })
            });
            const result = await response.json();
            
            if (result.confidence > bestRes.confidence) {
                bestRes = result;
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
    }

    if (bestRes.confidence > 0.6) {
        latestPrediction = bestRes.label;
        predictedCharEl.innerText = bestRes.label;
        predictionConfidenceEl.innerText = `${(bestRes.confidence * 100).toFixed(0)}%`;

        // --- Auto-Add Logic ---
        if (latestPrediction === stableGesture && latestPrediction !== "?") {
            if (!autoAddCooldown) {
                gestureCount++;
                const progress = (gestureCount / STABILITY_THRESHOLD) * 100;
                predictionProgress.style.width = `${progress}%`;

                if (gestureCount >= STABILITY_THRESHOLD) {
                    addCharacter(latestPrediction);
                    triggerLockEffect();
                }
            }
        } else {
            stableGesture = latestPrediction;
            gestureCount = 0;
            autoAddCooldown = false;
            predictionProgress.style.width = "0%";
            predictionOverlay.classList.remove('locked');
        }
    } else {
        latestPrediction = "?";
        predictedCharEl.innerText = "?";
        predictionConfidenceEl.innerText = "Low Conf.";
        
        stableGesture = "?";
        gestureCount = 0;
        predictionProgress.style.width = "0%";
    }
}

function addCharacter(char) {
    currentSentence += char;
    autoAddCooldown = true; // Use cooldown to prevent immediate re-add
    updateUI();
}

function triggerLockEffect() {
    predictionOverlay.classList.add('locked');
    setTimeout(() => {
        predictionOverlay.classList.remove('locked');
    }, 400);
}

function normalizeLandmarks(landmarks) {
    // 1. Extract (x, y)
    let pts = landmarks.map(lm => [lm.x, lm.y]);
    
    // 2. Zero-center at wrist (index 0)
    const wrist = pts[0];
    let ptsNorm = pts.map(p => [p[0] - wrist[0], p[1] - wrist[1]]);
    
    // 3. Max-abs scaling to [-1, 1]
    let maxVal = 0;
    ptsNorm.forEach(p => {
        maxVal = Math.max(maxVal, Math.abs(p[0]), Math.abs(p[1]));
    });
    
    if (maxVal > 0) {
        ptsNorm = ptsNorm.map(p => [p[0] / maxVal, p[1] / maxVal]);
    }
    
    // 4. Flatten
    return ptsNorm.flat();
}

// --- UI Actions ---

document.getElementById('btn-add').onclick = () => {
    if (latestPrediction && latestPrediction !== "?") {
        addCharacter(latestPrediction);
        triggerLockEffect();
    }
};

document.getElementById('btn-space').onclick = () => {
    currentSentence += " ";
    updateUI();
};

document.getElementById('btn-refine').onclick = () => {
    if (currentSentence.trim()) {
        refineSentence();
    }
};

async function refineSentence() {
    // UI Feedback: Show loading state
    refinedOutput.classList.remove('hidden');
    refinedTextEl.innerText = "Refining with Gemini AI...";
    refinedOutput.classList.add('shimmer');
    btnRefine.disabled = true;

    try {
        const response = await fetch('/refine', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentSentence })
        });
        const data = await response.json();
        
        refinedSentence = data.refined_text;
        refinedTextEl.innerText = refinedSentence;
    } catch (err) {
        console.error("Refine error:", err);
        refinedTextEl.innerText = "Could not refine at this time.";
    } finally {
        refinedOutput.classList.remove('shimmer');
        btnRefine.disabled = false;
    }
}

document.getElementById('btn-clear').onclick = () => {
    currentSentence = "";
    refinedSentence = "";
    refinedOutput.classList.add('hidden');
    updateUI();
};

document.getElementById('btn-delete').onclick = () => {
    currentSentence = currentSentence.slice(0, -1);
    updateUI();
};

document.getElementById('btn-speak').onclick = () => {
    const textToSpeak = refinedSentence || currentSentence;
    if (textToSpeak) {
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        window.speechSynthesis.speak(utterance);
    }
};

async function updateUI() {
    sentenceTextEl.innerText = currentSentence || "...";
    
    // Fetch Suggestions
    try {
        const response = await fetch('/suggestions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentSentence })
        });
        const data = await response.json();
        renderSuggestions(data.suggestions);
    } catch (err) {
        console.error("Suggestions error:", err);
    }
}

function renderSuggestions(suggestions) {
    suggestionsEl.innerHTML = "";
    suggestions.forEach(word => {
        const btn = document.createElement('button');
        btn.className = "suggestion-btn";
        btn.innerText = word;
        btn.onclick = () => {
            const words = currentSentence.trim().split(" ");
            words[words.length - 1] = word;
            currentSentence = words.join(" ") + " ";
            updateUI();
        };
        suggestionsEl.appendChild(btn);
    });
}
