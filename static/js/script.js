// ================================
// Emotion Recognition AI
// Live Voice Recording
// ================================

let mediaRecorder;
let audioChunks = [];

// Buttons
const startBtn = document.getElementById("startRecord");
const stopBtn = document.getElementById("stopRecord");
const status = document.getElementById("recordStatus");

// ================================
// Start Recording
// ================================

startBtn.addEventListener("click", async () => {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        mediaRecorder = new MediaRecorder(stream);

        audioChunks = [];

        mediaRecorder.start();

        status.innerHTML = "🎙 Recording...";

        startBtn.disabled = true;
        stopBtn.disabled = false;

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {

            const audioBlob = new Blob(audioChunks, {
                type: "audio/webm"
            });

            uploadRecording(audioBlob);

        };

    } catch (err) {

        console.error(err);

        alert("Please allow microphone permission.");

    }

});

// ================================
// Stop Recording
// ================================

stopBtn.addEventListener("click", () => {

    mediaRecorder.stop();

    status.innerHTML = "⏳ Processing Voice...";

    startBtn.disabled = false;
    stopBtn.disabled = true;

});

// ================================
// Upload Recording
// ================================

function uploadRecording(audioBlob) {

    const formData = new FormData();

    formData.append("audio", audioBlob, "recorded.webm");

    fetch("/record", {
        method: "POST",
        body: formData
    })

    .then(async (response) => {

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unknown Server Error");
        }

        return data;

    })

    .then((data) => {

        if (data.error) {

            status.innerHTML = "❌ " + data.error;

            return;

        }

        status.innerHTML =
            "✅ Emotion : <b>" +
            data.emotion +
            "</b> (" +
            data.confidence +
            "%)";

    })

    .catch((err) => {

        console.error(err);

        status.innerHTML =
            "❌ Error : " + err.message;

    });

}