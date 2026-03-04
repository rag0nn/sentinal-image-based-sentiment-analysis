document.addEventListener("DOMContentLoaded", () => {
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const webcamContainer = document.getElementById("webcam-container");
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("webcam-canvas");
    const resultsSection = document.getElementById("results-section");

    let stream = null;
    let analyzing = false;
    let pendingRequest = false;

    // ── Start Webcam ───────────────────────────────────
    btnStart.addEventListener("click", async () => {
        try {
            // Güvenli bağlam kontrolü
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                // Legacy fallback
                const legacyGetUserMedia = navigator.getUserMedia ||
                    navigator.webkitGetUserMedia ||
                    navigator.mozGetUserMedia;

                if (!legacyGetUserMedia) {
                    throw new Error(
                        "Webcam erişimi için lütfen localhost üzerinden erişin " +
                        "(http://localhost:5000) veya HTTPS kullanın."
                    );
                }

                stream = await new Promise((resolve, reject) => {
                    legacyGetUserMedia.call(navigator,
                        { video: { width: 640, height: 480, facingMode: "user" } },
                        resolve, reject
                    );
                });
            } else {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: "user" },
                });
            }

            video.srcObject = stream;
            webcamContainer.classList.remove("hidden");
            btnStart.classList.add("hidden");
            btnStop.classList.remove("hidden");
            analyzing = true;

            // Video hazır olunca analiz döngüsünü başlat
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                analyzeLoop();
            };
        } catch (err) {
            alert("Webcam erişimi sağlanamadı: " + err.message);
        }
    });

    // ── Stop Webcam ────────────────────────────────────
    btnStop.addEventListener("click", () => {
        analyzing = false;
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        video.srcObject = null;
        webcamContainer.classList.add("hidden");
        btnStop.classList.add("hidden");
        btnStart.classList.remove("hidden");
    });

    // ── Analyze Loop ───────────────────────────────────
    async function analyzeLoop() {
        if (!analyzing) return;

        // Önceki request henüz tamamlanmadıysa bekle
        if (pendingRequest) {
            requestAnimationFrame(analyzeLoop);
            return;
        }

        // Canvas'a frame çiz ve base64'e çevir
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameBase64 = canvas.toDataURL("image/jpeg", 0.7);

        pendingRequest = true;

        try {
            const res = await fetch("/api/analyze-frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: frameBase64 }),
            });
            const data = await res.json();

            if (!data.error) {
                displayResults(data);
            }
        } catch (err) {
            // Sessizce devam et
        } finally {
            pendingRequest = false;
        }

        if (analyzing) {
            // Sonraki frame'e geç (throttle: bir request bitince sonraki başlar)
            requestAnimationFrame(analyzeLoop);
        }
    }

    function displayResults(data) {
        document.getElementById("result-original").src =
            "data:image/jpeg;base64," + data.original_image;
        document.getElementById("result-annotated").src =
            "data:image/jpeg;base64," + data.annotated_image;
        document.getElementById("result-cropped").src =
            "data:image/jpeg;base64," + data.cropped_face;
        document.getElementById("pred-en").textContent = data.label_en;
        document.getElementById("pred-tr").textContent = data.label_tr;
        document.getElementById("pred-conf").textContent =
            (data.confidence * 100).toFixed(1) + "%";
        resultsSection.classList.remove("hidden");
    }
});
