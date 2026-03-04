document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const previewImage = document.getElementById("preview-image");
    const uploadPlaceholder = document.getElementById("upload-placeholder");
    const btnAnalyze = document.getElementById("btn-analyze");
    const loaderSection = document.getElementById("loader-section");
    const resultsSection = document.getElementById("results-section");

    let selectedFile = null;

    // ── Click to upload ────────────────────────────────
    uploadArea.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // ── Drag & Drop ────────────────────────────────────
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove("hidden");
            uploadPlaceholder.classList.add("hidden");
        };
        reader.readAsDataURL(file);
        btnAnalyze.disabled = false;
    }

    // ── Analyze ────────────────────────────────────────
    btnAnalyze.addEventListener("click", async () => {
        if (!selectedFile) return;

        // UI state
        btnAnalyze.disabled = true;
        loaderSection.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (data.error) {
                alert("Hata: " + data.error);
                return;
            }

            displayResults(data);
        } catch (err) {
            alert("Bir hata oluştu: " + err.message);
        } finally {
            loaderSection.classList.add("hidden");
            btnAnalyze.disabled = false;
        }
    });

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
