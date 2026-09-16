/**
 * DermAI - Front-end Application Controller
 * Handles image upload, webcam capture, API communication,
 * Chart.js probability rendering, and clinical reports.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ==========================================
    // DOM ELEMENTS
    // ==========================================
    const tabUploadBtn = document.getElementById('tab-upload-btn');
    const tabCameraBtn = document.getElementById('tab-camera-btn');
    const paneUpload = document.getElementById('pane-upload');
    const paneCamera = document.getElementById('pane-camera');

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadPreviewImg = document.getElementById('upload-preview-img');
    const clearUploadBtn = document.getElementById('clear-upload-btn');

    const webcamVideo = document.getElementById('webcam-video');
    const snapshotCanvas = document.getElementById('snapshot-canvas');
    const startCamBtn = document.getElementById('start-cam-btn');
    const captureCamBtn = document.getElementById('capture-cam-btn');

    const roiCheckbox = document.getElementById('roi-checkbox');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const scanNewBtn = document.getElementById('scan-new-btn');

    const resultsIdle = document.getElementById('results-idle');
    const resultsLoading = document.getElementById('results-loading');
    const resultsReport = document.getElementById('results-report');

    const resultTitle = document.getElementById('result-title');
    const resultCategory = document.getElementById('result-category');
    const resultConfidence = document.getElementById('result-confidence');
    const severityBadgeContainer = document.getElementById('severity-badge-container');
    const thresholdAlert = document.getElementById('threshold-alert');
    const thresholdIcon = document.getElementById('threshold-icon');
    const thresholdText = document.getElementById('threshold-text');

    const metricSharpness = document.getElementById('metric-sharpness');
    const metricBrightness = document.getElementById('metric-brightness');
    const metricRes = document.getElementById('metric-res');
    const metricStatus = document.getElementById('metric-status');

    const roiPreviewImg = document.getElementById('roi-preview-img');
    const medicalDescription = document.getElementById('medical-description');
    const medicalSymptoms = document.getElementById('medical-symptoms');
    const medicalRecommendations = document.getElementById('medical-recommendations');
    const medicalDoctorAdvice = document.getElementById('medical-doctor-advice');

    const samplesGrid = document.getElementById('samples-grid');
    const diseaseCardsContainer = document.getElementById('disease-cards-container');
    const downloadReportBtn = document.getElementById('download-report-btn');

    // State Variables
    let currentImageFile = null;
    let currentBase64Image = null;
    let mediaStream = null;
    let probabilityChart = null;
    let latestResultData = null;

    // ==========================================
    // INITIALIZATION
    // ==========================================
    loadSamples();
    loadDiseaseEncyclopedia();

    // ==========================================
    // TAB SWITCHING (UPLOAD vs CAMERA)
    // ==========================================
    tabUploadBtn.addEventListener('click', () => {
        tabUploadBtn.className = "flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 bg-white text-teal-700 shadow-sm flex items-center justify-center gap-2";
        tabCameraBtn.className = "flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 text-slate-600 hover:text-slate-900 flex items-center justify-center gap-2";
        paneUpload.classList.remove('hidden');
        paneCamera.classList.add('hidden');
        stopCamera();
    });

    tabCameraBtn.addEventListener('click', () => {
        tabCameraBtn.className = "flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 bg-white text-teal-700 shadow-sm flex items-center justify-center gap-2";
        tabUploadBtn.className = "flex-1 py-2 text-sm font-semibold rounded-lg transition-all duration-200 text-slate-600 hover:text-slate-900 flex items-center justify-center gap-2";
        paneCamera.classList.remove('hidden');
        paneUpload.classList.add('hidden');
        startCamera();
    });

    // ==========================================
    // DRAG AND DROP & FILE UPLOAD
    // ==========================================
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }

        currentImageFile = file;
        currentBase64Image = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            dropZone.classList.add('hidden');
            uploadPreviewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetInputState();
    });

    function resetInputState() {
        currentImageFile = null;
        currentBase64Image = null;
        fileInput.value = '';
        uploadPreviewImg.src = '';
        uploadPreviewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        analyzeBtn.disabled = true;
    }

    // ==========================================
    // WEBCAM CONTROLS
    // ==========================================
    async function startCamera() {
        try {
            if (mediaStream) {
                stopCamera();
            }
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            webcamVideo.srcObject = mediaStream;
            captureCamBtn.disabled = false;
            startCamBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Camera';
            startCamBtn.onclick = stopCamera;
        } catch (err) {
            console.error('Camera access error:', err);
            alert('Unable to access webcam. Please check browser camera permissions.');
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        webcamVideo.srcObject = null;
        captureCamBtn.disabled = true;
        startCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> Start Camera';
        startCamBtn.onclick = startCamera;
    }

    captureCamBtn.addEventListener('click', () => {
        if (!mediaStream) return;

        const videoWidth = webcamVideo.videoWidth || 640;
        const videoHeight = webcamVideo.videoHeight || 480;

        snapshotCanvas.width = videoWidth;
        snapshotCanvas.height = videoHeight;
        const ctx = snapshotCanvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, videoWidth, videoHeight);

        currentBase64Image = snapshotCanvas.toDataURL('image/jpeg', 0.95);
        currentImageFile = null;

        // Switch to upload preview mode to view snapshot
        uploadPreviewImg.src = currentBase64Image;
        tabUploadBtn.click();
        dropZone.classList.add('hidden');
        uploadPreviewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;

        // Auto trigger analysis
        triggerAnalysis();
    });

    // ==========================================
    // ANALYSIS SUBMISSION
    // ==========================================
    analyzeBtn.addEventListener('click', () => triggerAnalysis());

    scanNewBtn.addEventListener('click', () => {
        resultsReport.classList.add('hidden');
        resultsIdle.classList.remove('hidden');
        resetInputState();
        document.getElementById('scanner-section').scrollIntoView({ behavior: 'smooth' });
    });

    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            if (!latestResultData) return;

            const d = latestResultData;
            const med = d.medical_info || {};
            const dateStr = new Date().toLocaleString();

            let report = `==========================================================\n`;
            report += `           DERMAI CLINICAL SCREENING REPORT               \n`;
            report += `==========================================================\n`;
            report += `Timestamp:       ${dateStr}\n`;
            report += `Classification:  ${d.display_name || d.predicted_class}\n`;
            report += `Category:        ${med.category || 'N/A'}\n`;
            report += `Confidence:      ${d.confidence}%\n`;
            report += `Urgency / Triage:${med.urgency || 'N/A'}\n`;
            report += `70% Threshold:   ${d.threshold_met ? 'PASSED (>= 70%)' : 'BELOW THRESHOLD (< 70%)'}\n`;
            report += `----------------------------------------------------------\n`;
            report += `QUALITY INSPECTION METRICS:\n`;
            report += `  - Sharpness Score (Laplacian Var): ${d.quality?.metrics?.blur_score ?? 'N/A'}\n`;
            report += `  - Brightness Score (Mean Gray):    ${d.quality?.metrics?.brightness ?? 'N/A'}\n`;
            report += `  - Resolution:                      ${d.quality?.metrics?.resolution ?? 'N/A'}\n`;
            report += `----------------------------------------------------------\n`;
            report += `MULTI-CLASS PROBABILITIES:\n`;
            if (d.probabilities) {
                for (const [cls, prob] of Object.entries(d.probabilities)) {
                    report += `  - ${cls.padEnd(16)}: ${prob}%\n`;
                }
            }
            report += `----------------------------------------------------------\n`;
            report += `CLINICAL OVERVIEW:\n`;
            report += `${med.description || 'N/A'}\n\n`;
            report += `CHARACTERISTICS & SYMPTOMS:\n`;
            (med.symptoms || []).forEach(s => { report += `  * ${s}\n`; });
            report += `\nRECOMMENDED CARE:\n`;
            (med.recommendations || []).forEach(r => { report += `  * ${r}\n`; });
            report += `\nDOCTOR CONSULTATION ADVICE:\n`;
            report += `${med.when_to_see_doctor || 'Consult a dermatologist if symptoms persist.'}\n`;
            report += `----------------------------------------------------------\n`;
            report += `DISCLAIMER: This is an automated machine learning screening\n`;
            report += `prototype and not a formal medical diagnosis. Always consult\n`;
            report += `a licensed healthcare provider.\n`;
            report += `==========================================================\n`;

            const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `DermAI_Report_${(d.predicted_class || 'Screening').replace(/\\s+/g, '_')}_${Date.now()}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    async function triggerAnalysis() {
        if (!currentImageFile && !currentBase64Image) {
            alert('Please provide an image or take a webcam photo first.');
            return;
        }

        const useRoi = roiCheckbox.checked;

        // UI Loading state
        resultsIdle.classList.add('hidden');
        resultsReport.classList.add('hidden');
        resultsLoading.classList.remove('hidden');
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = 'Analyzing Lesion...';

        try {
            let response;

            if (currentImageFile) {
                const formData = new FormData();
                formData.append('image', currentImageFile);
                formData.append('use_roi', useRoi);

                response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
            } else {
                response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_base64: currentBase64Image,
                        use_roi: useRoi
                    })
                });
            }

            const data = await response.json();

            if (!data.success) {
                resultsLoading.classList.add('hidden');
                resultsIdle.classList.remove('hidden');
                alert(`Analysis Notice: ${data.error || 'The image could not be processed.'}`);
                return;
            }

            displayResults(data);

        } catch (err) {
            console.error('Inference error:', err);
            resultsLoading.classList.add('hidden');
            resultsIdle.classList.remove('hidden');
            alert('An error occurred during screening. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = 'Run Neural Screening';
        }
    }

    // ==========================================
    // RENDER DIAGNOSTIC RESULTS
    // ==========================================
    function displayResults(data) {
        latestResultData = data;
        resultsLoading.classList.add('hidden');
        resultsReport.classList.remove('hidden');

        // Condition Title & Category
        resultTitle.textContent = data.display_name || data.predicted_class;
        resultCategory.textContent = data.medical_info?.category || 'Dermatological Condition';
        resultConfidence.textContent = `${data.confidence.toFixed(1)}%`;

        // Severity / Urgency Badge
        const badgeStyle = getSeverityBadge(data.predicted_class, data.medical_info?.severity_badge);
        severityBadgeContainer.innerHTML = `
            <span class="px-3 py-1 rounded-full text-xs font-bold ${badgeStyle.class} flex items-center gap-1.5 shadow-sm">
                <i class="${badgeStyle.icon}"></i> ${data.medical_info?.urgency || 'Clinical Consultation'}
            </span>
        `;

        // 70% Threshold Warning Indicator
        if (!data.threshold_met && data.threshold_warning) {
            thresholdAlert.classList.remove('hidden');
            thresholdAlert.className = "mt-4 p-3.5 rounded-xl text-xs font-medium flex items-start gap-2.5 bg-amber-50 border border-amber-200 text-amber-900";
            thresholdIcon.className = "fa-solid fa-triangle-exclamation text-amber-600 text-base mt-0.5";
            thresholdText.textContent = data.threshold_warning;
        } else if (data.predicted_class === "Melanoma") {
            thresholdAlert.classList.remove('hidden');
            thresholdAlert.className = "mt-4 p-3.5 rounded-xl text-xs font-medium flex items-start gap-2.5 bg-rose-50 border border-rose-200 text-rose-900";
            thresholdIcon.className = "fa-solid fa-circle-exclamation text-rose-600 text-base mt-0.5";
            thresholdText.textContent = "CRITICAL: The model detected features consistent with Melanoma. Prompt professional clinical examination with dermoscopy is strongly recommended.";
        } else {
            thresholdAlert.classList.remove('hidden');
            thresholdAlert.className = "mt-4 p-3.5 rounded-xl text-xs font-medium flex items-start gap-2.5 bg-emerald-50 border border-emerald-200 text-emerald-900";
            thresholdIcon.className = "fa-solid fa-circle-check text-emerald-600 text-base mt-0.5";
            thresholdText.textContent = `Diagnosis meets the ≥70% confidence benchmark. Primary class: ${data.predicted_class}.`;
        }

        // Image Quality Indicators
        const quality = data.quality || {};
        metricSharpness.textContent = quality.metrics?.blur_score ?? 'N/A';
        metricBrightness.textContent = quality.metrics?.brightness ?? 'N/A';
        metricRes.textContent = quality.metrics?.resolution ?? 'N/A';

        if (quality.passed) {
            metricStatus.innerHTML = `<span class="text-emerald-600"><i class="fa-solid fa-check"></i> Quality Verified</span>`;
        } else {
            metricStatus.innerHTML = `<span class="text-amber-600"><i class="fa-solid fa-triangle-exclamation"></i> Low Quality Warning</span>`;
        }

        // Analyzed ROI Preview
        if (data.roi_preview) {
            roiPreviewImg.src = data.roi_preview;
        } else if (uploadPreviewImg.src) {
            roiPreviewImg.src = uploadPreviewImg.src;
        }

        // Render Probability Chart
        renderProbabilityChart(data.probabilities, data.predicted_class);

        // Medical Guidance Data
        const med = data.medical_info || {};
        medicalDescription.textContent = med.description || 'No detailed medical description available.';

        // Symptoms list
        medicalSymptoms.innerHTML = '';
        (med.symptoms || []).forEach(sym => {
            const li = document.createElement('li');
            li.textContent = sym;
            medicalSymptoms.appendChild(li);
        });

        // Recommendations list
        medicalRecommendations.innerHTML = '';
        (med.recommendations || []).forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            medicalRecommendations.appendChild(li);
        });

        // Doctor Advice
        medicalDoctorAdvice.textContent = med.when_to_see_doctor || 'Consult a certified healthcare provider if symptoms persist or deteriorate.';

        // Scroll to results
        resultsReport.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function getSeverityBadge(className, badgeType) {
        if (className === 'Melanoma') {
            return {
                class: 'badge-danger',
                icon: 'fa-solid fa-triangle-exclamation'
            };
        } else if (className === 'Healthy Skin') {
            return {
                class: 'badge-success',
                icon: 'fa-solid fa-circle-check'
            };
        } else if (badgeType === 'info') {
            return {
                class: 'badge-info',
                icon: 'fa-solid fa-circle-info'
            };
        } else {
            return {
                class: 'badge-warning',
                icon: 'fa-solid fa-circle-radiation'
            };
        }
    }

    // ==========================================
    // PROBABILITY DISTRIBUTION CHART
    // ==========================================
    function renderProbabilityChart(probabilities, predictedClass) {
        const labels = Object.keys(probabilities);
        const values = Object.values(probabilities);

        // Styling: Highlight the winning class
        const backgroundColors = labels.map(label => 
            label === predictedClass ? 'rgba(13, 148, 136, 0.85)' : 'rgba(148, 163, 184, 0.4)'
        );
        const borderColors = labels.map(label => 
            label === predictedClass ? '#0f766e' : '#94a3b8'
        );

        const ctx = document.getElementById('probabilityChart').getContext('2d');

        if (probabilityChart) {
            probabilityChart.destroy();
        }

        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.raw}% confidence`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        ticks: {
                            callback: function(value) { return value + "%"; },
                            font: { size: 10 }
                        },
                        grid: { color: '#f1f5f9' }
                    },
                    y: {
                        ticks: {
                            font: { size: 11, weight: 'bold' },
                            color: '#334155'
                        },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // ==========================================
    // 1-CLICK SAMPLE TEST CASES
    // ==========================================
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();

            if (!data.success || !data.samples || data.samples.length === 0) {
                samplesGrid.innerHTML = '<p class="text-xs text-slate-400 col-span-full">No samples found.</p>';
                return;
            }

            samplesGrid.innerHTML = '';
            data.samples.forEach(sample => {
                const btn = document.createElement('button');
                btn.className = "flex items-center gap-2 p-2 rounded-xl border border-slate-200 hover:border-teal-500 hover:bg-teal-50/40 text-left transition group";
                btn.innerHTML = `
                    <div class="w-10 h-10 rounded-lg overflow-hidden bg-slate-100 flex-shrink-0 border border-slate-200">
                        <img src="${sample.url}" alt="${sample.class_name}" class="w-full h-full object-cover group-hover:scale-110 transition duration-200">
                    </div>
                    <div class="overflow-hidden">
                        <div class="text-xs font-bold text-slate-800 truncate">${sample.class_name}</div>
                        <div class="text-[10px] text-teal-600 font-medium">Click to test</div>
                    </div>
                `;

                btn.addEventListener('click', async () => {
                    // Update preview and analyze sample directly
                    uploadPreviewImg.src = sample.url;
                    tabUploadBtn.click();
                    dropZone.classList.add('hidden');
                    uploadPreviewContainer.classList.remove('hidden');

                    // Run analysis via sample endpoint
                    resultsIdle.classList.add('hidden');
                    resultsReport.classList.add('hidden');
                    resultsLoading.classList.remove('hidden');

                    try {
                        const sRes = await fetch('/api/predict-sample', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                class_name: sample.class_name,
                                filename: sample.filename,
                                use_roi: roiCheckbox.checked
                            })
                        });
                        const sData = await sRes.json();
                        if (sData.success) {
                            displayResults(sData);
                        } else {
                            resultsLoading.classList.add('hidden');
                            resultsIdle.classList.remove('hidden');
                            alert(sData.error || 'Failed to analyze sample.');
                        }
                    } catch (e) {
                        console.error(e);
                        resultsLoading.classList.add('hidden');
                        resultsIdle.classList.remove('hidden');
                    }
                });

                samplesGrid.appendChild(btn);
            });

        } catch (err) {
            console.error('Failed to load test samples:', err);
            samplesGrid.innerHTML = '<p class="text-xs text-slate-400 col-span-full">Could not load samples.</p>';
        }
    }

    // ==========================================
    // DISEASE ENCYCLOPEDIA LOADER
    // ==========================================
    async function loadDiseaseEncyclopedia() {
        try {
            const res = await fetch('/api/disease-info');
            const data = await res.json();

            if (!data.success || !data.diseases) return;

            diseaseCardsContainer.innerHTML = '';
            Object.entries(data.diseases).forEach(([key, info]) => {
                const card = document.createElement('div');
                card.className = "bg-slate-50 rounded-2xl p-6 border border-slate-200 flex flex-col justify-between hover:shadow-md transition";
                
                const badge = getSeverityBadge(key, info.severity_badge);

                card.innerHTML = `
                    <div class="space-y-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs font-bold uppercase tracking-wider text-slate-400">${info.category}</span>
                            <span class="text-[10px] font-bold px-2.5 py-0.5 rounded-full ${badge.class}">${info.severity}</span>
                        </div>
                        <h3 class="text-lg font-bold text-slate-900">${info.title}</h3>
                        <p class="text-xs text-slate-600 leading-relaxed">${info.description}</p>
                        
                        <div class="pt-2">
                            <div class="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-1.5">Common Characteristics:</div>
                            <ul class="text-xs text-slate-500 space-y-1 list-disc list-inside">
                                ${(info.symptoms || []).slice(0, 3).map(s => `<li>${s}</li>`).join('')}
                            </ul>
                        </div>
                    </div>

                    <div class="pt-4 mt-4 border-t border-slate-200/60 text-xs text-teal-700 font-semibold flex items-center gap-1">
                        <i class="fa-solid fa-clock-rotate-left text-teal-600"></i> ${info.urgency}
                    </div>
                `;

                diseaseCardsContainer.appendChild(card);
            });

        } catch (err) {
            console.error('Failed to load encyclopedia:', err);
        }
    }

});
