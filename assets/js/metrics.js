/*
  =========================================
  Adversarial Machine Learning SVG Chart Engine
  =========================================
*/

document.addEventListener("DOMContentLoaded", () => {
  // Check if we are on the metrics layout
  const chartWrapper = document.getElementById("robustness-chart-wrapper");
  if (!chartWrapper) return;
  
  // Attempt to dynamically fetch computed results from the PyTorch backend run
  fetch("results.json")
    .then(response => {
      if (!response.ok) throw new Error("Local results.json not found, falling back to database");
      return response.json();
    })
    .then(data => {
      console.log("[INFO] Successfully loaded live computed PyTorch metrics from results.json");
      
      // Override default mock metrics with real-world custom computed numbers
      AML_DATABASE.metrics.epsilons = data.epsilons;
      AML_DATABASE.metrics.standardModel.adversarialAcc = data.standard_model.adversarial_accuracies;
      AML_DATABASE.metrics.standardModel.smoothingAcc = data.standard_model.smoothing_accuracies;
      AML_DATABASE.metrics.standardModel.jpegAcc = data.standard_model.jpeg_accuracies;
      AML_DATABASE.metrics.standardModel.bitDepthAcc = data.standard_model.bit_depth_accuracies;
      AML_DATABASE.metrics.robustModel.adversarialAcc = data.robust_model.adversarial_accuracies;
      
      // Update clean accuracy numbers in mock data database
      AML_DATABASE.metrics.standardModel.adversarialAcc[0] = data.standard_model.clean_accuracy;
      AML_DATABASE.metrics.robustModel.adversarialAcc[0] = data.robust_model.clean_accuracy;
    })
    .catch(err => {
      console.log("[INFO] Running on fallback local metrics (CORS or results.json ungenerated):", err.message);
    })
    .finally(() => {
      // Render both dynamic curves
      renderRobustnessChart();
      renderPreprocessingDefenseChart();
    });
  
  // Rerender on window resize to guarantee perfect responsive layout scaling
  window.addEventListener("resize", () => {
    renderRobustnessChart();
    renderPreprocessingDefenseChart();
  });
});

// CHART 1: MODEL ROBUSTNESS COMPARISON (ACCURACY VS EPSILON)
function renderRobustnessChart() {
  const container = document.getElementById("robustness-chart-wrapper");
  if (!container) return;
  
  // Clear previous SVG
  container.innerHTML = "";
  
  const width = container.clientWidth;
  const height = container.clientHeight;
  const paddingLeft = 55;
  const paddingRight = 30;
  const paddingTop = 30;
  const paddingBottom = 45;
  
  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;
  
  // Data extraction
  const epsilons = AML_DATABASE.metrics.epsilons;
  const stdAccs = AML_DATABASE.metrics.standardModel.adversarialAcc;
  const robustAccs = AML_DATABASE.metrics.robustModel.adversarialAcc;
  
  // Initialize SVG container
  let svg = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  
  // Draw Grid Lines (Y-Axis)
  for (let i = 0; i <= 5; i++) {
    const yVal = i * 0.2;
    const yCoord = paddingTop + chartH * (1 - yVal);
    
    // Horizontal grid line
    svg += `<line class="grid-line" x1="${paddingLeft}" y1="${yCoord}" x2="${width - paddingRight}" y2="${yCoord}" />`;
    
    // Y-Axis label
    svg += `<text class="axis-text" x="${paddingLeft - 10}" y="${yCoord + 3}" text-anchor="end">${(yVal * 100).toFixed(0)}%</text>`;
  }
  
  // Draw Grid Lines (X-Axis)
  for (let i = 0; i < epsilons.length; i++) {
    const xVal = epsilons[i];
    const xCoord = paddingLeft + (i * (chartW / (epsilons.length - 1)));
    
    // Vertical grid line
    svg += `<line class="grid-line" x1="${xCoord}" y1="${paddingTop}" x2="${xCoord}" y2="${paddingTop + chartH}" />`;
    
    // X-Axis label
    svg += `<text class="axis-text" x="${xCoord}" y="${paddingTop + chartH + 20}" text-anchor="middle">ε = ${xVal.toFixed(2)}</text>`;
  }
  
  // Axis Lines
  svg += `<line class="axis-line" x1="${paddingLeft}" y1="${paddingTop}" x2="${paddingLeft}" y2="${paddingTop + chartH}" />`;
  svg += `<line class="axis-line" x1="${paddingLeft}" y1="${paddingTop + chartH}" x2="${width - paddingRight}" y2="${paddingTop + chartH}" />`;
  
  // Helper to compute coordinates
  function getCoords(i, accList) {
    const x = paddingLeft + (i * (chartW / (epsilons.length - 1)));
    const y = paddingTop + chartH * (1 - accList[i]);
    return { x, y };
  }
  
  // Generate Line Paths
  let stdPath = "";
  let robustPath = "";
  
  for (let i = 0; i < epsilons.length; i++) {
    const pStd = getCoords(i, stdAccs);
    const pRob = getCoords(i, robustAccs);
    
    if (i === 0) {
      stdPath += `M ${pStd.x} ${pStd.y}`;
      robustPath += `M ${pRob.x} ${pRob.y}`;
    } else {
      stdPath += ` L ${pStd.x} ${pStd.y}`;
      robustPath += ` L ${pRob.x} ${pRob.y}`;
    }
  }
  
  // Render Paths
  svg += `<path class="line-std" d="${stdPath}" />`;
  svg += `<path class="line-robust" d="${robustPath}" />`;
  
  // Render Data Points
  for (let i = 0; i < epsilons.length; i++) {
    const pStd = getCoords(i, stdAccs);
    const pRob = getCoords(i, robustAccs);
    
    // Standard model point (Glows red under attack)
    svg += `<circle class="point-std" cx="${pStd.x}" cy="${pStd.y}" r="5" />`;
    
    // Robust model point (Glows teal)
    svg += `<circle class="point-robust" cx="${pRob.x}" cy="${pRob.y}" r="5" />`;
  }
  
  svg += `</svg>`;
  container.innerHTML = svg;
}

// CHART 2: MITIGATION DEFENSES COMPARISON (ACCURACY VS EPSILON)
function renderPreprocessingDefenseChart() {
  const container = document.getElementById("preprocessing-chart-wrapper");
  if (!container) return;
  
  container.innerHTML = "";
  
  const width = container.clientWidth;
  const height = container.clientHeight;
  const paddingLeft = 55;
  const paddingRight = 30;
  const paddingTop = 30;
  const paddingBottom = 45;
  
  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;
  
  const epsilons = AML_DATABASE.metrics.epsilons;
  const noneAccs = AML_DATABASE.metrics.standardModel.adversarialAcc;
  const smoothAccs = AML_DATABASE.metrics.standardModel.smoothingAcc;
  const jpegAccs = AML_DATABASE.metrics.standardModel.jpegAcc;
  const bitAccs = AML_DATABASE.metrics.standardModel.bitDepthAcc;
  
  let svg = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  
  // Draw Grid Lines (Y-Axis)
  for (let i = 0; i <= 5; i++) {
    const yVal = i * 0.2;
    const yCoord = paddingTop + chartH * (1 - yVal);
    
    svg += `<line class="grid-line" x1="${paddingLeft}" y1="${yCoord}" x2="${width - paddingRight}" y2="${yCoord}" />`;
    svg += `<text class="axis-text" x="${paddingLeft - 10}" y="${yCoord + 3}" text-anchor="end">${(yVal * 100).toFixed(0)}%</text>`;
  }
  
  // Draw Grid Lines (X-Axis)
  for (let i = 0; i < epsilons.length; i++) {
    const xVal = epsilons[i];
    const xCoord = paddingLeft + (i * (chartW / (epsilons.length - 1)));
    
    svg += `<line class="grid-line" x1="${xCoord}" y1="${paddingTop}" x2="${xCoord}" y2="${paddingTop + chartH}" />`;
    svg += `<text class="axis-text" x="${xCoord}" y="${paddingTop + chartH + 20}" text-anchor="middle">ε = ${xVal.toFixed(2)}</text>`;
  }
  
  // Axis Lines
  svg += `<line class="axis-line" x1="${paddingLeft}" y1="${paddingTop}" x2="${paddingLeft}" y2="${paddingTop + chartH}" />`;
  svg += `<line class="axis-line" x1="${paddingLeft}" y1="${paddingTop + chartH}" x2="${width - paddingRight}" y2="${paddingTop + chartH}" />`;
  
  function getCoords(i, accList) {
    const x = paddingLeft + (i * (chartW / (epsilons.length - 1)));
    const y = paddingTop + chartH * (1 - accList[i]);
    return { x, y };
  }
  
  // Generate Line Paths for preprocessors
  let nonePath = "";
  let smoothPath = "";
  let jpegPath = "";
  let bitPath = "";
  
  for (let i = 0; i < epsilons.length; i++) {
    const pNone = getCoords(i, noneAccs);
    const pSmooth = getCoords(i, smoothAccs);
    const pJpeg = getCoords(i, jpegAccs);
    const pBit = getCoords(i, bitAccs);
    
    if (i === 0) {
      nonePath += `M ${pNone.x} ${pNone.y}`;
      smoothPath += `M ${pSmooth.x} ${pSmooth.y}`;
      jpegPath += `M ${pJpeg.x} ${pJpeg.y}`;
      bitPath += `M ${pBit.x} ${pBit.y}`;
    } else {
      nonePath += ` L ${pNone.x} ${pNone.y}`;
      smoothPath += ` L ${pSmooth.x} ${pSmooth.y}`;
      jpegPath += ` L ${pJpeg.x} ${pJpeg.y}`;
      bitPath += ` L ${pBit.x} ${pBit.y}`;
    }
  }
  
  // Render Paths using contrasting elegant HSL-spaced accents
  svg += `<path class="line-std" d="${nonePath}" style="stroke: var(--accent-warn); stroke-width: 2;" />`; // Red (None)
  svg += `<path class="line-preproc" d="${smoothPath}" style="stroke: var(--primary); stroke-dasharray: none;" />`; // Indigo (Smoothing)
  svg += `<path class="line-preproc" d="${jpegPath}" style="stroke: var(--secondary); stroke-dasharray: none;" />`; // Teal (JPEG)
  svg += `<path class="line-preproc" d="${bitPath}" style="stroke: var(--accent-gold); stroke-dasharray: 4;" />`; // Yellow (Bit depth)
  
  // Render Points
  for (let i = 0; i < epsilons.length; i++) {
    const pNone = getCoords(i, noneAccs);
    const pSmooth = getCoords(i, smoothAccs);
    const pJpeg = getCoords(i, jpegAccs);
    const pBit = getCoords(i, bitAccs);
    
    svg += `<circle cx="${pNone.x}" cy="${pNone.y}" r="3" style="fill: var(--bg-main); stroke: var(--accent-warn); stroke-width: 2;" />`;
    svg += `<circle cx="${pSmooth.x}" cy="${pSmooth.y}" r="3" style="fill: var(--bg-main); stroke: var(--primary); stroke-width: 2;" />`;
    svg += `<circle cx="${pJpeg.x}" cy="${pJpeg.y}" r="3" style="fill: var(--bg-main); stroke: var(--secondary); stroke-width: 2;" />`;
    svg += `<circle cx="${pBit.x}" cy="${pBit.y}" r="3" style="fill: var(--bg-main); stroke: var(--accent-gold); stroke-width: 2;" />`;
  }
  
  svg += `</svg>`;
  container.innerHTML = svg;
}
