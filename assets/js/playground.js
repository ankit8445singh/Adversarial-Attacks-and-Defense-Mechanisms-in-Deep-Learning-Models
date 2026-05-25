/*
  =========================================
  Adversarial Machine Learning Sandbox UI
  =========================================
*/

document.addEventListener("DOMContentLoaded", () => {
  // Ensure the page actually has a playground layout
  const playground = document.getElementById("playground-container");
  if (!playground) return;
  
  // State management
  let activeDataset = "cifar10";
  let activeSampleIdx = 0;
  let activeAttack = "fgsm";
  let activeDefense = "none";
  let epsilon = 0.05;
  
  // DOM Elements
  const btnMnist = document.getElementById("btn-mnist");
  const btnCifar = document.getElementById("btn-cifar");
  const epsSlider = document.getElementById("epsilon-slider");
  const epsValBadge = document.getElementById("epsilon-val-badge");
  
  // Attack cards
  const attackCards = document.querySelectorAll(".attack-card");
  // Defense cards
  const defenseCards = document.querySelectorAll(".defense-card");
  
  // Dynamic Canvas elements
  const canvasClean = document.getElementById("canvas-clean");
  const canvasGrad = document.getElementById("canvas-grad");
  const canvasAdv = document.getElementById("canvas-adv");
  const canvasDefended = document.getElementById("canvas-defended");
  
  // Prediction elements
  const confidenceSection = document.getElementById("confidence-section");
  
  // 1. Initial Render Workspace
  updatePlayground();
  
  // 2. Event Listeners for Dataset Buttons
  if (btnMnist && btnCifar) {
    btnMnist.addEventListener("click", () => {
      activeDataset = "mnist";
      activeSampleIdx = 0;
      btnMnist.classList.add("active");
      btnCifar.classList.remove("active");
      updatePlayground();
    });
    
    btnCifar.addEventListener("click", () => {
      activeDataset = "cifar10";
      activeSampleIdx = 0;
      btnCifar.classList.add("active");
      btnMnist.classList.remove("active");
      updatePlayground();
    });
  }
  
  // 3. Epsilon Slider Listener
  if (epsSlider && epsValBadge) {
    epsSlider.addEventListener("input", (e) => {
      epsilon = parseFloat(e.target.value);
      epsValBadge.textContent = epsilon.toFixed(2);
      updatePlaygroundVisualsOnly();
    });
  }
  
  // 4. Attack Selection Handlers
  attackCards.forEach(card => {
    card.addEventListener("click", () => {
      attackCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      activeAttack = card.getAttribute("data-attack");
      updatePlayground();
    });
  });
  
  // 5. Defense Selection Handlers
  defenseCards.forEach(card => {
    card.addEventListener("click", () => {
      defenseCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      activeDefense = card.getAttribute("data-defense");
      updatePlayground();
    });
  });
  
  // Main update routing
  function updatePlayground() {
    // Obtain Sample Object
    const dataset = AML_DATABASE.datasets[activeDataset];
    const sample = dataset.samples[activeSampleIdx];
    
    // Draw canvases and perform pixel-wise math simulations
    drawAdversarialProcess(
      canvasClean, 
      canvasGrad, 
      canvasAdv, 
      canvasDefended, 
      sample, 
      epsilon, 
      activeAttack, 
      activeDefense
    );
    
    // Calculate and render realistic prediction confidence bars
    updatePredictions(sample);
  }
  
  // Optimised visual-only draw (slider updates) for ultra-fluid real-time feedback
  function updatePlaygroundVisualsOnly() {
    const dataset = AML_DATABASE.datasets[activeDataset];
    const sample = dataset.samples[activeSampleIdx];
    
    drawAdversarialProcess(
      canvasClean, 
      canvasGrad, 
      canvasAdv, 
      canvasDefended, 
      sample, 
      epsilon, 
      activeAttack, 
      activeDefense
    );
    
    updatePredictions(sample);
  }
  
  // Dynamic Classifier Confidence Bar Calculations
  function updatePredictions(sample) {
    if (!confidenceSection) return;
    
    confidenceSection.innerHTML = "";
    
    const trueLabel = sample.label;
    const advLabel = sample.advLabel;
    
    // Dynamic math for classification probabilities under standard vs defended models
    let trueConfidence = 0;
    let advConfidence = 0;
    let otherConfidence = 0;
    
    if (activeDefense === "none") {
      // Direct standard model classification under attack
      // Confidence in true class decays exponentially based on epsilon
      trueConfidence = Math.max(2, Math.round(98 * Math.exp(-epsilon * 12)));
      advConfidence = Math.min(95, Math.round(96 * (1 - Math.exp(-epsilon * 15))));
      otherConfidence = 100 - trueConfidence - advConfidence;
    } 
    else if (activeDefense === "smoothing") {
      // Preprocessing filters successfully mitigate some perturbation but blur detail
      trueConfidence = Math.max(5, Math.round(92 * Math.exp(-epsilon * 5.2)));
      advConfidence = Math.min(85, Math.round(90 * (1 - Math.exp(-epsilon * 4.8))));
      otherConfidence = 100 - trueConfidence - advConfidence;
    } 
    else if (activeDefense === "jpeg") {
      // JPEG removes high frequencies, preserving true class longer
      trueConfidence = Math.max(10, Math.round(94 * Math.exp(-epsilon * 3.8)));
      advConfidence = Math.min(75, Math.round(82 * (1 - Math.exp(-epsilon * 3.5))));
      otherConfidence = 100 - trueConfidence - advConfidence;
    } 
    else if (activeDefense === "bit_depth") {
      // Reduces bit depth. Less accurate clean but mitigates small perturbations
      trueConfidence = Math.max(4, Math.round(86 * Math.exp(-epsilon * 4.5)));
      advConfidence = Math.min(80, Math.round(85 * (1 - Math.exp(-epsilon * 4.2))));
      otherConfidence = 100 - trueConfidence - advConfidence;
    } 
    else if (activeDefense === "adv_training") {
      // Robust model retains high classification confidence even under strong epsilon
      trueConfidence = Math.max(45, Math.round(85 * Math.exp(-epsilon * 1.5)));
      advConfidence = Math.min(35, Math.round(40 * (1 - Math.exp(-epsilon * 1.2))));
      otherConfidence = 100 - trueConfidence - advConfidence;
    }
    
    // Sort classifications for visual list
    const predictions = [
      { label: trueLabel, confidence: trueConfidence, type: "std" },
      { label: advLabel, confidence: advConfidence, type: "adv" },
      { label: "Other classes", confidence: otherConfidence, type: "other" }
    ];
    
    predictions.sort((a, b) => b.confidence - a.confidence);
    
    // Render the UI confidence progress bars
    predictions.forEach(pred => {
      const item = document.createElement("div");
      item.className = "pred-item";
      
      let colorClass = "std";
      if (pred.type === "adv") colorClass = "adv";
      if (pred.type === "other") colorClass = "std"; // simple dark gray fallback
      
      // Customize standard vs adversarial colors
      let barFillColor = "linear-gradient(to right, var(--secondary), #0dd6c2)";
      if (pred.type === "adv") {
        barFillColor = "linear-gradient(to right, var(--accent-warn), #ff3657)";
      } else if (pred.type === "other") {
        barFillColor = "linear-gradient(to right, var(--text-dark), var(--text-muted))";
      }
      
      item.innerHTML = `
        <div class="pred-label-row">
          <span>${pred.label}</span>
          <span style="font-family: var(--font-mono); font-weight: 600;">${pred.confidence}%</span>
        </div>
        <div class="pred-bar-bg">
          <div class="pred-bar-fill" style="width: ${pred.confidence}%; background: ${barFillColor};"></div>
        </div>
      `;
      
      confidenceSection.appendChild(item);
    });
  }
});
