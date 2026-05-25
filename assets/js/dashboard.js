/*
  =========================================
  Adversarial Machine Learning Dashboard UI
  =========================================
*/

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Theme Toggle Handler
  setupThemeToggle();
  
  // Render Terminal Logs
  renderTerminalLogs();
  
  // Render Quick Analytics Progress Bars
  renderQuickStats();
});

// 1. LIGHT / DARK THEME TOGGLE SWITCH
function setupThemeToggle() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  if (!toggleBtn) return;
  
  // Check for local storage preference
  const savedTheme = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(toggleBtn, savedTheme);
  
  toggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    updateThemeIcon(toggleBtn, newTheme);
  });
}

function updateThemeIcon(btn, theme) {
  const icon = btn.querySelector("i");
  if (!icon) return;
  
  if (theme === "light") {
    icon.className = "fas fa-moon";
    btn.setAttribute("title", "Switch to Dark Mode");
  } else {
    icon.className = "fas fa-sun";
    btn.setAttribute("title", "Switch to Light Mode");
  }
}

// 2. TERMINAL LOGS GENERATION WITH TYPING SIMULATION
function renderTerminalLogs() {
  const logContainer = document.getElementById("console-logs-container");
  if (!logContainer) return;
  
  logContainer.innerHTML = "";
  let logIdx = 0;
  const logs = AML_DATABASE.terminalLogs;
  
  function addNextLog() {
    if (logIdx >= logs.length) return;
    
    const line = document.createElement("div");
    line.style.marginBottom = "6px";
    line.style.lineHeight = "1.4";
    
    const text = logs[logIdx];
    
    // Semantic coloring
    if (text.includes("[INFO]")) {
      line.innerHTML = `<span style="color: var(--secondary);">[INFO]</span> ${text.replace("[INFO]", "")}`;
    } else if (text.includes("[SUCCESS]")) {
      line.innerHTML = `<span style="color: #22c55e;">[SUCCESS]</span> <strong style="color: #fff;">${text.replace("[SUCCESS]", "")}</strong>`;
    } else if (text.includes("[BENCHMARK]")) {
      line.innerHTML = `<span style="color: var(--accent-gold);">[BENCHMARK]</span> ${text.replace("[BENCHMARK]", "")}`;
    } else if (text.includes("[EPOCH")) {
      line.innerHTML = `<span style="color: var(--primary);">[TRAINING]</span> ${text}`;
    } else {
      line.textContent = text;
    }
    
    logContainer.appendChild(line);
    logContainer.scrollTop = logContainer.scrollHeight;
    
    logIdx++;
    // Staggered typing timer
    setTimeout(addNextLog, 600 + Math.random() * 400);
  }
  
  addNextLog();
}

// 3. QUICK STATS LOAD & BACKEND SYNCHRONIZATION
function renderQuickStats() {
  const cards = {
    stdClean: document.querySelector(".metric-card.indigo .metric-value"),
    advVuln: document.querySelector(".metric-card.rose .metric-value"),
    robustAcc: document.querySelector(".metric-card.teal .metric-value"),
    preprocessMit: document.querySelector(".metric-card.amber .metric-value")
  };
  
  fetch("results.json")
    .then(response => {
      if (!response.ok) throw new Error("results.json ungenerated");
      return response.json();
    })
    .then(data => {
      console.log("[INFO] Home dashboard synchronized with live computed results.json");
      
      const cleanStd = (data.standard_model.clean_accuracy * 100).toFixed(1) + "%";
      // Find FGSM epsilon = 0.15 index
      const epsIndex = data.epsilons.indexOf(0.15);
      const vulnStd = (data.standard_model.adversarial_accuracies[epsIndex !== -1 ? epsIndex : 3] * 100).toFixed(1) + "%";
      const robustAccVal = (data.robust_model.adversarial_accuracies[epsIndex !== -1 ? epsIndex : 3] * 100).toFixed(1) + "%";
      const jpegAccVal = (data.standard_model.jpeg_accuracies[epsIndex !== -1 ? epsIndex : 3] * 100).toFixed(1) + "%";
      
      if (cards.stdClean) cards.stdClean.textContent = cleanStd;
      if (cards.advVuln) cards.advVuln.textContent = vulnStd;
      if (cards.robustAcc) cards.robustAcc.textContent = robustAccVal;
      if (cards.preprocessMit) cards.preprocessMit.textContent = jpegAccVal;
    })
    .catch(err => {
      console.log("[INFO] Running home dashboard on high-fidelity preset statistics");
    });
}
