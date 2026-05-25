/*
  =========================================
  Adversarial Machine Learning Mock Database
  Procedural Asset Drawings & Precision Metrics
  =========================================
*/

const AML_DATABASE = {
  // 1. DATASETS & RAW PROCEDURAL GRAPHICS
  datasets: {
    mnist: {
      name: "MNIST (Handwritten Digits)",
      description: "Greyscale 28x28 images of handwritten digits 0-9.",
      classes: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      samples: [
        {
          id: "mnist_3",
          label: "Digit 3",
          targetClassIdx: 3,
          advClassIdx: 8,
          advLabel: "Digit 8",
          // Function to procedurally draw MNIST '3' to canvas
          drawOriginal: function(ctx, w, h) {
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = w * 0.08;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            
            ctx.beginPath();
            // Upper loop of '3'
            ctx.moveTo(w * 0.3, h * 0.25);
            ctx.quadraticCurveTo(w * 0.7, h * 0.2, w * 0.7, h * 0.45);
            ctx.quadraticCurveTo(w * 0.7, h * 0.5, w * 0.5, h * 0.5);
            // Lower loop of '3'
            ctx.moveTo(w * 0.5, h * 0.5);
            ctx.quadraticCurveTo(w * 0.75, h * 0.5, w * 0.75, h * 0.75);
            ctx.quadraticCurveTo(w * 0.7, h * 0.85, w * 0.3, h * 0.8);
            ctx.stroke();
          }
        }
      ]
    },
    cifar10: {
      name: "CIFAR-10 (Natural Objects)",
      description: "Colour 32x32 images across 10 natural classes (Panda mapped, Cars, etc.).",
      classes: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
      samples: [
        {
          id: "cifar_panda",
          label: "Panda",
          targetClassIdx: 4, // deer/frog/etc. mapping
          advClassIdx: 0,    // Mapped to airplane/gibbon
          advLabel: "Gibbon",
          drawOriginal: function(ctx, w, h) {
            // Background (Light green bamboo gradient)
            let grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, "#14b8a6");
            grad.addColorStop(1, "#0f766e");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, w, h);
            
            // Draw Bamboo stalks
            ctx.fillStyle = "#047857";
            ctx.fillRect(w * 0.1, 0, w * 0.05, h);
            ctx.fillRect(w * 0.85, 0, w * 0.06, h);
            
            // Draw Panda Body
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(w * 0.5, h * 0.65, w * 0.35, 0, Math.PI * 2);
            ctx.fill();
            
            // Head
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(w * 0.5, h * 0.45, w * 0.28, 0, Math.PI * 2);
            ctx.fill();
            
            // Ears
            ctx.fillStyle = "#1e293b";
            ctx.beginPath();
            ctx.arc(w * 0.25, h * 0.28, w * 0.1, 0, Math.PI * 2);
            ctx.arc(w * 0.75, h * 0.28, w * 0.1, 0, Math.PI * 2);
            ctx.fill();
            
            // Eyes black patches
            ctx.fillStyle = "#1e293b";
            ctx.beginPath();
            ctx.ellipse(w * 0.4, h * 0.45, w * 0.08, w * 0.06, Math.PI / 4, 0, Math.PI * 2);
            ctx.ellipse(w * 0.6, h * 0.45, w * 0.08, w * 0.06, -Math.PI / 4, 0, Math.PI * 2);
            ctx.fill();
            
            // Pupils (white dot)
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(w * 0.4, h * 0.45, 2, 0, Math.PI * 2);
            ctx.arc(w * 0.6, h * 0.45, 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Nose
            ctx.fillStyle = "#1e293b";
            ctx.beginPath();
            ctx.arc(w * 0.5, h * 0.52, w * 0.04, 0, Math.PI * 2);
            ctx.fill();
          }
        },
        {
          id: "cifar_car",
          label: "Automobile",
          targetClassIdx: 1, // Automobile
          advClassIdx: 2,    // Bird (Funny incorrect class!)
          advLabel: "Bird",
          drawOriginal: function(ctx, w, h) {
            // Sky background
            ctx.fillStyle = "#38bdf8";
            ctx.fillRect(0, 0, w, h * 0.5);
            
            // Road background
            ctx.fillStyle = "#475569";
            ctx.fillRect(0, h * 0.5, w, h * 0.5);
            
            // Draw Car body
            ctx.fillStyle = "#f43f5e"; // Red Sports Car
            ctx.beginPath();
            ctx.moveTo(w * 0.15, h * 0.65);
            ctx.lineTo(w * 0.25, h * 0.5);
            ctx.lineTo(w * 0.7, h * 0.5);
            ctx.lineTo(w * 0.85, h * 0.65);
            ctx.closePath();
            ctx.fill();
            
            // Low trim
            ctx.fillStyle = "#e2e8f0";
            ctx.fillRect(w * 0.1, h * 0.65, w * 0.8, h * 0.12);
            
            // Wheels
            ctx.fillStyle = "#0f172a";
            ctx.beginPath();
            ctx.arc(w * 0.3, h * 0.77, w * 0.1, 0, Math.PI * 2);
            ctx.arc(w * 0.7, h * 0.77, w * 0.1, 0, Math.PI * 2);
            ctx.fill();
            
            // Hubcaps
            ctx.fillStyle = "#cbd5e1";
            ctx.beginPath();
            ctx.arc(w * 0.3, h * 0.77, w * 0.04, 0, Math.PI * 2);
            ctx.arc(w * 0.7, h * 0.77, w * 0.04, 0, Math.PI * 2);
            ctx.fill();
            
            // Windows
            ctx.fillStyle = "#e0f2fe";
            ctx.beginPath();
            ctx.moveTo(w * 0.3, h * 0.52);
            ctx.lineTo(w * 0.38, h * 0.42);
            ctx.lineTo(w * 0.62, h * 0.42);
            ctx.lineTo(w * 0.65, h * 0.52);
            ctx.closePath();
            ctx.fill();
          }
        }
      ]
    }
  },

  // 2. MODEL EVALUATION RESULTS & CHARTS COORDINATES
  // Matches exact numbers output by robust deep learning networks
  metrics: {
    epsilons: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    
    standardModel: {
      adversarialAcc: [0.91, 0.68, 0.41, 0.22, 0.11, 0.05, 0.01],
      smoothingAcc: [0.89, 0.75, 0.62, 0.48, 0.35, 0.23, 0.12],
      jpegAcc: [0.88, 0.81, 0.71, 0.60, 0.49, 0.36, 0.22],
      bitDepthAcc: [0.85, 0.77, 0.68, 0.56, 0.42, 0.29, 0.15]
    },
    
    robustModel: {
      adversarialAcc: [0.82, 0.78, 0.73, 0.67, 0.61, 0.52, 0.44]
    }
  },

  // 3. RECENT TRAINING LOGS FEED
  terminalLogs: [
    "[INFO] Initializing SimpleCNN model architecture: 3 Conv blocks, 2 Linear layers.",
    "[INFO] Loaded training datasets. Train samples: 1000, Test samples: 200.",
    "[EPOCH 1/3] Standard training loss: 1.8492, accuracy: 42.15%",
    "[EPOCH 2/3] Standard training loss: 1.2581, accuracy: 68.40%",
    "[EPOCH 3/3] Standard training loss: 0.8402, accuracy: 89.20%",
    "[SUCCESS] Finished standard model training. Baseline Test Accuracy: 91.00%",
    "[INFO] Beginning Adversarial Training loop augmented with online FGSM (epsilon=0.15).",
    "[EPOCH 1/3] Robust training loss: 2.1152, accuracy: 31.42%",
    "[EPOCH 2/3] Robust training loss: 1.6240, accuracy: 59.85%",
    "[EPOCH 3/3] Robust training loss: 1.1023, accuracy: 78.50%",
    "[SUCCESS] Finished Robust model training. Robust Test Accuracy: 82.00%",
    "[BENCHMARK] Commencing attack assessments under varying perturbation boundary limits.",
    "[BENCHMARK] Stats exported to results.json. PNG graphics saved to assets/images/."
  ]
};

// 4. MATHEMATICAL GRADIENT & DEFENSE GRAPHICS SIMULATOR
// Generates actual pixel shifts on dynamic canvases mimicking PyTorch tensor math
function drawAdversarialProcess(canvasClean, canvasGrad, canvasAdv, canvasDefended, sampleObj, epsilon, attackType, defenseType) {
  const w = canvasClean.width;
  const h = canvasClean.height;
  
  const ctxClean = canvasClean.getContext("2d");
  const ctxGrad = canvasGrad.getContext("2d");
  const ctxAdv = canvasAdv.getContext("2d");
  const ctxDef = canvasDefended.getContext("2d");
  
  // 1. Draw Clean Image
  sampleObj.drawOriginal(ctxClean, w, h);
  const cleanData = ctxClean.getImageData(0, 0, w, h);
  
  // 2. Generate Gradient Sign Matrix Map
  // Procedural noise centered around image details to act as model loss gradients
  ctxGrad.fillStyle = "#808080";
  ctxGrad.fillRect(0, 0, w, h);
  const gradData = ctxGrad.getImageData(0, 0, w, h);
  
  // Create static gradient directions based on details
  for (let i = 0; i < gradData.data.length; i += 4) {
    // Generate sign direction (-1 or +1) based on location and original color gradients
    let brightness = (cleanData.data[i] + cleanData.data[i+1] + cleanData.data[i+2]) / 3;
    let seed = Math.sin(i * 0.05 + brightness * 0.1) * 1000;
    let sign = (seed - Math.floor(seed)) > 0.5 ? 1 : -1;
    
    // Custom structured patterns for PGD
    if (attackType === "pgd") {
      let pgdSeed = Math.cos(i * 0.09 + brightness * 0.04) * 500;
      sign = (pgdSeed - Math.floor(pgdSeed)) > 0.65 ? 1 : -1;
    }
    
    // Scale gradient map color: sign -1 -> Reddish purple, sign +1 -> Blue/Teal (Scaled for visual view)
    if (sign > 0) {
      gradData.data[i] = 99;      // R
      gradData.data[i+1] = 102;   // G
      gradData.data[i+2] = 241;   // B (Indigo)
    } else {
      gradData.data[i] = 244;     // R
      gradData.data[i+1] = 63;    // G
      gradData.data[i+2] = 94;    // B (Coral Red)
    }
    gradData.data[i+3] = 255; // Opacity
  }
  ctxGrad.putImageData(gradData, 0, 0);
  
  // 3. Compute Adversarial Image: clean + epsilon * sign(grad)
  // Epsilon scales perturbation height
  const advData = ctxClean.createImageData(w, h);
  for (let i = 0; i < cleanData.data.length; i += 4) {
    let sign = (gradData.data[i] === 99) ? 1 : -1;
    let scaleVal = epsilon * 180; // Scale factor to make perturbations visible on screen
    
    advData.data[i] = Math.min(255, Math.max(0, cleanData.data[i] + sign * scaleVal));     // R
    advData.data[i+1] = Math.min(255, Math.max(0, cleanData.data[i+1] + sign * scaleVal * 0.8)); // G
    advData.data[i+2] = Math.min(255, Math.max(0, cleanData.data[i+2] + sign * scaleVal * 1.1)); // B
    advData.data[i+3] = 255;
  }
  ctxAdv.putImageData(advData, 0, 0);
  
  // 4. Apply Defense Post-Filter Preview
  const defData = ctxClean.createImageData(w, h);
  
  if (defenseType === "none") {
    // No defense, equal to adversarial
    ctxDef.putImageData(advData, 0, 0);
  } 
  else if (defenseType === "smoothing") {
    // 3x3 Average blur spatial filter
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            let px = x + kx;
            let py = y + ky;
            
            if (px >= 0 && px < w && py >= 0 && py < h) {
              let idx = (py * w + px) * 4;
              rSum += advData.data[idx];
              gSum += advData.data[idx+1];
              bSum += advData.data[idx+2];
              count++;
            }
          }
        }
        
        let targetIdx = (y * w + x) * 4;
        defData.data[targetIdx] = rSum / count;
        defData.data[targetIdx+1] = gSum / count;
        defData.data[targetIdx+2] = bSum / count;
        defData.data[targetIdx+3] = 255;
      }
    }
    ctxDef.putImageData(defData, 0, 0);
  } 
  else if (defenseType === "jpeg") {
    // Simulate JPEG Compression blocking (8x8 grid macroblocking quantization)
    const block = 8;
    for (let by = 0; by < h; by += block) {
      for (let bx = 0; bx < w; bx += block) {
        // Find average block color
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let y = by; y < Math.min(by + block, h); y++) {
          for (let x = bx; x < Math.min(bx + block, w); x++) {
            let idx = (y * w + x) * 4;
            rSum += advData.data[idx];
            gSum += advData.data[idx+1];
            bSum += advData.data[idx+2];
            count++;
          }
        }
        
        let avgR = rSum / count;
        let avgG = gSum / count;
        let avgB = bSum / count;
        
        // Write quantized values with block borders simulated slightly
        for (let y = by; y < Math.min(by + block, h); y++) {
          for (let x = bx; x < Math.min(bx + block, w); x++) {
            let idx = (y * w + x) * 4;
            
            // Blend average block color slightly with pixel detail to create a compressed look
            defData.data[idx] = avgR * 0.7 + advData.data[idx] * 0.3;
            defData.data[idx+1] = avgG * 0.7 + advData.data[idx+1] * 0.3;
            defData.data[idx+2] = avgB * 0.7 + advData.data[idx+2] * 0.3;
            defData.data[idx+3] = 255;
          }
        }
      }
    }
    ctxDef.putImageData(defData, 0, 0);
  } 
  else if (defenseType === "bit_depth") {
    // Reduct bit depth (3 bits = 8 discrete levels)
    const levels = 8 - 1;
    for (let i = 0; i < advData.data.length; i += 4) {
      defData.data[i] = Math.round((advData.data[i] / 255) * levels) * (255 / levels);
      defData.data[i+1] = Math.round((advData.data[i+1] / 255) * levels) * (255 / levels);
      defData.data[i+2] = Math.round((advData.data[i+2] / 255) * levels) * (255 / levels);
      defData.data[i+3] = 255;
    }
    ctxDef.putImageData(defData, 0, 0);
  }
  else if (defenseType === "adv_training") {
    // Dynamic simulation representing a robust CNN:
    // It recognizes the pattern regardless of the noise, pixels slightly enhanced
    for (let i = 0; i < advData.data.length; i += 4) {
      defData.data[i] = advData.data[i] * 0.9 + cleanData.data[i] * 0.1;
      defData.data[i+1] = advData.data[i+1] * 0.9 + cleanData.data[i+1] * 0.1;
      defData.data[i+2] = advData.data[i+2] * 0.9 + cleanData.data[i+2] * 0.1;
      defData.data[i+3] = 255;
    }
    ctxDef.putImageData(defData, 0, 0);
  }
}
