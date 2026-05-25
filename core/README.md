# Deep Learning Core: Adversarial Attacks and Defense Mechanisms

This directory houses the PyTorch implementation of the Deep Learning pipeline. It trains standard and robust CNN networks, crafts adversarial images via **Fast Gradient Sign Method (FGSM)**, and benchmarks multi-tier defenses (Adversarial Training, Spatial Smoothing, JPEG Compression, and Bit-Depth Quantization).

---

## Directory Layout

*   `requirements.txt`: Python package requirements (`torch`, `torchvision`, `numpy`, `matplotlib`, `pillow`).
*   `model.py`: Double-configurable CNN (`SimpleCNN`) supporting MNIST and CIFAR-10 classification.
*   `attack.py`: Mathematics-grounded implementations of the **FGSM** and **Projected Gradient Descent (PGD)** adversarial algorithms.
*   `defense.py`: Preprocessing filters and the full **Adversarial Training** gradient optimization loop.
*   `run_suite.py`: Master pipeline execution script which trains models, benchmarks accuracy under various perturbation levels ($\epsilon$), and generates synchronization files for the frontend dashboard.

---

## Getting Started

Follow these steps to run the PyTorch evaluation pipeline on your local machine:

### 1. Initialize Virtual Environment (Recommended)
Open a terminal in the root of the project directory and create a standard Python virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or Windows Command Prompt
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies
Install all required Deep Learning and plotting dependencies via pip:

```bash
pip install -r core/requirements.txt
```

### 3. Run the Suite
Execute the master runner to train the model, evaluate the robustness metrics, and export files:

```bash
python core/run_suite.py
```

---

## Output Synchronization

Running `run_suite.py` creates the following crucial assets:

1.  **`results.json`** (placed in the root folder): Stores exact, detailed precision accuracy coordinates across standard and adversarially trained models for multiple epsilon intensities, alongside spatial filters benchmark measurements.
2.  **Visualization Assets** (placed in `assets/images/`):
    *   `sample_original.png`: The baseline clean CIFAR-10/MNIST sample.
    *   `sample_gradient.png`: The sign of the calculated loss gradient with respect to the input pixels ($\text{sign}(\nabla_x L(\theta, x, y))$).
    *   `sample_adversarial.png`: The adversarial image after applying FGSM ($\epsilon = 0.15$).
    *   `sample_defended.png`: The resulting image after applying JPEG preprocessor filtering defense.
