# ThermVision: Exploring FLUX for Synthesizing Hyper-Realistic Thermal Face Data and Animations via Image to Video Translation

![Thermal Banner](static/images/teaser.gif)

ThermVision is a synthetically curated thermal imaging dataset designed to support research in facial analysis under infrared modalities. Unlike conventional thermal datasets captured through physical sensors, ThermVision leverages FLUX diffusion models, a class of high-fidelity generative architectures—to simulate photo-realistic thermal facial animations under controlled variations in head pose and facial expression.

The dataset is generated using a two-stage process:

1. Thermal Domain Synthesis with FLUX Diffusion: The core of this pipeline includes fine-tuning in thermal domain using a FLUX diffusion model trained on various subsets of real-world thermal face mappings. This results in high-resolution synthetic thermal sequences that preserve realistic temperature gradients, structural fidelity, and dynamic consistency across frames.

2. Video Retargeting Pipeline: Clean visible-spectrum facial videos of male and female subjects are processed using a retargeting framework to extract motion dynamics including pose shifts, gaze changes, and facial expressions (e.g., smile, frown, surprise).

Each sample in ThermVision consists of:

1. A sequence of synthetic thermal frames depicting continuous head movement or expression changes,

2. Corresponding pose and expression annotations,

3. Subject metadata (gender label, synthetic ID),

4. The dataset includes balanced representations of both male and female synthetic subjects, ensuring coverage across gender, expression types (neutral, smiling, surprised, etc.), and yaw/pitch/roll head rotations.

Applications supported by ThermVision include:

1. Thermal face detection and tracking,

2. Infrared expression recognition,

3. Facial animation learning in low-light or occluded scenarios.

By eliminating the need for costly thermal video capture while retaining realism and controllability, ThermVision enables scalable and reproducible research in thermal vision, particularly useful for surveillance, defense, health diagnostics, and privacy-preserving facial analytics.

---

## 🔬 Project Overview

ThermVision synthesizes high-fidelity thermal facial animations of male and female subjects under varying expressions and head poses. Key highlights include:

- ⚙️ Video retargeting of pose/expression sequences
- 🔥 FLUX-style diffusion-based thermal synthesis
- 🧠 Edge-conditioned ControlNet + GPT-4 for precise inpainting prompts
- 🎯 Targeted for thermal gender classification under realistic distortions

---

## 📁 Dataset Access

- [📥 ThermVision Sample Dataset](static/datasets/sample.zip)
- [📦 ThermVision Full Dataset (request access)](mailto:youremail@example.com)

---

## 🧩 ComfyUI Workflows

We provide two complete [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows for training and inference:

| Workflow | Preview | JSON File |
|----------|---------|-----------|
| **Training Pipeline** | ![Training](static/images/comfyui_training.png) | [Download `training_workflow.json`](static/workflows/training_workflow.json) |
| **Inference Pipeline** | ![Inference](static/images/comfyui_inference.png) | [Download `inference_workflow.json`](static/workflows/inference_workflow.json) |

These workflows are optimized for ControlNet + inpainting-based synthesis.

---

## 📊 Thermal Gender Classification Results

We evaluated multiple backbone architectures under both pure synthetic and hybrid training regimes:

![Classification Table](static/images/gender_classification_table.png)

**Datasets**: Tufts, CARL, SF-TL54, Charlotte-ThermalFace  
**Models**: MobileNet_v2, EfficientNet_b0, ResNet

---

## ⚠️ Bad Sample Results

While the majority of our generated images are structurally sound and realistic, failure cases still occur due to pose mismatches or prompt ambiguity:

| Failure Sample | Issue |
|----------------|-------|
| ![Failure 1](static/images/failure1.png) | Thermal noise hallucination, unrealistic temperature map |
| ![Failure 2](static/images/failure2.png) | Loss of facial structure under extreme yaw |

We are actively refining prompt guidance and ControlNet signals to reduce these artifacts.

---

## 📚 Paper & Citation

- [📄 Read on arXiv](https://arxiv.org/abs/your-paper-id)
- **BibTeX:**
```bibtex
@article{your2025thermvision,
  title={ThermVision: Diffusion-based Synthetic Thermal Facial Animation for Gender Classification},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
🤖 Authors & Contributions
Muhammad Ali Farooq – Project Lead, Model Development

Vaishali Lalit – Dataset Curation, Evaluation

John P. McCrae – Review, Technical Oversight

🛠️ How to Reproduce
Clone this repo

Open ComfyUI

Load the provided workflows

Insert input prompts and ControlNet signals

Generate thermal samples or fine-tune on classification task

📎 License
MIT License – Feel free to reuse, modify, and build upon this work for research and education purposes.

⭐ Show your support
Star this repo if you find it useful. Contributions and discussions are welcome!
