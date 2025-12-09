# Estimating friction coefficients from synthetic video data

This project explores the estimation of friction coefficients from short video sequences using deep learning.
It combines large-scale synthetic data generation, distributed rendering, and sequence-based neural architectures to model a physical parameter traditionally difficult to infer from vision alone.

The work was completed during an engineering internship at **Moda Live**, where the objective was to automate the evaluation of material interactions in motion.

---

## Overview

The project is structured around three main components:

### 1. Synthetic Dataset Generation

A distributed rendering farm produces a multi-parameter video dataset simulating fabrics sliding on surfaces with varying materials and friction levels.

* Rendering tasks are dispatched to cloud workers through a lightweight orchestration layer using **Socket.IO**.
* Each worker loads a Houdini scene, renders a sequence, and assembles frames into a video.
* The system handles worker availability, task reassignment, progression monitoring and fault tolerance.

This automated generation pipeline enables the creation of consistent, high-volume synthetic data — essential for training models under a wide range of controlled parameters.

---

### 2. Learning to Predict Material Categories

The first model predicts the **material class** from a short video clip.
It acts as a conditioning signal for the friction-estimation model.

* Architecture: LRCN-style encoder (CNN + temporal pooling + LSTM).
* Input: 16 frames sampled from each rendered sequence.
* Output: Softmax distribution over material classes.
* Training includes early stopping, adaptive LR scheduling, and full training history logging.

This stage captures surface-specific visual cues that help disambiguate friction later on.

---

### 3. Conditional Friction Estimation

The second model estimates a discretized friction coefficient, **conditioned on the material distribution** predicted by the first model.

* Same video encoder as the material model.
* Late-fusion with the material probability vector.
* Multi-class classification over friction bins; evaluation includes both accuracy and RMSE on the continuous coefficient.

The conditional formulation significantly stabilizes training by separating two entangled sources of variation: material appearance and friction dynamics.

<div align="center">
  <img width="678" height="661" alt="image" src="https://github.com/user-attachments/assets/8fef5ee8-1466-4d81-b405-1f3b660290ae" />
</div>


---

## Dataset Structure

The dataset is organized so that each video sequence corresponds to a unique combination of:

* material class
* texture variant
* friction coefficient
* camera configuration

Each folder encodes its labels directly in its name (e.g. `cam1_mat3_text2_frict0.7`).

Both **raw image sequences** and **assembled MP4 videos** are supported by the loaders.

---

## Model Training & Evaluation

The `training/` directory contains:

* **dataset utilities** (parsing, frame sampling, generators)
* **model definitions** (video encoder, material model, conditional friction model)
* **training scripts** (with CSV logs, checkpointing, and metric summaries)
* **evaluation scripts** (classification reports, confusion matrices, RMSE computation)
* **plotting utilities** for training curves

All training artifacts (metrics, histories, curves, final models) can be saved in dedicated experiment folders for reproducibility and comparison.

---

## Results

The pipeline reproduces the main objectives of the internship:

* Robust classification of material type from short synthetic video sequences.
* Conditional prediction of friction coefficients with a **mean accuracy of ~82%** on the discretized bins.
* A corresponding **RMSE of ~0.12** when mapped back to the continuous coefficient range.

These results demonstrate that synthetic data, when generated with precise physical parameters and sufficient variability, can effectively train models to infer latent physical properties from vision.


---

## Repository Structure

```text
render_farm/          Distributed video rendering pipeline
training/             Dataset utilities, models, training & evaluation
report/               Project documentation and internship report
README.md             Overview of the project
```
* prepare a polished repository banner or diagram,
* or help you assemble the GitHub repo structure itself.
