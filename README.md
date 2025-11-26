# SNN_Classification

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![GitHub stars](https://img.shields.io/github/stars/Ikhadija-5/SNN_Classification?style=social)

This repository contains code for **Eye Movement Classification** using **Spiking Neural Networks (SNNs)** and neuromorphic vision sensors. All models are implemented using the **LAVA framework**.  

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [DenseSNN](#densesnn)
  - [SpikingConvNet](#spikingconvnet)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

We provide two models for SNN-based classification:

1. **DenseSNN** – Dense fully-connected SNN architecture  
2. **SpikingConvNet** – Convolutional SNN for spatial feature extraction  

Each folder contains scripts for:

- `dataset.py` – Data loading and preprocessing  
- `network.py` – Network architecture  
- `train.py` – Training loop  
- `utils.py` – Helper functions  

All models and training routines use the **LAVA deep learning framework** for SNNs.  

---

## Installation

This project depends on the [LAVA framework](https://github.com/lava-nc/lava-dl/releases). Follow their instructions to install it, or quickly install from releases into a Python environment.  

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
