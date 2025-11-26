

# SNN_Classification

This repository contains code for **Eye Movement Classification** using **Spiking Neural Networks** (SNNs) and neuromorphic vision sensors.

## Project Overview

We provide two models for classification:

1. **DenseSNN** – Dense fully-connected SNN architecture  
2. **SpikingConvnet** – Convolutional SNN for spatial feature extraction

Each folder contains scripts for:
- `dataset.py` – data loading and preprocessing  
- `network.py` – network architecture  
- `train.py` – training loop  
- `utils.py` – helper functions

---

## Installation

# SNN_Classification
This repo contains code for Eye Movement Classification Using Neuromorphic Vision Sensors


## Installation
This project uses the LAVA Framework form link. Install via the instruction on the official LAVA Github Repo

[Lavadl Repo]([https://github.com/lava-nc/lava-dl/releases](https://github.com/lava-nc/lava-dl)).


(Optional) Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


## Install dependencies:
pip install -r requirements.txt


#Usage
-python DenseSNN/train.py
-python SpikingConvnet/train.py


1. Clone the repository:
```bash
git clone https://github.com/Ikhadija-5/SNN_Classification.git
cd SNN_Classification
