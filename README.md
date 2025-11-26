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
- [Dataset](#dataset)
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
```




# Eye Movement Classification Using Neuromorphic Vision
Sensors

If you are using this dataset in your paper, please **cite the following paper** : 

```
@article{iddrisu2024event,
  title={Event camera-based eye motion analysis: A survey},
  author={Iddrisu, Khadija and Shariff, Waseem and Corcoran, Peter and O'Connor, Noel E and Lemley, Joe and Little, Suzanne},
  journal={IEEE Access},
  volume={12},
  pages={136783--136804},
  year={2024},
  publisher={IEEE}
} 
```




## Dataset 

The dataset is an extension of 10 users from the EV-Eye dataset. The dataset was collected from 48 participants encompassing diverse genders and age group. Each participant participates in four sessions of data collection. The first tow sessions capture both saccade and fixation state of the movement, the last two sessions record eye movement in smooth pursuit. Moe Info about the dataset can be found in the official [EV-Eye Repository](https://github.com/Ningreka/EV-Eye.git). For our task, we manually annotated left eye data of 10 users into saccades and fixations. Download the preprocessed dataset from Figshare: [Ev-Eye-Fixations-Saccades](https://figshare.com/articles/dataset/Ev-eye_Dataset_Annotated_into_Saccades_and_Fixations/30722108/1?file=59868794)


After downloading, provide the dataset path when initializing the dataset class in your scripts:

```
from dataset import augment, EyeDataset

dataset_path = "path/to/downloaded/dataset"
dataset = EyeDataset(dataset_path)
```
The Raw unprocessed greyscale sequences can be found on the dataset page pn Figshare. To access more information about the data curation process and data characteristics, kindly refer to Section 3 of the corresponding paper.


## Running the Benchmark
To train the models provided, Simply navigate to the model you want to run with 

``` cd DenseSNN``` 
 and then run train.py. For example for DenseSNN, run the command

```
python train.py --network dense --trained_folder Dense_200 --batch_size 8 --max_timeStamps 200 --sample_length 200
```
or 
```
python train.py --network conv --trained_folder Dense_200 --batch_size 8 --max_timeStamps 200 --sample_length 200
```
to use SpikingConvnet

Notes:

--network can be dense or conv for DenseSNN or SpikingConvNet, respectively.

--trained_folder is the folder where trained weights will be saved.

--batch_size sets the training batch size.

--max_timeStamps and --sample_length control the temporal resolution. You can use the same values for both models to experiment consistently across temporal resolutions.


Contributing

Contributions are welcome! Please follow these steps:

-Fork the repository

-Create a new branch: git checkout -b feature-name

-Make your changes

-Commit your changes: git commit -m "Add feature"

-Push to the branch: git push origin feature-name

-Open a pull request




<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons
Attribution-NonCommercial 4.0 International License</a>.

