# **Convexity Bias Makes Languages Efficient**  
## Online Supplement  

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11355636.svg)](https://doi.org/10.5281/zenodo.11355636) -->

### Authors:  

## Overview  

This repository contains code and data for the study on convexity bias and its role in shaping the simplicity-informativeness trade-off. The study is based on the analysis of the Word Color Survey (WCS) data and computational modeling of linguistic biases affecting the simplicity-informativeness trade-off in artificial color lexicons.  

---

## Reproduction  

### 1. Downloading the Code & Requirements  

The code in this repository was executed using Python 3.11.2. To begin, clone the repository:  

```bash
git clone https://github.com/alexeykosh/2024-convexity-efficient-communication/
```

Navigate to the repository:  

```bash
cd 2024-convexity-efficient-communication
```

Install the required dependencies:  

```bash
pip install -r requirements.txt
```

---

### 2. Data  

#### Word Color Survey (WCS) Data  

The WCS dataset must be downloaded from the [WCS website](https://wcs.ijs.si/) and placed in the `data/` directory. After downloading, extract the data by running:  

```bash
python3 src/wcs_preprocessing.py 
```

#### Probability of Naming Data (Zaslavsky et al., 2018)  

For some analyses, additional data from Zaslavsky et al. (2018) is required. Download it from [this link](https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1) and place it in the `data/` directory.  

---

### 3. Analysis  

The following Jupyter notebooks contain the three main analyses reported in the paper:

- **Study 1: Degree of convexity explains the simplicity-informativeness trade-off**  
  - 📄 [`notebooks/analysis-main.ipynb`](https://github.com/alexeykosh/2024-convexity-efficient-communication/blob/main/notebooks/analysis-main.ipynb)  

- **Study 2: Replication using the Information Bottleneck framework**  
  - 📄 [`notebooks/analysis-IB.ipynb`](https://github.com/alexeykosh/2024-convexity-efficient-communication/blob/main/notebooks/analysis-IB.ipynb)  

- **Study 3: Rotation analysis of WCS languages**  
  - 📄 [`notebooks/analysis-rotation.ipynb`](https://github.com/alexeykosh/2024-convexity-efficient-communication/blob/main/notebooks/analysis-rotation.ipynb)  
 
