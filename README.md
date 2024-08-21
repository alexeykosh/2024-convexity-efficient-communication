# **Online supplement**: Convexity bias makes languages efficient.

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11355636.svg)](https://doi.org/10.5281/zenodo.11355636) -->

Authors: 


## Reproduction 

### Downloading the code & requirements:

The code provided in this repository was executed using Python 3.11.2. First, clone the repository:

```bash
git clone https://github.com/alexeykosh/2024-convexity-efficient-communication/
```

Then, navigate to the repository:

```bash
cd 2024-convexity-efficient-communication
```

All the required packages are listed in the `requirements.txt` file. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Data:

Word Color Survey data used in this study needs to be downloaded from the [WCS](https://wcs.ijs.si/) website. The downloaded zip file needs to be placed in the `data/` directory. 

After downloading the data, run the following command to extract the data:

```bash
python3 preprocessing.py 
```


### Analysis:

- [analysis.ipynb](https://github.com/alexeykosh/2024-convexity-efficient-communication/blob/main/analysis.ipynb) -- this notebook contains the code for the analysis of the Word Color Survey data (Study 1).
- [modelling.ipynb](https://github.com/alexeykosh/2024-convexity-efficient-communication/blob/main/modelling.ipynb) --  this notebook contains the code for the modelling of effects of different biases on simplicity-informativeness trade-off in artifical lexicons (Study 2).
