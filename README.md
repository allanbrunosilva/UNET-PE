# UNET Segmentation for Pulmonary-Embolism-Detection on CT Scan Images using TensorFlow 2.0 
The repository contains the code for UNET segmentation on CT scan dataset in TensorFlow 2.0 framework.

# Overview
- Dataset
- Setup

# Dataset
The dataset contains the CT scan image and their respective binary mask.

Download the dataset: [CAD-PE](https://ieee-dataport.org/open-access/cad-pe)

Ensure that the `images` and `rs` folders, extracted from the dataset, are placed inside the `./data/00_original/CAD-PE directory`.

# Setup

Initialize a new environment using Python 3.11
`py -3.11 -m venv <path_to_your_environment>`

Install the necessary libraries
`pip install -r requirements.txt`

Use the `main.ipynb` file to prepare the dataset and train the model.
