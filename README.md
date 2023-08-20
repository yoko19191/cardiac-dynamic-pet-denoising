# Exploration of self-supervised deep denoising methods of Dynamic PET Image

## Overview

PET (Positron Emission Tomography) is an advanced cardiac imaging modality for studying cardiac functionality and blood flow dynamics. This project aims to explore the impact of various sub-supervised denoising techniques on Cardiac Dynamic PET images to optimize image clarity and analytical accuracy.

## Installation

```
git clone git@github.com:yoko19191/cardiac-dynamic-pet-denoising.git
cd cardiac-dynamic-pet-denoising
virtualenv .env --python=python3.11
cd .env
source /bin/activate 
pip install -r requirements.txt
```

## Compared methods

- [X] Block-Matching and 4D filtering
- [ ] Deep Image Prior(DIP)
- [ ] Noise2Void(N2V)
- [ ] Noise-As-Clean(NAC)
- [ ] Neighbor2Neighbor(Nb2Nb)
- [ ] Zero-shot Noise2Noise(ZS-N2N)
- [ ] Proposed ZS-N2N variant

## Metrics

- [Peak signal-to-noise ratio(PSNR)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [Structural similarity(SSIM)](https://en.wikipedia.org/wiki/Structural_similarity)
- [Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) ](https://archive.is/20201213161243/https://towardsdatascience.com/automatic-image-quality-assessment-in-python-391a6be52c11#selection-931.0-931.61)
- Time-activity curve(TAC)

## Dataset

you may downlaod the example dataset [here](https://zenodo.org/record/6580182)

## Experiment

## Result

PSNR

SSIM

BRISQUE

TAC

## Acknowledge
