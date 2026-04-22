# Messidor-2 — dataset acknowledgement + licence boilerplate

Copy-paste block for the paper's Acknowledgements / Data availability section.

## Acknowledgement paragraph (Acknowledgements section)

> The Messidor-2 colour fundus photographs used in this work were kindly provided by the Messidor program partners (see https://www.adcis.net/en/third-party/messidor2/). The original Messidor-2 image set was collected by Dr Bruno Lay et al. at Techniques en imagerie médicale (TIM) and released by ADCIS. The adjudicated 5-class DR grades used for evaluation in this paper were contributed by Google Research's Medical Brain team and released as the "messidor2-dr-grades" supplement on Kaggle.

## Citation block (References section — BibTeX-ready)

```bibtex
@article{decenciere2014messidor,
  title   = {{F}eedback on a publicly distributed image database: {T}he {M}essidor database},
  author  = {Decenci{\`e}re, Etienne and Zhang, Xiwei and Cazuguel, Guy and Lay, Bruno and Cochener, B{\'e}atrice and Trone, Caroline and Gain, Philippe and Ordonez, Richard and Massin, Pascale and Erginay, Ali and others},
  journal = {Image Analysis \& Stereology},
  volume  = {33},
  number  = {3},
  pages   = {231--234},
  year    = {2014},
  note    = {Original Messidor / Messidor-2 image release.}
}

@article{abramoff2013automated,
  title   = {Automated analysis of retinal images for detection of referable diabetic retinopathy},
  author  = {Abr{\`a}moff, Michael D and Folk, James C and Han, Dae Park and Walker, Jeffery D and Williams, David F and Russell, Stephen R and Massin, Pascale and Cochener, B{\'e}atrice and Gain, Philippe and Tang, Li and others},
  journal = {JAMA Ophthalmology},
  volume  = {131},
  number  = {3},
  pages   = {351--357},
  year    = {2013},
  note    = {Reference grading protocol used as the basis for subsequent Messidor-2 adjudicated releases.}
}

@misc{krause2018messidor2grades,
  title        = {Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy},
  author       = {Krause, Jonathan and Gulshan, Varun and Rahimy, Ehsan and Karth, Peter and Widner, Kasumi and Corrado, Greg S and Peng, Lily and Webster, Dale R},
  year         = {2018},
  howpublished = {Ophthalmology 125(8):1264--1272 (and Kaggle \texttt{google-brain/messidor2-dr-grades})},
  note         = {Source of the adjudicated 5-class DR labels used here.}
}
```

## Data-availability statement (if the venue requires a dedicated section)

> The APTOS 2019 Blindness Detection dataset is publicly available on Kaggle (https://www.kaggle.com/c/aptos2019-blindness-detection). The IDRiD dataset (Porwal et al., 2018) is available at https://idrid.grand-challenge.org/. The Messidor-2 colour fundus images were obtained from ADCIS under their research-use licence (https://www.adcis.net/en/third-party/messidor2/), and the adjudicated DR grades were obtained from the Google Brain Kaggle release (https://www.kaggle.com/google-brain/messidor2-dr-grades). All pre-trained RETFound weights are available from the authors of Zhou et al. 2023 (https://github.com/rmaphoh/RETFound_MAE). Code for this paper is available at https://github.com/vineetkumar-sudo/retfound-capsnet.

## Notes on usage restrictions

- **Messidor-2 images** are distributed by ADCIS under a research-only licence. **Do not redistribute** the raw images (keep `data/messidor2/IMAGES/` gitignored; we already do).
- **APTOS 2019** images are distributed under the competition's Kaggle terms — also research use only, no redistribution.
- **IDRiD** images are distributed under IEEE DataPort's terms (free research use after registration).
- **RETFound weights** are distributed under the licence in the RETFound repo (research use).

The published paper + its code can reference all four datasets by name / DOI without redistributing any raw pixel data.
