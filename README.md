# A Public Database of Thermoelectric Materials and System-Identified Material Representation for Data-Driven Discovery
Thermoelectric materials have received much attention for energy harvesting devices and power generators. However, discovering novel high-performance thermoelectric materials is a challenging task due to the diversity and the structural complexities of the thermoelectric materials containing alloys and dopants. For efficient data-driven discovery of novel thermoelectric materials, we constructed a public dataset that contains experimentally synthesized thermoelectric materials and their experimental thermoelectric properties. In our dataset, we achieved $R^2$-scores greater than 0.9 in the regression problems for predicting experimentally measured thermoelectric properties of the materials from their chemical compositions. Furthermore, we devised a material descriptor for the chemical compositions of the materials to improve extrapolation capabilities of machine learning methods. Based on transfer learning with the proposed material descriptor, we greatly improved $R^2$-score from 0.13 to 0.71 in predicting experimental ZTs of the materials from completely unseen material groups.

Reference: https://doi.org/10.xxxx/xxxxxxxxx

# Run
This repository provides an implementation of transfer learning based on System-Identified Material Representation (SIMD). By executing ``exec.py``, you can train and evaluate the ``XGBoost regressor`` with SIMD to predict ZTs of thermoelectric materials from unexplored material groups.

# Datasets
To reproduce the extrapolation results of SIMD, we should prepare the following two datasets of thermoelectric materials.
- **Starry dataset:** It is a large materials dataset containing thermoelectric materials. Since it was collected by text mining, data pre-processing should be conducted to remove invalid data (reference: https://www.starrydata2.org).
- **ESTM dataset:** It is a refined thermoelectric materials dataset for machine learning. ESTM dataset contains 5,205 experimental observations of thermoelectric materials and their properties (reference: https://doi.org/10.xxxx/xxxxxxxxx).

# Notes
- This repository contains only a subset of the source Starry dataset due to the dataset license. Please visit [Starrydata](https://www.starrydata2.org) to download the full data of the source Starry dataset.
- The full data of the ESTM dataset is provided in the ``dataset folder`` of this repository.
- The ``results folder`` provides the extrapolation results on the full data of the Starry and ESTM dataset. You can check the extrapolation results reported in the paper.
