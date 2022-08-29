# A Public Database of Thermoelectric Materials and System-Identified Material Representation for Data-Driven Discovery
Thermoelectric materials have received much attention for energy harvesting devices and power generators. However, discovering novel high-performance thermoelectric materials is a challenging task due to the diversity and the structural complexities of the thermoelectric materials containing alloys and dopants. For efficient data-driven discovery of novel thermoelectric materials, we constructed a public dataset that contains experimentally synthesized thermoelectric materials and their experimental thermoelectric properties. In our dataset, we achieved $R^2$-scores greater than 0.9 in the regression problems for predicting experimentally measured thermoelectric properties of the materials from their chemical compositions. Furthermore, we devised a material descriptor for the chemical compositions of the materials to improve extrapolation capabilities of machine learning methods. Based on transfer learning with the proposed material descriptor, we greatly improved $R^2$-score from 0.13 to 0.71 in predicting experimental ZTs of the materials from completely unseen material groups.

Reference: https://doi.org/10.xxxx/xxxxxxxxx

# Run
This repository provides an implemetaion of transfer learning based on System-Identified Material Representation (SIMD). You can train and evaluate a XGBoostRegressor with SIMD to predict ZTs of thermoelectric materials from unexplored material groups.
