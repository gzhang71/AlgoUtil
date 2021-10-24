# AlgoUtil

This is a demo repo to show how to process data, construct a scalarable machine learning class and do tests on it. The goal of this model is to extract model to extract a few factors from existing data, which will be used to compare the relative performance of stocks. Additionally, we can predict the distribution of trade price. 

Data is downloaded from [https://polygon.io/][PlDb] free tier subcription. The data includes: 
- Reference Data. 
- Trade Data. 
- Snapshot Data. 

Functions are provided to query data via WebSocket and RESTful API

Currently, the following topics in data exploration are demonstrated: 
- Data Exploring Using matplotlib and seaborn
- Interact with Data in Jupyter
- Animation to see market movement over time

The following machine learning models/algos are demonstrated. Additionally, a study note is attached to describe the key factors of this model.  
- PCA/PCR
- ARMA
- Lasso and Ridge
- ElasticNet
- Decision Tree
- Random Forest and ExtraTree
- SVM
- XGBoost
- LightGBM
- Neural Network Using TensorFlow
- Neural Network Using PyTorch

A few additional questions are discussed: 
- Output the importance of each factor
- Show directional impact of one factor
- Show MAD, MSE, R^2 for price prediction
- Show Entropy, AUC/ROC curve
