# AlgoUtil

This is a demo repo to show how to process data, construct a scalarable machine learning class and do tests on it. The goal of this model is to extract model to extract a few factors from existing data, which will be used to compare the relative performance of stocks. Additionally, we can predict the distribution of trade price. 

Raw data is publically avaiable from *[Polygon.io](https://polygon.io/)* free tier subcription. The data includes: 
- Reference Data. Stored in ticker_detail.pkl. It has the following columns: 
    - ticker
    - name (company name)
    - primiary_exchange
    - type (Industry type)
    - cik code (The Central Index Key (CIK) is used on the SEC's computer systems to identify corporations and individual people who have filed disclosure with the SEC.)
    - composite_figi (The Financial Instrument Global Identifier (FIGI) (formerly Bloomberg Global Identifier (BBGID)) is an open standard, unique identifier of financial instruments that can be assigned to instruments including common stock, options, derivatives, futures, corporate and government bonds, municipals, currencies, and mortgage products)
    - share_class_figi
    - outstanding_shares
    - market_cap
    - address
    - sic_code (Standard Industrial Classification (SIC) codes are four-digit numerical codes that categorize the industries that companies belong to based on their business activities. Standard Industrial Classification codes were mostly replaced by the six-digit North American Industry Classification System (NAICS).)
    - sic_description
    - ticker_root
    - ticker_suffix
    - base_currency_symbol
    
- Trade Data. Stored in ticker_price.pkl. It has the following columns: 
    - ticker
    - date
    - after_hours
    - high
    - low
    - open
    - close
    - pre_market
    - volume
- Snapshot Data. Stored in snapshot.pkl

Data will also be found via *[Google Drive](https://drive.google.com/drive/folders/1MbLXsBuGxRpc2FpT7w4tRRhMNtzDj3RH?usp=sharing)*

Functions are provided to query data via WebSocket and RESTful API

Currently, the following topics in data exploration are demonstrated: 
- Data Exploring Using matplotlib and seaborn
- Interact with Data in Jupyter
- Animation to see market movement over time

The following machine learning models/algos are demonstrated. Additionally, a study note is attached to describe the key factors of this model.  
- Linear Regression. Intercept vs non-intercept regression
- Logistic Regression. Draw AUC/ROC curve, precision-recall curve, and show some associated metrics 
- Regularization. Lasso, Ridge and ElasticNet. Made a plot to visualize the difference between different regularization approaches
- ARMA & ARIMA. Demonstrated unit test, ACF/PACF, and the residual diagnostic plot
- PCA. Showed the ways to read loadings, component variance and variance ratio; made a few plots leveraging plotly 
- SVM. 
- Decision Tree
- Random Forest and ExtraTree
- XGBoost
- LightGBM
- sklearn
- Neural Network Using TensorFlow
- Neural Network Using PyTorch

A few additional questions are discussed: 
- Output the importance of each factor
