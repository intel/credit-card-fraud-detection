# Credit Card Fraud Detection Blueprint 

Fraud detection ref kit helps to boost detection accuracy and developer efficiency through Intel’s end-to-end, no-code, graph-neural-networks-boosted and multi-node distributed workflows. 

The blueprint utilizes Intel’s Enhanced Fraud Detection reference kit to capture complex behavioral patterns (e.g., fraudsters performing multiple small transactions from different cards to not get caught) and enrich the data features through Graph Neural Networks (GNN), and uses configuration-driven data-preprocessing and XGBoost-training of Classical Machine Learning workflow to efficiently build a distributed end-to-end machine learning pipeline to solve the fraud-detection task. Thus, it significantly improves the developer efficiency and boosts fraud-detection accuracy. 

## How to Use 

### [CNVRG IO FLows] 

> Note: you can experience the workflow step by step and view each step logs and results. 

* How to execute: Flows -> FraudDetection -> Click ‘Run’.

* How to view results: Experiments -> Click and Check each experiment result.
    ![1690943053146](https://github.com/intel/credit-card-fraud-detection/assets/1573221/e8f8f398-5578-4a18-a9ba-761b9126ad4c)

    For example, click the “xgb-training” to check the result:  
    
    <img width="1516" alt="image" src="https://github.com/intel/credit-card-fraud-detection/assets/1573221/ad21d1ec-df9f-4d9d-9baa-335c4d0318e8">

* Step-by-step explanation: 

   1) preprocess: seamlessly reads and parses the data to generate preprocessed data with features and the label - fraud or not fraud. 

    2) baseline-training: trains a binary classification model using the preprocess generated features. 

    3) gnn-analytics: creates homogenous graphs by consuming the processed data to learn the latent representations of the nodes. 

    4) xgb-training: trains a binary classification model using the GNN-enhanced data features. 
