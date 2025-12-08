# Company Layoff Risk Prediction

## Overview
This project predicts which companies have **high layoff risk** based on their layoff history, funding information, and industry metrics. The goal is to help investors and stakeholders identify unstable companies early.

## Project Structure
```
├── Data/
│   ├── layoffs_raw.csv              	# Original dataset from kaggle
│   ├── X_train.csv                  	# Training features (80%)
│   ├── X_test.csv                   	# Test features (20%)
│   ├── y_train.csv                  	# Training target
│   └── y_test.csv                  	# Test target
│
├── Notebooks/
│   ├── EDA.ipynb                   	# Exploratory Data Analysis
│   ├── Modeling.ipynb	              	# Model training & evaluation
│   └── Prediction_demo.ipynb        	# Prediction demonstration
│
├── App/
│   └── app.py                       	# Streamlit prediction app
│
├── Model/
│   ├── xgboost_model.pkl            	# Best XGBoost model
│   ├── decision_tree_model.pkl      	
│   └── random_forest_model.pkl      
│
├── summary/
│   └── project_summary.pdf         
│
├── Slides/
│   └── presentation.pdf             
│
└── README.md
```

## Dataset
- **Source**: https://www.kaggle.com/datasets/swaptr/layoffs-2022
- **Size**: 1,209 companies
- **Features**: 9 engineered features (industry, stage, funds_raised, region, recency, events_deviation, recency_deviation, layoff_events_category, funds_raised_binned)
- **Target**: high_risk (binary classification: 1 = high risk if risk_score >= 65th percentile, 0 = low risk)

## Results
**Best Model: XGBoost**
- Accuracy: 78.51%
- Precision: 67.37%
- Recall: 75.29%
- F1-Score: 0.7111
- AUC: 0.8375

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone or download the project
2. Navigate to the project directory:
```bash
   cd INFO\ 6105\ Final\ Project
```

3. Install required packages:
```bash
   pip install -r requirements.txt
```

## How to Run

### 1. Exploratory Data Analysis
```bash
cd Notebooks
jupyter notebook EDA.ipynb
```
This notebook loads the raw data, cleans it, performs feature engineering, and visualizes key distributions.

### 2. Model Training & Evaluation
```bash
jupyter notebook Train_and_Evaluate.ipynb
```
This notebook trains three models (Decision Tree, Random Forest, XGBoost), compares their performance, and saves the best model.

### 3. Make Predictions (Jupyter)
```bash
jupyter notebook Prediction_demo.ipynb
```
This notebook loads the trained XGBoost model and demonstrates predictions on the test set.

### 4. Interactive Prediction App (Streamlit)
```bash
cd ../App
streamlit run app.py
```
Upload a CSV file with the required features to get instant risk predictions. The app displays:
- Prediction results (High Risk / Low Risk)
- Confidence scores
- Industry information

**Required CSV columns:**
- company
- industry
- stage
- funds_raised
- region
- recency
- events_deviation
- recency_deviation
- layoff_events_category
- funds_raised_binned

## Model Details

### Best Model: XGBoost
**Optimal Parameters:**
- max_depth: 3
- learning_rate: 0.01
- n_estimators: 300
- subsample: 0.7
- colsample_bytree: 0.7

### Feature Importance
The model weighs the following factors:
- Total laid-off sum (30%)
- Max percentage laid off (30%)
- Layoff events count (20%)
- Layoff intensity (20%)

## Limitations
1. Missing features: company growth rate, CEO changes, market sentiment, etc.
2. Limited dataset size (1,210 companies) may restrict generalization ability.
3. Risk threshold (65th percentile) is arbitrary; alternative thresholds may provide better discrimination.

## Future Work
1. Collect additional features (revenue growth, CEO changes, hiring trends, news sentiment)
2. Combine models trained on different feature groups to improve stability
3. Add prediction confidence intervals instead of binary classification
4. Conduct fairness analysis across industries and regions

## Author
Yuwei Ni & Ruixuan Xu

## License
Educational purposes only.