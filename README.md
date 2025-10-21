# ml-product-category-classifier

## Predicting product categories based on product titles using machine learning.

Automatically classify **product titles** into predefined **product categories** using machine learning.  
This project demonstrates an end-to-end ML workflow — from **data cleaning** and **feature engineering** to **model evaluation and selection**.

## Overview

- **Goal:** Predict the product category from its title.
- **Dataset:** 35,311 products (8 columns)
- **Best Model:** Random Forest (97% Accuracy)
- **Key Features:**  
  - TF-IDF vectorized product titles  
  - Numeric feature: *Title Length*

## Project Structure

├── data/
│ └── products.csv
├── notebooks/
│ └── product_category_classification.ipynb
├── src/
│ ├── train_models.py
│ └── predict_category.py
├── requirements.txt
└── README.md

##  Workflow

### 1 Data Cleaning
- Standardized column names and labels  
- Removed missing values  
- Unified inconsistent categories (e.g., `CPU` → `CPUs`)

**Important Decision:**  
 After testing both cleaned and raw versions of the dataset, I found that the **Random Forest model achieved slightly better accuracy on the *uncleaned* product titles**.  
>Therefore, I decided **not to clean the `Product Title` column** in the final pipeline, as the raw titles contained valuable token patterns that improved classification accuracy.

### 2 Feature Engineering
- Created a new feature: **Title Length**  
- Combined TF-IDF and numeric features via **ColumnTransformer**

### 3 Models Tested
| Model | Accuracy | Weighted F1 |
|--------|-----------|-------------|
| **Random Forest** | **0.97** | **0.97** |
| SVM (Linear) | 0.97 | 0.97 |
| Logistic Regression | 0.96 | 0.96 |
| Gradient Boosting | 0.95 | 0.95 |
| KNN | 0.95 | 0.95 |

 **Best Model:** Random Forest  
 **Feature Set:** Product Title + Title Length  
 **Pipeline:** TF-IDF + MinMaxScaler + Random Forest

 ## Evaluation

- Overall accuracy: **97%**  
- Balanced performance across major categories  
- Slightly lower recall for minority class (*Fridge Freezers*)

## How to Run

### Clone the repository
```bash
git clone https://github.com/yourusername/product-category-classification.git
cd product-category-classification