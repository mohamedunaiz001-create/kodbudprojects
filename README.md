# Kodbud Machine Learning Internship Project

## Overview

This repository contains the Machine Learning projects completed during the Kodbud ML Internship Program. The objective of this internship was to gain hands-on experience in applying machine learning algorithms to real-world problems such as prediction, classification, and clustering.

The internship focused on the complete machine learning workflow:

* Data Generation and Collection
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Model Training
* Model Evaluation
* Visualization of Results
* Performance Comparison

Five ML tasks were completed successfully using Python and Scikit-Learn.

---

##Technologies Used

* Python 3.10+
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

---

##Internship Learning Objectives

During this internship, the following concepts were implemented and understood:

### Regression

Used to predict continuous numerical values.

### Classification

Used to classify data into predefined categories.

### Clustering

Used to discover hidden groups within datasets.

### Model Evaluation

Used metrics such as:

* Accuracy
* R² Score
* ROC Curve
* AUC Score
* Confusion Matrix
* Silhouette Score

### Data Visualization

Created visual insights using:

* Histograms
* Correlation Heatmaps
* Scatter Plots
* ROC Curves
* Feature Importance Graphs

---

# Completed Tasks

---

## Task 1: House Price Prediction

### Objective

Predict the selling price of houses based on various property features.

### Dataset

Synthetic dataset containing:

* Area
* Bedrooms
* Bathrooms
* House Age
* Distance from City
* Garage Availability
* Number of Floors
* School Rating

### Models Used

* Linear Regression
* Ridge Regression
* Lasso Regression

### Skills Demonstrated

* Regression Analysis
* Feature Correlation
* Model Comparison
* Residual Analysis

### Outcome

* Achieved approximately **95% R² Score**
* Compared three regression algorithms
* Visualized actual vs predicted prices

---

## Task 4: Titanic Survival Prediction

### Objective

Predict whether a passenger survived the Titanic disaster.

### Dataset

Titanic dataset from Seaborn containing passenger information.

### Models Used

* Logistic Regression
* Random Forest Classifier

### Skills Demonstrated

* Missing Value Handling
* Feature Encoding
* Classification
* ROC Analysis

### Outcome

* Achieved approximately **80% Accuracy**
* Identified important survival factors:

  * Gender
  * Passenger Class
  * Age

---

## Task 6: Customer Segmentation

### Objective

Group customers into meaningful segments for marketing analysis.

### Dataset

Synthetic mall customer dataset containing:

* Annual Income
* Spending Score

### Model Used

* K-Means Clustering

### Skills Demonstrated

* Unsupervised Learning
* Elbow Method
* Silhouette Score Evaluation
* Cluster Visualization

### Outcome

* Determined optimal clusters using Elbow Method
* Achieved approximately **0.65 Silhouette Score**
* Identified customer groups such as:

  * Premium Customers
  * Budget Savers
  * Impulsive Buyers

---

## Task 7: Email Spam Detection

### Objective

Classify emails as Spam or Ham (Not Spam).

### Dataset

Labeled collection of email messages.

### Models Used

* TF-IDF Vectorizer
* Multinomial Naive Bayes

### Skills Demonstrated

* Natural Language Processing (NLP)
* Text Vectorization
* Spam Classification
* Probability Prediction

### Outcome

* Achieved approximately **95% Accuracy**
* High spam detection capability
* Live prediction testing performed

---

## Task 8: Diabetes Prediction

### Objective

Predict the likelihood of diabetes based on medical attributes.

### Dataset

PIMA-style patient dataset containing:

* Glucose
* Blood Pressure
* BMI
* Age
* Insulin
* Other clinical measurements

### Models Used

* Logistic Regression
* Random Forest Classifier

### Skills Demonstrated

* Medical Data Analysis
* Binary Classification
* Feature Importance Analysis
* ROC and Precision-Recall Evaluation

### Outcome

* Achieved approximately **95% Accuracy**
* Glucose and BMI identified as the most influential features

---

# Project Structure

```text
Kodbud_ML_Internship/
│
├── ml_task1_house_price.py
├── ml_task4_titanic.py
├── ml_task6_customer_seg.py
├── ml_task7_spam.py
├── ml_task8_diabetes.py
│
├── task1_plots/
├── task4_plots/
├── task6_plots/
├── task7_plots/
├── task8_plots/
│
└── README.md
```

---

# Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# Running the Projects

```bash
python ml_task1_house_price.py
python ml_task4_titanic.py
python ml_task6_customer_seg.py
python ml_task7_spam.py
python ml_task8_diabetes.py
```

---

# Key Skills Gained

* Machine Learning Fundamentals
* Data Analysis
* Feature Engineering
* Regression Models
* Classification Models
* Clustering Techniques
* Natural Language Processing
* Model Evaluation
* Data Visualization
* Python Programming

---

# Internship Outcome

This internship provided practical experience in building complete machine learning solutions from data preparation to model deployment-ready evaluation. The projects demonstrate the ability to solve real-world business and analytical problems using machine learning techniques and industry-standard Python libraries.

**Intern:** Mohamed Unaiz A
**Organization:** Kodbud Technologies
**Domain:** Machine Learning Internship
**Tasks Completed:** 1, 4, 6, 7, 8

This version looks professional for recruiters, GitHub visitors, LinkedIn, and internship evaluators.
