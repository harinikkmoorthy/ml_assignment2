# Heart Disease Classification Project

## a. Problem Statement
Cardiovascular diseases are one of the leading causes of death globally. Early detection of heart disease can significantly improve patient outcomes by enabling timely medical intervention. Traditional diagnostic methods often require invasive procedures or extensive testing, which may not always be accessible.  

The objective of this project is to leverage machine learning techniques to predict the presence of heart disease based on patient attributes such as age, sex, chest pain type, blood pressure, cholesterol levels, and other medical indicators. By implementing and comparing multiple classification models, we aim to identify which algorithms perform best in terms of accuracy, precision, recall, F1 score, AUC, and Matthews Correlation Coefficient (MCC).  

This project demonstrates how data-driven approaches can support healthcare professionals in making faster, more reliable, and cost-effective decisions for heart disease diagnosis.

---

## b. Dataset Description
- **Source**: HeartDiseaseTrain-Test (1).csv (derived from the UCI Heart Disease dataset).  
- **Instances**: Patient records with medical attributes and diagnosis labels.  
- **Features**: 14 attributes (a mix of numeric and categorical variables).  
- **Target Variable**: Binary classification  
  - '0' → No heart disease  
  - '1' → Heart disease present  

### Key Features
- **Age**: Patient's age in years  
- **Sex**: Male/Female  
- **Chest Pain Type**: Typical angina, atypical angina, non-anginal pain, asymptomatic  
- **Resting Blood Pressure**: Measured in mm Hg  
- **Cholesterol**: Serum cholesterol in mg/dl  
- **Fasting Blood Sugar**: >120 mg/dl (binary: yes/no)  
- **Resting ECG Results**: Normal, ST-T wave abnormality, left ventricular hypertrophy  
- **Maximum Heart Rate Achieved**  
- **Exercise-Induced Angina**: Yes/No  
- **ST Depression (oldpeak)**: Depression induced by exercise relative to rest  
- **Slope of Peak Exercise ST Segment**: Upsloping, flat, downsloping  
- **Number of Major Vessels Colored by Fluoroscopy**: 0–3  
- **Thalassemia**: Normal, fixed defect, reversible defect  

### Observations from the Dataset
- The dataset is relatively balanced between patients with and without heart disease.  
- It contains both categorical and continuous variables, making it suitable for testing a wide range of classification algorithms.  
- Preprocessing steps include encoding categorical variables and scaling numerical features.  


## c. Models Used and Evaluation Metrics

### Comparison Table

| ML Model Name             | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|---------------------------|----------|------|-----------|--------|------|------|
| Logistic Regression       | 0.84     | 0.93 | 0.82      | 0.89   | 0.85 | 0.69 |
| Decision Tree             | 0.99     | 0.99 | 1.00      | 0.97   | 0.99 | 0.97 |
| kNN                       | 0.86     | 0.97 | 0.86      | 0.88   | 0.87 | 0.73 |
| Naive Bayes               | 0.84     | 0.91 | 0.83      | 0.88   | 0.85 | 0.69 |
| Random Forest (Ensemble)  | 1.00     | 1.00 | 1.00      | 1.00   | 1.00 | 1.00 |
| XGBoost (Ensemble)        | 1.00     | 1.00 | 1.00      | 1.00   | 1.00 | 1.00 |



## d. Observations on Model Performance

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | Provided a strong baseline with balanced accuracy and AUC. It generalizes well but may miss complex nonlinear relationships. |
| Decision Tree             | Achieved very high accuracy but shows signs of overfitting. Performance is excellent on the test set, but interpretability is limited compared to Logistic Regression. |
| kNN                       | Delivered solid results with good accuracy and AUC. Sensitive to scaling and the choice of k, but effective for capturing local patterns. |
| Naive Bayes               | Simple and fast, performed reasonably well despite independence assumptions. Slightly lower accuracy compared to ensemble methods. |
| Random Forest (Ensemble)  | Achieved perfect scores across all metrics, indicating strong ensemble learning. Robust against overfitting and captures feature interactions effectively. |
| XGBoost (Ensemble)        | Also achieved perfect scores, showing its strength in handling complex patterns. Highly effective and often the best performer in practice. |

