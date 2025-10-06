# Customer Churn Prediction Analysis

This notebook implements a comprehensive customer churn prediction model using multiple machine learning algorithms including Logistic Regression, Random Forest, and Gradient Boosting.

## Dataset Overview
- **Dataset**: Churn_Modelling.csv
- **Size**: 10,000 customers with 14 features
- **Target Variable**: Exited (1 = Churned, 0 = Retained)
- **Class Distribution**: 79.63% retained, 20.37% churned

## Features:
- **Demographics**: Geography, Gender, Age
- **Financial**: CreditScore, Balance, EstimatedSalary, HasCrCard
- **Behavioral**: Tenure, NumOfProducts, IsActiveMember
- **Identifiers**: RowNumber, CustomerId, Surname

# Customer Churn Prediction Analysis

This project implements a comprehensive customer churn prediction model using multiple machine learning algorithms to identify customers at risk of leaving a subscription-based service.

## Overview

The analysis uses the `Churn_Modelling.csv` dataset containing 10,000 customer records with 14 features to predict customer churn. The target variable `Exited` indicates whether a customer has churned (1) or retained (0).

## Dataset Information

- **Size**: 10,000 customers
- **Features**: 14 (demographics, financial, behavioral)
- **Target Variable**: Exited (1 = Churned, 0 = Retained)
- **Churn Rate**: 20.37%

### Features:
- **Demographics**: Geography, Gender, Age
- **Financial**: CreditScore, Balance, EstimatedSalary, HasCrCard
- **Behavioral**: Tenure, NumOfProducts, IsActiveMember
- **Identifiers**: RowNumber, CustomerId, Surname

## Models Implemented

1. **Logistic Regression**: Linear model for binary classification
2. **Random Forest**: Ensemble method using multiple decision trees
3. **Gradient Boosting**: Sequential ensemble method for improved performance

## Key Features

### Data Preprocessing
- Removal of unnecessary identifier columns
- Label encoding for categorical variables
- Feature engineering with new derived features:
  - Age groups (Young, Middle, Senior, Elderly)
  - Balance to Salary ratio
  - Credit Score groups (Poor, Fair, Good, Excellent)
  - Tenure groups (New, Short, Medium, Long)
- Standard scaling for numerical features

### Model Evaluation
- Cross-validation with 5-fold CV
- Multiple metrics: Accuracy, Precision, Recall, F1-Score, AUC
- ROC curve analysis
- Confusion matrix visualization
- Feature importance analysis

### Business Insights
- Geography impact analysis
- Age-based churn patterns
- Activity status correlation
- Product usage patterns

## Files Structure

```
Churn/
├── Churn_Modelling.csv              # Original dataset
├── Customer_Churn_Prediction.ipynb  # Main analysis notebook
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── best_churn_model.pkl            # Saved best model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── geography_encoder.pkl           # Geography encoder (generated)
└── gender_encoder.pkl              # Gender encoder (generated)
```

## Installation and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis**:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

3. **Model Deployment**:
   The notebook includes a `predict_churn()` function for making predictions on new customer data.

## Key Findings

### High-Risk Customer Segments:
- Customers from Germany (highest churn rate)
- Senior customers (50+ years old)
- Inactive members
- Customers with 3-4 products (unusual pattern)

### Business Recommendations:
1. **Targeted Retention**: Focus on high-risk customer segments
2. **Engagement Programs**: Implement programs for inactive members
3. **Age-Specific Offerings**: Create products for different age groups
4. **Regional Campaigns**: Develop geography-specific marketing strategies

## Model Performance

The analysis compares all three models across multiple metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## Future Enhancements

1. **Hyperparameter Tuning**: Grid search optimization for best model
2. **Feature Selection**: Advanced feature selection techniques
3. **Ensemble Methods**: Combining multiple models for better performance
4. **Real-time Deployment**: API development for production use
5. **Model Monitoring**: Performance tracking and retraining strategies

## Technical Details

- **Python Version**: 3.8+
- **Key Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Cross-Validation**: 5-fold stratified CV
- **Train-Test Split**: 80-20 split with stratification
- **Feature Scaling**: StandardScaler for numerical features

## Contact

For questions or contributions, please refer to the project documentation or create an issue in the repository.

## Conclusion

This comprehensive analysis has successfully developed and evaluated multiple machine learning models for customer churn prediction. The key findings include:

1. **Dataset Analysis**: The dataset contains 10,000 customers with a 20.37% churn rate, providing a good balance for model training.

2. **Model Performance**: All three algorithms (Logistic Regression, Random Forest, and Gradient Boosting) achieved good performance, with the best model achieving high AUC scores.

3. **Key Insights**: Geography, age, activity status, and number of products are significant predictors of customer churn.

4. **Business Value**: The model can be deployed to identify high-risk customers and implement targeted retention strategies.

The analysis provides a solid foundation for implementing a customer churn prediction system in a real business environment.
