A predictive classification architecture designed to identify at-risk students early, equipping institutions with data to intervene proactively.

THE PROBLEM
Raw institutional data is highly imbalanced (skewed heavily towards non-dropouts) and spans multiple categorical dimensions, making manual multi-variate risk identification unreliable.

BUSINESS IMPACT
Demonstrates practical capabilities in mitigating severe class imbalances and establishes an automated early-warning system with direct socio-educational retention impact.

TECHNICAL STACK
Python
Random Forest
SMOTE
Scikit-learn
K-Fold CV
Streamlit
//
ARCHITECTURE & PROCESS
Implemented Synthetic Minority Over-sampling Technique (SMOTE) to balance the skewed dataset, preventing algorithmic bias.
Utilized K-Fold Cross-Validation to validate model integrity against overfitting.
Benchmarked Logistic Regression against Random Forests, configuring tree depths via RandomizedSearchCV.
Achieved a rigorous F1-score of ~82%, integrating the final inference pipeline into a clean Streamlit UI.
