# DATA1030_MIDTERM_PROJECT

Contact: bangxi_xiao@brown.edu

This project made use of a credit default dataset from Kaggle.com. Multiple famous machine learning models are built to predict the credit default situations, such as Logistic regression, random forest and extreme gradient boosting model. Also, please note that the XGB (extreme gradient boosting model) is trained with GPU (the booster is gpu_hist), which is only allowed in Windows and linux systems (not mac). 

All the resulting figures are in the figures file and the model scripts are stored in the src file. 

The finalized overall accuracy of the model is 0.71 and the recall is 0.67.

Requirements:
  - python=3.8
  - matplotlib=3.3.4
  - pandas=1.2.4
  - scikit-learn=1.0.1
  - numpy=1.20.1
  - imblearn 0.0
  - imbalanced-learn 0.8.1
  - xgboost=1.3.3
  - shap=0.40.0
