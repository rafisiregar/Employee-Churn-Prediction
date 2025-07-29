
# Employee Attrition Prediction Using Machine Learning

## Repository Outline

1. **P1M2_rafi_siregar.ipynb** – Notebook for analysis and model development.
2. **P1M2_rafi_siregar_inf.ipynb** – Notebook for model inference.
3. **P1M2_rafi_siregar.csv** – Kaggle employee attrition dataset.
4. **datainference.csv** – Inference model results dataset for model analysis.
5. **P1M2_rafi_siregar_conceptual.txt** – Conceptual Problem.
6. **eda_package.py** – Custom package for performing Exploratory Data Analysis (EDA).
7. **README.md** – Instructions for Milestone 2.
8. **description.md** – General project overview.
9. **AttritionBoosting.pkl** – Machine learning model used to predict employee attrition.
10. **Folder deployment** – Contains files for the deployment phase of the project.

## Problem Background

I am a data scientist tasked with developing a machine learning model to predict employee attrition. This project stems from the high attrition reports received by the company over the last year. High attrition can impact organizational stability, leading to increased recruitment costs, loss of knowledge and experience, and decreased morale among remaining employees. By predicting potential resignations, the company can take preventive actions, such as improving retention policies and employee satisfaction, to ensure the continued stability of the organization.

## Project Output

The objective is to build a machine learning model that can predict the likelihood of employee resignation based on historical data. Evaluation will be performed using classification metrics such as accuracy, precision, recall, and F1-score. This model will provide insights into the factors affecting employee attrition, such as age, job satisfaction, salary, and commute time. The project will also generate visualizations that show the distribution and relationships between variables. Based on this analysis, recommendations will be made to reduce attrition and improve employee retention, which can be implemented in company policies.

## Method

The approach used for building this model is **supervised machine learning**, applying algorithms such as **KNN**, **Boosting**, or **Random Forest**. Historical employee data, including features such as age, job satisfaction, salary, and others, will be used to train the model. The model will then be evaluated using classification metrics, especially **recall**, to measure its ability to predict employees at risk of resigning.

## Tech Stack

The project will utilize Python for data management and statistical analysis, with **Streamlit** and **HuggingFace** for real-time (online) model testing. In Python, **pandas** will be used to manage dataframes, **numpy** for numerical operations, **scipy** for statistical analysis, and **seaborn** and **matplotlib** for data visualization. **sklearn** will be used for data preprocessing, model algorithms, and evaluation metrics such as **recall**, **precision**, and **F1-score**, along with **RandomizedSearchCV** for hyperparameter tuning. For handling missing data, **KNNImputer** will be used, and various **scaling** techniques will be applied using **RobustScaler** to normalize features for better model performance. **OneHotEncoder** and **OrdinalEncoder** will handle categorical feature encoding. The model will be saved and loaded using **pickle**, while the **eda_package** will serve as a custom package for exploratory data analysis. **Streamlit** will be used to create an interactive web application that allows users to view visualizations and analysis results in real-time.

## Reference

1. [Employee Attrition Classification Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset): This dataset includes various features such as age, job satisfaction, salary, tenure, and more, which can be used to build a machine learning model to predict employee attrition.
2. [A Comprehensive Guide to Data Imputation Techniques, Strategies, and Best Practices](https://medium.com/@tarangds/a-comprehensive-guide-to-data-imputation-techniques-strategies-and-best-practices-152a10fee543): A comprehensive article on data imputation techniques and best practices to handle missing data.
3. **[ANALYSIS OF FACTORS AFFECTING EMPLOYEE RESIGNATION AT PT. SUMBER ALFARIA TRIJAYA TBK KOTA BANJARMASIN](https://eprints.uniska-bjm.ac.id/8005/1/ARTIKEL%20RATNA%20SARI%20FIX.pdf)**: This article discusses analysis and prediction related to employee attrition.
4. [Streamlit – Interactive Web Applications](https://huggingface.co/spaces/egar1444/EmployeeAttritionPrediction): Streamlit is used to create an interactive web application allowing users to view visualizations and analysis results directly from the machine learning model for employee attrition prediction.
