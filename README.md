Diabetes Prediction and Healthcare Data Analysis Project

This project focuses on analyzing a real-world healthcare dataset and predicting diabetes risk using machine learning techniques. The main goal of the project is to understand patterns in patient health data and build a predictive system that can identify the possibility of diabetes based on medical attributes.

The project includes data cleaning, exploratory data analysis, visualization, machine learning model training, and prediction using a Streamlit web application.

Project Objectives

- Analyze patient health data
- Identify important factors affecting diabetes
- Perform exploratory data analysis (EDA)
- Build a machine learning prediction model
- Visualize trends and correlations
- Apply data science concepts in a real-world healthcare domain

Dataset Information

The dataset contains patient medical and lifestyle details such as:

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level
- Diabetes Status

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

Project Files

- diabetes.csv → Dataset used for analysis
- eda_project.py → Exploratory Data Analysis code
- train_model.py → Machine learning model training
- model.pkl → Saved trained model
- app.py → Streamlit web application

Data Analysis Performed

- Checked missing values
- Removed duplicate records
- Generated statistical summaries
- Identified correlations between features
- Created visualizations using graphs and heatmaps

Visualizations Included

- Diabetes distribution charts
- Age distribution graphs
- BMI analysis
- Correlation heatmap
- Blood glucose level analysis
- Outlier detection using boxplots

Machine Learning Model

A Logistic Regression model was used to predict diabetes risk based on patient health data. The dataset was divided into training and testing data for model evaluation.

Model evaluation techniques used:

- Accuracy Score
- Confusion Matrix
- ROC Curve

Key Findings

- Higher blood glucose levels are strongly associated with diabetes
- HbA1c level is one of the most important indicators
- Patients with higher BMI show increased diabetes risk
- Older individuals are more likely to develop diabetes
- Hypertension and smoking habits also influence diabetes occurrence

Application

A Streamlit web application was developed where users can enter patient details and get diabetes risk predictions along with basic lifestyle recommendations.

How to Run the Project

Run EDA Analysis:
python eda_project.py

Train the Model:
python train_model.py

Run the Streamlit Application:
streamlit run app.py

Conclusion

This project demonstrates the practical application of data science and machine learning in the healthcare domain. It helps in understanding disease patterns, analyzing medical data, and building predictive systems for diabetes detection using real-world datasets.
