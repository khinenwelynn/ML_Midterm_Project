# Midterm_Project_ML
Customer Churn Prediction App – Predicts whether a customer is likely to stay or leave using machine learning classification model.

### Dataset Overview

**Dataset source** : https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset 

The project uses a **customer churn dataset** with 11 features and 1 target column (`Churn`).
With this dataset, this project predict whether a customer will leave the service using classification model and deploy streamlit app.

### Features:

- **CustomerID** – Unique identifier for each customer.  
- **Age** – Age of the customer.  
- **Gender** – Customer’s gender.  
- **Tenure** – Number of months the customer has been using the service.  
- **Usage Frequency** – How often the customer used the service in the last month.  
- **Support Calls** – Number of calls the customer made to customer support in the last month.  
- **Payment Delay** – Days of delayed payment in the last month.  
- **Subscription Type** – The type of subscription the customer has chosen (e.g., Basic, Premium, Standard).  
- **Contract Length** – Duration of the customer’s contract (e.g., Monthly, Quarterly, Annual).  
- **Total Spend** – Total amount spent by the customer on the company’s services.  
- **Last Interaction** – Days since the customer’s last interaction with the company.  
- **Churn** – Binary label indicating if the customer has left (`1`) or stayed (`0`).

### File Overview:

All project files are in the same folder:

- churn_streamlit.py – Streamlit app for prediction.
- 01_churn.ipynb – Notebook for model development and exploration
- finalchurn.py – model development saved as python file
- customer_churn_dataset-training-master.csv – given dataset from source
- customer_churn_dataset-testing-master.csv – given dataset from source
- churn_model.pkl – Trained ML model 
- requirements.txt – Python dependencies
- Images/PUlogo.png – Logo used in the app
- .gitattributes – Git LFS tracking file
