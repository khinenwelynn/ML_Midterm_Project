import streamlit as st
import pandas as pd
import pickle
import os 

# Sidebar 

logo_path = "Images/PUlogo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown(":yellow[Student Name:**Khine Nwe Lin**]")
st.sidebar.markdown(":yellow[Student ID:**PIUS20230089**]")

# Load model
def load_model():
    with open("churn_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()
scaler = pipeline['scaler']
encoder = pipeline['encoder']
model = pipeline['model']
scale_cols = pipeline['scale_cols']
cat_cols = pipeline['cat_cols']

st.title(":blue[💠Customer Churn Prediction App]")

st.markdown(
    ":blue[This app predicts whether a customer is likely to **churn (leave)** or **stay** based on their behavior and subscription details.]"
    ":blue[Fill in the information below and click :gray-background[**Predict**]]."
)

st.markdown("---")

# User inputs
st.subheader(":gray[Enter Customer Information]")

with st.expander("Personal Information", expanded= True):
    col1,col2 = st.columns(2)
    with col1: 
        Age = st.number_input("Age", 18, 65, 24)
    with col2: 
        Gender= st.selectbox("Gender", ["Male", "Female"])

with st.expander("User Behaviors", expanded= True):
    col1,col2 = st.columns(2)
    with col1: 
        UsageFrequency = st.number_input("Monthly Usage", 0, 30, 10)
        
    with col2: 
        LastInteraction = st.slider("Days Since Last Interaction", 1, 30, 3)
    SupportCalls= st.number_input("Support Calls", 0, 10, 2)

with st.expander("User Subscription Details", expanded= True):
    col1,col2 = st.columns(2)
    with col1: 
        Tenure= st.number_input("Duration of a customer's active relationship in month", 0, 60, 12)
        
    with col2: 
        TotalSpend= st.number_input("Total Spend so far in USD ", 0.00, 1000.00, 500.00)
        
    PaymentDelay= st.slider("Payment Delay by customer (in days)", 0, 30, 0)
    SubscriptionType= st.selectbox("Subscription Type", ["Basic", "Premium", "Standard"])
    ContractLength= st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

input_data = {
    'Age': Age,
    'Tenure': Tenure,
    'Usage Frequency': UsageFrequency,
    'Support Calls': SupportCalls,
    'Payment Delay': PaymentDelay,
    'Total Spend': TotalSpend,
    'Last Interaction': LastInteraction,
    'Gender': Gender,
    'Subscription Type': SubscriptionType,
    'Contract Length': ContractLength
}
    
st.markdown("---")

# Predict button
if st.button("🔮 Predict"):
    Age = input_data['Age']
    Tenure = input_data['Tenure']
    Usage_Frequency = input_data['Usage Frequency']
    Support_Calls = input_data['Support Calls']
    Payment_Delay = input_data['Payment Delay']
    Total_Spend = input_data['Total Spend']
    Last_Interaction = input_data['Last Interaction']
    Gender = input_data['Gender']
    Subscription_Type = input_data['Subscription Type']
    Contract_Length = input_data['Contract Length']

    new_data = pd.DataFrame([{
        'Age': Age,
        'Tenure': Tenure,
        'Usage Frequency': Usage_Frequency,
        'Support Calls': Support_Calls,
        'Payment Delay': Payment_Delay,
        'Total Spend': Total_Spend,
        'Last Interaction': Last_Interaction,
        'Gender': Gender,
        'Subscription Type': Subscription_Type,
        'Contract Length': Contract_Length
    }])

    # Encode categorical features
    encoded_df = pd.DataFrame(
        encoder.transform(new_data[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=new_data.index
    )
    new_data = new_data.drop(columns=cat_cols)
    new_data = pd.concat([new_data, encoded_df], axis=1)

    # Scale numeric features
    new_data[scale_cols] = scaler.transform(new_data[scale_cols])

    # Predict churn
    prediction = model.predict(new_data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️This customer is likely to **churn**.")
        st.markdown(":red[Unfortunate..This customer is likely to leave our service.]")
    else:
        st.success("✅This customer is likely to **stay**.")
        st.markdown(":blue[Good News! This customer is likely to continue with our service.]")


    st.markdown("---")
    st.markdown(":rainbow[Machine Learning Midterm Project | Customer Churn Prediction App by *Khine Nwe Lin*].")
