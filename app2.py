import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import streamlit as st
import pandas as pd
import numpy as np
import pickle 


## load the trained model

model = tf.keras.models.load_model('regression_model.h5')

st.title("Churn Salary Prediction")

# Loading pickle files 


with open("label_encoder_reg_gender.pkl",'rb') as f:
    label_encoder_gender = pickle.load(f)

with open("one_hot_encoder_reg_geo.pkl",'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open("StandardScaler_reg.pkl",'rb') as f:
    scaler = pickle.load(f)


## taking input from the user 


## taking input from the user 

geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credite Score")
tenure = st.slider("Tenure",0,10)
num_of_Products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox("Is Active Member",[0,1])
exited = st.selectbox("Exited",[0,1])

# input_data = {
#     'CreditScore': 600,
#     'Geography': 'France',
#     'Gender': 'Male',
#     'Age': 40,
#     'Tenure': 3,
#     'Balance': 60000,
#     'NumOfProducts': 2,
#     'HasCrCard': 1,
#     'IsActiveMember': 1,
#     'Exited' : 0
# }
input_data = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Gender" : [gender],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_Products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_number],
    "Exited" : [exited],
    "Geography" : [geography]
})


input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])

df = one_hot_encoder_geo.transform(input_data[["Geography"]]).toarray()

df = pd.DataFrame(df,columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data,df],axis=1)

input_data.drop("Geography",axis = 1,inplace = True)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

predicted_salary = int(prediction[0][0])

st.write(f"The Estimated Salary of the person is {predicted_salary}")