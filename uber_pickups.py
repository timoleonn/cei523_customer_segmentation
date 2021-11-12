import streamlit as st
import pandas as pd
import numpy as np

def dataPreperation():
    st.header("Data Preperation")
    st.write("Those are the inports we used.")
    st.code("code for data preperation")



st.title('Customer Segmentation')

col1,col2,col3,col4 = st.columns(4)

st.sidebar.title("Contents")
radioBtn = st.sidebar.radio("Steps for Customer Segmentation", 
["Data Preparation","Exploring the content of variables","Inside on product categories",
"Customer Categories", "Classifying Customers", "Testing the predictions"])

if(radioBtn == "Data Preparation"):
    dataPreperation()
elif(radioBtn == "Exploring the content of variables"):
    st.write("2")
elif(radioBtn == "Inside on product categories"):
    st.write("3")
elif(radioBtn == "Customer Categories"):
    st.write("4")
elif(radioBtn == "Classifying Customers"):
    st.write("5")
elif(radioBtn == "Testing the predictions"):
    st.write("6")


