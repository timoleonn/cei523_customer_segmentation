import streamlit as st
import pandas as pd
import numpy as np

#   FIRST TAB
def dataPreperation():
    st.subheader("Data Preperation")
    st.write("Those are the inports we used.")
    st.code("imports...")
    st.write("We first load the data and we give some informations on the content of the dataframe such as the type of the various variables, the number of null values and their percentage with respect to the total number of entries.")
    st.code("code for data preperation")
    if st.button("Run Code for Data Preperation"):
        runCodeDataPreperation()


#run code function for data preparation
def runCodeDataPreperation():
    # here we will run the script for data preparation
    st.write("123")
    

#second tab
def secondTab():
    st.subheader("Exploring the Content of variables")
    st.write("This dataframe contains 8 variables that correspond to:\n\n"
            "**InvoiceNo:** Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.\n\n"
            "**StockCode:** Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.\n\n"
            "**Description:** Product (item) name. Nominal.\n\n"
            "**Quantity:** The quantities of each product (item) per transaction. Numeric.\n\n"
            "**InvoiceDate:** Invice Date and time. Numeric, the day and time when each transaction was generated.\n\n"
            "**UnitPrice:** Unit price. Numeric, Product price per unit in sterling.\n\n"
            "**CustomerID:** Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.\n\n"
            "**Country:** Country name. Nominal, the name of the country where each customer resides.")
    
    st.write("This is our code for this section")
    st.code("")
    if st.button("run code"):
        runCodeForSection2()
        

def runCodeForSection2():
    st.write("hello")


def thirdTab():
    st.subheader("Inside on product categories")

st.title('Customer Segmentation')

col1,col2,col3,col4 = st.columns(4)

st.sidebar.title("Contents")
radioBtn = st.sidebar.radio("Steps for Customer Segmentation", 
["Data Preparation","Exploring the content of variables","Inside on product categories",
"Customer Categories", "Classifying Customers", "Testing the predictions"])

if(radioBtn == "Data Preparation"):
    dataPreperation()
elif(radioBtn == "Exploring the content of variables"):
    secondTab()
elif(radioBtn == "Inside on product categories"):
    thirdTab()
elif(radioBtn == "Customer Categories"):
    st.write("4")
elif(radioBtn == "Classifying Customers"):
    st.write("5")
elif(radioBtn == "Testing the predictions"):
    st.write("6")


