#   STREAMLIT IMPORT
import streamlit as st

#   PROJECT IMPORTS
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from IPython.display import display
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import matplotlib.patches as mpatches
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

#   FIRST TAB
def dataPreperation():
    st.subheader("Data Preperation")
    st.write("Those are the inports we used.")
    st.code("imports...")
    st.write("We first load the data and we give some informations on the content of the dataframe such as the type of the various variables, the number of null values and their percentage with respect to the total number of entries.")
    st.code("code for data preperation")
    if st.button("Run Code for Data Preperation"):
        runCodeDataPreperation()


#   RUN CODE FOR DATA PREPARATION 
def runCodeDataPreperation():
    # Read the dataframe
    dataframe = pd.read_csv("data.csv",encoding='unicode_escape')

    # Make the country and customerID values strings
    dataframe['Country']=dataframe['Country'].astype('string')
    dataframe['CustomerID']=dataframe['CustomerID'].astype('string')

    # See how many rows and columns has the dataframe
    st.write('Dimensions:', dataframe.shape)

    # Make the date time from object to date time
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])

    # Make a dataframe to see the types of the columns we have
    dataframeTypeInfo=pd.DataFrame(dataframe.dtypes).T.rename(index={0:'Column Type'})

    # Append the sum of null values for each column
    dataframeTypeInfo=dataframeTypeInfo.append(pd.DataFrame(dataframe.isna().sum()).T.rename(index={0:'Null values'}))

    # Print the dataframe implemented above
    st.dataframe(dataframeTypeInfo.astype(str))

    # Print the initial dataframe head ( 5 rows )
    st.dataframe(dataframe.head().astype(str))
    
    # Find the values of invoice numbers which CustomerID is not null
    customerIDFilled = dataframe['InvoiceNo'].where(~dataframe['CustomerID'].isna())

    # Find the values of invoice numbers which CustomerID is null
    customerIDEmpty = dataframe['InvoiceNo'].where(dataframe['CustomerID'].isna())

    # Run a small programm to see if there are common values so we can find the null CustomerID
    check = []
    for id in customerIDFilled.unique():
        if id in customerIDEmpty:
            check.append(id)

    # If the size of check is 0, which is , we dont have any common values so we cant  fill any null CustomerID so drop them
    # We do this because each product has its own row on the dataset and propably there is orders which only one item of them
    # has the CusteomerID
    if (len(check)==0):
        st.write('We dont have any related data into the 2 lists of ids so we drop the empty CustomerIDs')
    else:
        st.write(check)
    
    # From the dataframe we drop the rows that the CustomerID = null
    dataframe.dropna(axis = 0, subset = ['CustomerID'], inplace = True)

    # We make the CustomerID to float and then to int to avoid errors 
    dataframe['CustomerID']=dataframe['CustomerID'].astype('float') 
    dataframe['CustomerID']=dataframe['CustomerID'].astype('int') 

    # We check for duplicate values inside the our dataframe 
    st.write('Duplicates: {}'.format(dataframe.duplicated().sum()))

    # Duplicates are not usefull data so we drop them
    dataframe.drop_duplicates(inplace = True)

    if st.button("Run Code for Exploring the content of variables"):
        secondTab()



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


#   LEFT SIDEBAR COLUMN

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


