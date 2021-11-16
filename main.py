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

#   Read the dataframe
dataframe = pd.read_csv("data.csv",encoding='unicode_escape')

def runCodeDataPreperation():
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


#   SECOND TAB
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
    # Now we are going to explore the countries that our orders corresponds

    # We are groub the orders by invoice number and customer id to find how many orders we have from each country
    temp = dataframe[['CustomerID','InvoiceNo','Country']].groupby(['CustomerID','InvoiceNo','Country']).count()

    # We are reset the index to make the rows to a single column
    temp = temp.reset_index(drop = False)

    # We take and store to the variable countries the list of countries of the above dataframe which contains more than one column
    countries = temp['Country'].value_counts()

    # We st.write the column of countries
    st.write(countries)

    # Visualization of the Countries
    data = dict(type='choropleth',
    locations = countries.index,
    locationmode = 'country names', z = countries,
    text = countries.index, colorbar = {'title':'Order nb.'},
    colorscale=[[0, '#ff6b6b'],
                [0.01, '#ff7575'], [0.02, '#ff7a7a'],
                [0.03, '#ff3838'], [0.05, '#ff4242'],
                [0.10, '#ff2929'], [0.20, '#ff0f0f'],
                [1, '#9e0000']],    
    reversescale = False)
    layout = dict(title='Order per Country',
    geo = dict(showframe = True, projection={'type':'mercator'}))
    choromap = go.Figure(data = [data], layout = layout)
    iplot(choromap, validate=False)

    # We found all items
    items = len(dataframe['StockCode'].value_counts())

    # We found all orders
    orders = len(dataframe['InvoiceNo'].value_counts())

    # We found all customers
    customers = len(dataframe['CustomerID'].value_counts())

    # Initiate a dataframe with the above 3 columns and the word ammount as our metric
    numOfCustAndProd = pd.DataFrame([{'Items':items,'Orders':orders,'Customers':customers}], columns = ['Items', 'Orders', 'Customers'], index=['Amount'])
    st.write(numOfCustAndProd)

    # We found the products per Order because every order has many rows of products inside the dataframe
    productsPerOrder = dataframe.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()

    # We renamed the column invoice date as a number of products because then we would have the amount of products with column
    # name InvoiceDate
    productsPerOrder =productsPerOrder.rename(columns = {'InvoiceDate':'# of products'})
    st.write(productsPerOrder.sort_values('CustomerID').head())

    # We assume that the C in front of the InvoiceNo is the "Cancel" as we can see from the output that we take when we group
    # the purchases of each customer as the rows are the same and only the quantity is negative and the C character
    productsPerOrder['Canceled Order'] = productsPerOrder['InvoiceNo'].apply(lambda x: 1 if 'C' in x else 0)
    st.write(productsPerOrder.sort_values('CustomerID').head())
    st.write("The ammount of canceled orders is {}".format(productsPerOrder['Canceled Order'].sum()))
    st.write(dataframe.sort_values('CustomerID')[:10])

    # With this loop we want to examine the dataset and find the orders which are going to delete because its canceled
    # and the other if we have same recocords but the one has negative value and the other the same value but positive
    # We see that there are cancelled orders alone, cancelled orders with counterparts and other with various counterparts
    # We will run this code and take the results to show to the class because of its time which wants to run
    discount = []
    cancellation = []
    dataframeClean = dataframe.copy(deep = True)
    dataframeClean['QuantityCanceled'] = 0

    for index,row in dataframe.iterrows():
        if (row['Quantity'] > 0) or row['Description'] == 'Discount': 
            continue        
        check = dataframe[(dataframe['CustomerID'] == row['CustomerID']) &
                            (dataframe['StockCode']  == row['StockCode']) & 
                            (dataframe['InvoiceDate'] < row['InvoiceDate']) & 
                            (dataframe['Quantity']  > 0)].copy()
        # Cancelation WITHOUT counterpart
        if(check.shape[0] == 0):
            discount.append(index)
        # Cancelation WITH a counterpart
        elif (check.shape[0]==1):
            index_order = check.index[0]
            dataframeClean.loc[index_order, 'QuantityCanceled'] = -row['Quantity']
            cancellation.append(index)
        # Various counterparts exist in orders: we delete the last one        
        elif (check.shape[0]>1):
            check.sort_index(axis=0 ,ascending=False, inplace = True)        
            for ind, value in check.iterrows():
                if value['Quantity'] < -row['Quantity']: 
                    continue
                dataframeClean.loc[ind, 'QuantityCanceled'] = -row['Quantity']
                cancellation.append(index) 
                break

    print("Cancellation: {}".format(len(cancellation)))
    print("Discount: {}".format(len(discount)))

    # Here we check for the entries  of cancellations that we left from the process before
    dataframeClean.drop(cancellation, axis = 0, inplace = True)
    dataframeClean.drop(discount, axis = 0, inplace = True)
    remaining_entries = dataframeClean[(dataframeClean['Quantity'] < 0) & (dataframeClean['StockCode'] != 'D')]
    print("Number of entries to delete: {}".format(remaining_entries.shape[0]))
    remaining_entries[:5]

    # Stock Code which indicates a transaction type
    special_codes = dataframeClean[dataframeClean['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
    st.write(special_codes)

    # Basket Price
    dataframeClean['TotalPrice'] = dataframeClean['UnitPrice'] * (dataframeClean['Quantity'] - dataframeClean['QuantityCanceled'])
    st.write(dataframeClean.sort_values('CustomerID')[:5])

    # Here we sum the total price for each product which correspoonds to the same order
    temp = dataframeClean.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket = temp.rename(columns = {'TotalPrice':'Basket Price'})
    basket = basket[basket['Basket Price'] > 0]

    st.write(basket.head())

    # Here we take care of the minimum and maximum values to find ranges to present the percentage of orders that has a specific
    # ammount paid
    print("Maximum total price of order",basket['Basket Price'].max())
    print("Minimum total price of order",basket['Basket Price'].min())

    # Initialize the ranges to find how many total prices of orders are inside this ranges
    ranges = [0,50,100,200,500,1000,3000]
    countRanges = []
    for i, range1 in enumerate(ranges):
        if i == 0:
            continue
        val = basket[(basket['Basket Price'] < range1) &
            (basket['Basket Price'] > ranges[i-1])]['Basket Price'].count()
        countRanges.append(val)

    # Pie chart of the above results
    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    colors = ['red', 'yellow', 'orange', 'grey', 'blue', 'green','brown']
    labels = [ '{}<.<{}'.format(ranges[i-1], s) for i,s in enumerate(ranges) if i != 0]
    sizes  = countRanges
    explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
    ax.pie(sizes, explode = explode, labels=labels, colors = colors,
        autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
        shadow = False, startangle=0)
    ax.axis('equal')
    f.text(0.5, 1.01, "Répartition des montants des commandes", ha='center', fontsize = 18);


def thirdTab():
    st.subheader("Inside on product categories")

#   ===================
#   LEFT SIDEBAR COLUMN
#   ===================

st.title('Customer Segmentation')

col1, col2, col3, col4 = st.columns(4)

st.sidebar.title("Contents")
radioBtnOptions = [ "Data Preparation",
                    "Exploring the content of variables",
                    "Inside on product categories",
                    "Customer Categories",
                    "Classifying Customers",
                    "Testing the predictions" ]

radioBtn = st.sidebar.radio("Steps for Customer Segmentation", radioBtnOptions, 0)

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


