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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
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

import csv

#   FIRST TAB
def dataPreperation():
    st.subheader("Data Preperation")
    st.write("Those are the inports we used.")
    st.code("imports...")
    st.write("We first load the data and we give some informations on the content of the dataframe such as the type of the various variables, the number of null values and their percentage with respect to the total number of entries.")
    if st.button("Prepare Data"):
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

    # Save Dataframe to csv so we can use in other sections
    dataframe.to_csv('newDataframe.csv')


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
    # Read Dataframe so we can use in other sections
    dataframe = pd.read_csv("newDataframe.csv",encoding='unicode_escape')

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
    st.plotly_chart(choromap)

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

    st.write("Cancellation: {}".format(len(cancellation)))
    st.write("Discount: {}".format(len(discount)))

    # Here we check for the entries  of cancellations that we left from the process before
    dataframeClean.drop(cancellation, axis = 0, inplace = True)
    dataframeClean.drop(discount, axis = 0, inplace = True)
    remaining_entries = dataframeClean[(dataframeClean['Quantity'] < 0) & (dataframeClean['StockCode'] != 'D')]
    st.write("Number of entries to delete: {}".format(remaining_entries.shape[0]))
    remaining_entries[:5]

    # Stock Code which indicates a transaction type
    special_codes = dataframeClean[dataframeClean['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
    st.write(special_codes)

    # Basket Price
    dataframeClean['TotalPrice'] = dataframeClean['UnitPrice'] * (dataframeClean['Quantity'] - dataframeClean['QuantityCanceled'])
    st.write(dataframeClean.sort_values('CustomerID')[:5])

    # Save Dataframe to csv so we can use in other sections
    dataframeClean.to_csv('dataframeClean.csv')

    # Here we sum the total price for each product which correspoonds to the same order
    temp = dataframeClean.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket = temp.rename(columns = {'TotalPrice':'Basket Price'})
    basket = basket[basket['Basket Price'] > 0]

    st.write(basket.head())

    # Here we take care of the minimum and maximum values to find ranges to present the percentage of orders that has a specific
    # ammount paid
    st.write("Maximum total price of order",basket['Basket Price'].max())
    st.write("Minimum total price of order",basket['Basket Price'].min())

    # Initialize the ranges to find how many total prices of orders are inside this ranges
    ranges = [0,50,100,200,500,1000,3000]
    countRanges = []
    for i, range1 in enumerate(ranges):
        if i == 0:
            continue
        val = basket[(basket['Basket Price'] < range1) &
            (basket['Basket Price'] > ranges[i-1])]['Basket Price'].count()
        countRanges.append(val)

    # Save Dataframe to csv so we can use in other sections
    dataframe.to_csv('newDataframe.csv')

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
    
    st.pyplot(f)


#   ==========
#   THIRD TAB
#   ==========
def thirdTab():
    st.subheader("3. Insight on product categories")
    
    if st.button("run code"):
        return runCodeForSection3()

def runCodeForSection3():
    # Read Dataframe so we can use in other sections
    dataframe = pd.read_csv("newDataframe.csv",encoding='unicode_escape')
    dataframeClean = pd.read_csv("dataframeClean.csv",encoding='unicode_escape')

    # Here we have the function that we are going to use to find usefull information from the description of each product
    def keywords(dataframe):
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots  = dict()  
        keywords_select = dict()  
        category_keys   = []
        count_keywords  = dict()
        for description in dataframe['Description']:
            if pd.isnull(description): continue
            line = description.lower()
            tokenized = nltk.word_tokenize(line)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos[:2]=='NN'] 
            
            for w in nouns:
                w = w.lower() 
                root = stemmer.stem(w)
                if root in keywords_roots:                
                    keywords_roots[root].add(w)
                    count_keywords[root] += 1                
                else:
                    keywords_roots[root] = {w}
                    count_keywords[root] = 1
        
        for value in keywords_roots.keys():
            if len(keywords_roots[value]) > 1:  
                min_length = 1000
                for k in keywords_roots[value]:
                    if len(k) < min_length:
                        selected = k 
                        min_length = len(k)            
                category_keys.append(selected)
                keywords_select[value] = selected
            else:
                category_keys.append(list(keywords_roots[value])[0])
                keywords_select[value] = list(keywords_roots[value])[0]

        return category_keys, keywords_roots, keywords_select, count_keywords

    # We are taking the unique descriptions to avoid the duplicates
    dataframeProducts = pd.DataFrame(dataframe['Description'].unique()).rename(columns = {0:'Description'})

    # We are running the function we constructed above to fing the keywords
    keywords, keywords_roots, keywords_select, count_keywords = keywords(dataframeProducts)

    # Here we convert the dictionary to list and sort the keywords from the larger occurrence number of keyword to the smaller
    list_products = []
    for key,value in count_keywords.items():
        list_products.append([keywords_select[key],value])
    list_products.sort(key = lambda x:x[1], reverse = True)
    list1 = sorted(list_products, key = lambda x:x[1], reverse = True)

    # Here we make a sketch to see the 50 keywords with larger occurence number
    plt.rc('font', weight='normal')
    fig, ax = plt.subplots(figsize=(7, 25))
    y_axis = [i[1] for i in list1[:50]]
    x_axis = [k for k,i in enumerate(list1[:50])]
    x_label = [i[0] for i in list1[:50]]
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 13)
    plt.yticks(x_axis, x_label)
    plt.xlabel("Number of occurences of first 50 keywords", fontsize = 8, labelpad = 8)
    ax.barh(x_axis, y_axis, align = 'center')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 8)
    # plt.show()
    st.pyplot(fig)

    # Here we remove some words that we saw inside the keywords and was colours or had special characters and keep only the
    # keywords which is larger than 3 characters to avoid some spowords and keywords that appears more than 13 times
    new_list_products = []
    for key,value in count_keywords.items():
        word = keywords_select[key]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']:
            continue
        if len(word) < 3 or value < 13:
            continue
        if ('+' in word) or ('/' in word):
            continue
        new_list_products.append([word, value])  
    new_list_products.sort(key = lambda x:x[1], reverse = True)
    st.write('Remaining words: ',len(new_list_products))

    # We used the keywrods to create groups of products and encode our data with 1 hot encoding
    description1 = dataframeClean['Description'].unique()
    X = pd.DataFrame()
    for key, occurence in new_list_products:
        X.loc[:, key] = list(map(lambda x:int(key.upper() in x), description1))

    # We check for how many clusters is better for our data base on silouette score which is show us 
    # if the clusters are well apart from each other and clearly distinguished
    matrix = X.values
    for n_clusters in range(3, 10):
        kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)
        st.write("For n_clusters = ", n_clusters, "The average silhouette_score is:", silhouette_avg)

    # We decided to take 5 clusters because when we chose to proceed with more cluster we saw that many clusters had only a few
    # words and if we choose less than 3 we will go to binary classification and we dont want it
    # Here we are itterating the process until we obtain a good sillouette average which is arround 0.1+-0.05
    n_clusters = 5
    silhouette_avg = -1
    while silhouette_avg < 0.145:
        kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)
        
        st.write("For n_clusters = ", n_clusters, "The average silhouette_score is: ", silhouette_avg)

    st.write('Words in clusters\n',pd.Series(clusters).value_counts())

    liste = pd.DataFrame(description1)
    liste_words = [word for (word, occurence) in new_list_products]

    occurence = [dict() for _ in range(n_clusters)]

    for i in range(n_clusters):
        liste_cluster = liste.loc[clusters == i]
        for word in liste_words:
            if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
            occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))
            
            

    #________________________________________________________________________
    def random_color_func(word=None, font_size=None, position=None,
                        orientation=None, font_path=None, random_state=None):
        h = int(360.0 * tone / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
        return "hsl({}, {}%, {}%)".format(h, s, l)
    #________________________________________________________________________
    def make_wordcloud(liste, increment):
        new_figure = plt.figure()
        ax1 = new_figure.add_subplot(4, 2, increment)
        # ax1 = fig.add_subplot(4,2,increment)
        words = dict()
        trunc_occurences = liste[0:150]
        for s in trunc_occurences:
            words[s[0]] = s[1]
        #________________________________________________________
        wordcloud = WordCloud(width=1000, height=400, background_color='lightgrey', 
                            max_words=1628, relative_scaling = 1,
                            color_func = random_color_func,
                            normalize_plurals = False)
        wordcloud.generate_from_frequencies(words)
        ax1.imshow(wordcloud, interpolation = "bilinear")
        ax1.axis('off')
        plt.title('cluster nº{}'.format(increment - 1))
        st.pyplot(ax1.figure)  
    #________________________________________________________________________
    fig = plt.figure(1, figsize=(14,14))
    color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
    for i in range(n_clusters):
        list_cluster_occurences = occurence[i]

        tone = color[i] # define the color of the words
        liste = []
        for key, value in list_cluster_occurences.items():
            liste.append([key, value])
        liste.sort(key = lambda x:x[1], reverse = True)
        make_wordcloud(liste, i+1)  

    # Save Dataframe to csv so we can use in other sections
    pd.DataFrame(description1).to_csv('description1.csv')
    pd.DataFrame(clusters).to_csv('clusters.csv')
    st.write(type(clusters))
    st.write(clusters)
            

#   ==========
#   FOURTH TAB
#   ==========
def fourthTab():
    st.subheader("4. Customer categories")
    
    if st.button("run code"):
        runCodeForSection4()

def runCodeForSection4():
    # Read Dataframe so we can use in other sections
    dataframe = pd.read_csv("newDataframe.csv",encoding='unicode_escape')
    dataframeClean = pd.read_csv("dataframeClean.csv",encoding='unicode_escape')
    description1 = pd.read_csv("description1.csv",encoding='unicode_escape').to_numpy()
    description1 = np.delete(description1, [0]).astype(str)
    # clusters = pd.read_csv("clusters.csv")

    file = open("clusters.csv", "r")
    csv_reader = csv.reader(file)

    clusters1 = []
    for row in csv_reader:
        clusters1.append(row)

    # st.write(np.array(clusters1))
    # st.write(type(np.array(clusters1)))

    # clusters = np.array(clusters1)
    # st.write(type(description1))

    clusters = np.delete(clusters1, [0]).astype(str)


    # Here we specify the category of each product for all of our records
    corresp = dict()
    for key, val in zip (description1, clusters):
        corresp[key] = val 
    dataframeClean['categ_product'] = dataframeClean.loc[:, 'Description'].map(corresp)

    # Here we have the amount spent in each category for each product
    for i in range(5):
        column = 'categ_{}'.format(i)        
        df_temp = dataframeClean[dataframeClean['categ_product'] == i]
        price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
        price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
        dataframeClean.loc[:, column] = price_temp
        dataframeClean[column].fillna(0, inplace = True)
    dataframeClean[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]

    # Here we did again the total basket price for all invoices
    temp = dataframeClean.groupby(['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket = temp.rename(columns = {'TotalPrice':'Basket Price'})

    # We calculate for each orders how much money the customer spend for each category
    temp0 = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['categ_0'].sum()
    temp1 = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['categ_1'].sum()
    temp2 = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['categ_2'].sum()
    temp3 = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['categ_3'].sum()
    temp4 = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['categ_4'].sum()

    # Here we merge the above information to our basket dataset
    basket.insert(loc=3,column='categ_0',value=temp0['categ_0'])
    basket.insert(loc=4,column='categ_1',value=temp1['categ_1'])
    basket.insert(loc=5,column='categ_2',value=temp2['categ_2'])
    basket.insert(loc=6,column='categ_3',value=temp3['categ_3'])
    basket.insert(loc=7,column='categ_4',value=temp4['categ_4'])

    # Here we take into accoung the date and time of the orders
    dataframeClean['InvoiceDate_int'] = dataframeClean['InvoiceDate'].astype('int64')
    temp = dataframeClean.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    dataframeClean.drop('InvoiceDate_int', axis = 1, inplace = True)
    basket.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
    basket = basket[basket['Basket Price'] > 0]
    basket.sort_values('CustomerID', ascending = True)[:5]

    # Split the dataset so we can train and test our algorithm
    set_entrainement = basket[basket['InvoiceDate'] < pd.to_datetime('2011-10-1')]
    set_test         = basket[basket['InvoiceDate'] >= pd.to_datetime('2011-10-1')]
    basket = set_entrainement.copy(deep = True)

    # Here we find the max, the min and the mean of each customer transaction per user per category
    transactions_per_user=basket.groupby(by=['CustomerID'])['Basket Price'].agg(['count','min','max','mean','sum'])
    temp0 = basket.groupby(by=['CustomerID'])['categ_0'].sum()/transactions_per_user['sum']*100
    temp1 = basket.groupby(by=['CustomerID'])['categ_1'].sum()/transactions_per_user['sum']*100
    temp2 = basket.groupby(by=['CustomerID'])['categ_2'].sum()/transactions_per_user['sum']*100
    temp3 = basket.groupby(by=['CustomerID'])['categ_3'].sum()/transactions_per_user['sum']*100
    temp4 = basket.groupby(by=['CustomerID'])['categ_4'].sum()/transactions_per_user['sum']*100

    # Here we inserted this data to the basket
    transactions_per_user.insert(loc=5,column='categ_0',value=temp0)
    transactions_per_user.insert(loc=6,column='categ_1',value=temp1)
    transactions_per_user.insert(loc=7,column='categ_2',value=temp2)
    transactions_per_user.insert(loc=8,column='categ_3',value=temp3)
    transactions_per_user.insert(loc=9,column='categ_4',value=temp4)

    transactions_per_user.reset_index(drop = False, inplace = True)
    transactions_per_user.sort_values('CustomerID', ascending = True)[:5]

    # Here we are going to use the invoice date to find the days after last purchases and the days after first purchases 
    last_date = basket['InvoiceDate'].max().date()
    first_registration = pd.DataFrame(basket.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase      = pd.DataFrame(basket.groupby(by=['CustomerID'])['InvoiceDate'].max())
    test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
    test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)
    transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
    transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']

    transactions_per_user[:5]

    # Our data encoding
    list_cols = ['count','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4']
    selected_customers = transactions_per_user.copy(deep = True)
    matrix = selected_customers[list_cols].to_numpy()

    # Here we found the variables means
    scaler = StandardScaler()
    scaler.fit(matrix)
    st.write('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
    scaled_matrix = scaler.transform(matrix)

    # Here we are making categories of customers so we can predict then better
    n_clusters = 11
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
    kmeans.fit(scaled_matrix)
    clusters_clients = kmeans.predict(scaled_matrix)

    # Here we evalutate the clustering by the silhouette score
    silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
    st.write('Silhouette Score: {:<.3f}'.format(silhouette_avg))

    # Here we see the number of clients in each category after the classification of them
    pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['Number of clients in each category']).T

    # Here we impelemented a report via pca to see the content of the clusters that we did
    pca = PCA(n_components=6)
    matrix_3D = pca.fit_transform(scaled_matrix)
    mat = pd.DataFrame(matrix_3D)
    mat['cluster'] = pd.Series(clusters_clients)

    # Here we are presenting the above data and is a function  that we found on sklearn-documentation
    # PCA is is a tool which  is used to reduce the high dimensional dataset to lower-dimensional 
    # dataset without losing the information from it.
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

    LABEL_COLOR_MAP = {0:'r', 1:'tan', 2:'b', 3:'k', 4:'c', 5:'g', 6:'deeppink', 7:'skyblue', 8:'darkcyan', 9:'orange',
                    10:'yellow', 11:'tomato', 12:'seagreen'}
    label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

    fig = plt.figure(figsize = (12,10))
    increment = 0
    for ix in range(6):
        for iy in range(ix+1, 4):   
            increment += 1
            ax = fig.add_subplot(4,3,increment)
            ax.scatter(mat[ix], mat[iy], c= label_color, alpha=0.5) 
            plt.ylabel('PCA {}'.format(iy+1), fontsize = 12)
            plt.xlabel('PCA {}'.format(ix+1), fontsize = 12)
            ax.yaxis.grid(color='lightgray', linestyle=':')
            ax.xaxis.grid(color='lightgray', linestyle=':')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            if increment == 12: break
        if increment == 12: break
            
    #_______________________________________________
    # I set the legend: abreviation -> airline name
    comp_handler = []
    for i in range(n_clusters):
        comp_handler.append(mpatches.Patch(color = LABEL_COLOR_MAP[i], label = i))

    plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9), 
            title='Cluster', facecolor = 'lightgrey',
            shadow = True, frameon = True, framealpha = 1,
            fontsize = 13, bbox_transform = plt.gcf().transFigure)

    plt.tight_layout()

    # Here we find the cluster that each client belongs and put it to the final dataframe
    selected_customers.loc[:, 'cluster'] = clusters_clients

    # We average the contents of this dataframe by first selecting the different groups of clients.
    merged_df = pd.DataFrame()
    for i in range(n_clusters):
        test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
        test = test.T.set_index('cluster', drop = True)
        test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
        merged_df = pd.concat([merged_df, test])
    #_____________________________________________________
    merged_df.drop('CustomerID', axis = 1, inplace = True)
    st.write('Number of customers:', merged_df['size'].sum())

    merged_df = merged_df.sort_values('sum')

    # We re-organized the content of the dataframe by ordering the different clusters to find the finalized dataset to be train
    list_index = []

    list_index_reordered = list_index
    list_index_reordered += [ s for s in merged_df.index if s not in list_index]

    merged_df = merged_df.reindex(index = list_index_reordered)
    merged_df = merged_df.reset_index(drop = False)

    st.write(merged_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'categ_0',
                        'categ_1', 'categ_2', 'categ_3', 'categ_4', 'size']])

    pd.DataFrame(selected_customers).to_csv('selected_customers.csv')

#   ==========
#   FIFTH TAB
#   ==========
def fifthTab():
    st.subheader("5. Classifying customers")
    
    if st.button("run code"):
        runCodeForSection5()

def runCodeForSection5():
    selected_customers = pd.read_csv("selected_customers.csv",encoding='unicode_escape')
    
    class Class_Fit(object):
        def __init__(self, clf, params=None):
            if params:            
                self.clf = clf(**params)
            else:
                self.clf = clf()

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)
        
        def grid_search(self, parameters, Kfold):
            self.grid = GridSearchCV(estimator = self.clf, param_grid = parameters, cv = Kfold)
            
        def grid_fit(self, X, Y):
            self.grid.fit(X, Y)
            
        def grid_predict(self, X, Y):
            self.predictions = self.grid.predict(X)
            st.write("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y, self.predictions)))

    columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4' ]
    X = selected_customers[columns]
    Y = selected_customers['cluster']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)
    # Initiate and run the Support Vector Machine with the help of a class because we have 2 algorithsm to run
    svc = Class_Fit(clf = svm.LinearSVC)
    svc.grid_search(parameters = [{'C':np.logspace(-2,2,10)}], Kfold = 5)
    svc.grid_fit(X = X_train, Y = Y_train)
    svc.grid_predict(X_test, Y_test)
    # Initiate and run the K-Nearest-Neighbours with the help of a class because we have 2 algorithsm to run
    knn = Class_Fit(clf = neighbors.KNeighborsClassifier)
    knn.grid_search(parameters = [{'n_neighbors': np.arange(1,50,1)}], Kfold = 5)
    knn.grid_fit(X = X_train, Y = Y_train)
    knn.grid_predict(X_test, Y_test)

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
if(radioBtn == "Exploring the content of variables"):
    secondTab()
if(radioBtn == "Inside on product categories"):
    cluster = thirdTab()
if(radioBtn == "Customer Categories"):
    fourthTab()
if(radioBtn == "Classifying Customers"):
    fifthTab()
if(radioBtn == "Testing the predictions"):
    st.write("6")


