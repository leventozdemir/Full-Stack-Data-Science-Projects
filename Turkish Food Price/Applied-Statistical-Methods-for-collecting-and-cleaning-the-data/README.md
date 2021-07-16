# Applied-Statistical-Methods-for-cleaning-data-📊
![Global Food Prices](https://user-images.githubusercontent.com/51120437/125589408-287d737b-2450-411a-94af-65feb00b3ff8.jpeg)

## First let’s define the dataset:
Global Food Prices Database (WFP), This dataset contains Global Food Prices data from the World Food Programme covering foods such as maize, rice, beans, fish, and sugar for 76 countries and some 1,500 markets. It is updated weekly but contains to a large extent monthly data. The data goes back as far as 1992 for a few countries, although many countries started reporting from 2003 or thereafter.

### Source: https://data.humdata.org/dataset/wfp-food-prices

### License: Creative Commons Attribution for Intergovernmental Organisations .

## Exploring the story of this dataset:
### we will start by calling the libraries :
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    %matplotlib inline
    import seaborn as sns
### Now let’s call the data:
    path = '../input/food-price-me/wfpvam_foodprices.csv'
    data = pd.read_csv(path,low_memory=False)
### Story Time:
      print(data.columns)
      #Output: Index(['adm0_id', 'adm0_name', 'adm1_id', 'adm1_name',     'mkt_id', 'mkt_name',
             'cm_id', 'cm_name', 'cur_id', 'cur_name', 'pt_id', 'pt_name', 'um_id',
             'um_name', 'mp_month', 'mp_year', 'mp_price', 'mp_commoditysource'],
            dtype='object')
      print(data.shape)
      #Output: (2004959, 18)
#### so we have 2004960 sample and 18 feature
### Let’s see how many countries are there
    print(data.adm0_name.unique().shape)
    #Output: (98,)
#### so we have 98 countries
    print(data.adm0_name.unique())
    #Output:array(['Afghanistan', 'Algeria', 'Angola', 'Argentina', 'Armenia',
           'Azerbaijan', 'Bangladesh', 'Bassas da India', 'Belarus', 'Benin',
           'Bhutan', 'Bolivia', 'Burkina Faso', 'Burundi', 'Cambodia',
           'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad',
           'China', 'Colombia', 'Congo', 'Costa Rica', "Cote d'Ivoire",
           'Democratic Republic of the Congo', 'Djibouti',
           'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea',
           'Ethiopia', 'Gabon', 'Gambia', 'Georgia', 'Ghana', 'Guatemala',
           'Guinea', 'Guinea-Bissau', 'Haiti', 'Honduras', 'Indonesia',
           'Iran  (Islamic Republic of)', 'Iraq', 'Japan', 'Jordan',
           'Kazakhstan', 'Kenya', 'Kyrgyzstan',
           "Lao People's Democratic Republic", 'Lebanon', 'Lesotho',
           'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
           'Mexico', 'Moldova Republic of', 'Mongolia', 'Mozambique',
           'Myanmar', 'Namibia', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria',
           'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines',
           'Russian Federation', 'Rwanda', 'Senegal', 'Sierra Leone',
           'Somalia', 'South Africa', 'South Sudan', 'Sri Lanka',
           'State of Palestine', 'Sudan', 'Swaziland', 'Syrian Arab Republic',
           'Tajikistan', 'Thailand', 'Timor-Leste', 'Togo', 'Turkey',
           'Uganda', 'Ukraine', 'United Republic of Tanzania', 'Venezuela',
           'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe'], dtype=object)
## From 98 countries i choosed Turkey.
    data_TR=data.loc[data.adm0_name=='Turkey']
    data_TR.isnull().sum()
    #Output: 
    adm0_id                   0
    adm0_name                 0
    adm1_id                   0
    adm1_name             10319
    mkt_id                    0
    mkt_name                  0
    cm_id                     0
    cm_name                   0
    cur_id                    0
    cur_name                  0
    pt_id                     0
    pt_name                   0
    um_id                     0
    um_name                   0
    mp_month                  0
    mp_year                   0
    mp_price                  0
    mp_commoditysource    10319
    data_TR.describe().transpose()
![describe-1](https://user-images.githubusercontent.com/51120437/125589568-5ca26099-e5b4-4b38-88b9-215b5004ce5a.png)

### By looking to the table we can see that the features with
### Standard Deviation=0 is not useful.
#### Standard Deviation = 0, this mean that all the values are equals to each others
### So we will drop adm0_id, cur_id and pt_id
    data_TR['cur_name'].value_counts()
    #Output: TRY    10319
### So cur_name is a categorical data and equals to ‘TRY’ in all line so we will drop it.
    print(data_TR.mkt_id.unique())
    #Output:[1319 2053 2054 2055]
    print(data_TR['mkt_id'].value_counts())
    #Output: 
    1319    3366
    2055    2318
    2054    2318
    2053    2317
    Name: mkt_id, dtype: int64
### mkt_id have 4 different integers that present the mkt_name so i preferir to drop it and keep mkt_name which make it more easy to understand by user.
#### Note: if we kept mkt_id we have to make normalization when we will use predict algorithms.
### Let’s make drop and see the data again
    data_TR = data_TR.drop(['mp_commoditysource','adm1_name','adm0_name','mkt_id','adm1_id','cur_id','pt_id', 'adm0_id', 'pt_name', 'cur_name'],axis=1)
    print(data_TR.describe().transpose())
<img width="677" alt="Screen Shot 2021-07-14 at 11 07 17" src="https://user-images.githubusercontent.com/51120437/125589105-46ecaa33-0fc4-48a5-a649-8cfa49bb62bc.png">

### Rename the Feature to make it easy to understand by users:
    data_TR.columns=['Place', 'ProductId', 'ProductName', 'UmId', 'UmName', 'Month', 'Year', 'Price']
### Split the data to train data before 2020 and test data from 2020 to 2021:
    data_train = data_TR.loc[data_TR.Year<2020]
    data_test = data_TR.loc[data_TR.Year>2020]
### Save the data to csv files:
    data_train.to_csv('train.csv', index=False)
    data_test.to_csv('test.csv', index=False)
### Now it’s your turn download the data from the source and get the full code from my Github and choose the country you want and create your own data set.
## Food Prices in Turkey on Kaggle: https://www.kaggle.com/leventoz/food-prices-in-turkey
## Check this also on Medium: https://leventozdemir.medium.com/applied-statistical-methods-for-cleaning-data-6872e9604dba
# Keep Coding…
