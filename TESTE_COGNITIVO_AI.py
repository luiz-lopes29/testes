#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries

import pandas as pd
import numpy as np
import os
import matplotlib as plt
import seaborn as sns
import datetime


# ## 1 - Load Dataset

# In[2]:


#load the dataset

df_airbnb_calendar = pd.read_csv(r'D:\COGNI\Python\calendar.csv')
df_airbnb_listings = pd.read_csv(r'D:\COGNI\Python\listings.csv')
df_airbnb_reviews = pd.read_csv(r'D:\COGNI\Python\reviews.csv')
df_airbnb_reviews_date = pd.read_csv(r'D:\COGNI\Python\reviews_date.csv')


# In[3]:


#set max columns/rows visualization dataframe

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# In[4]:


#check if the dataframe was load properly
#[df_airbnb_calendar.listing_id == 1526604/ 2050752/17878]

df_airbnb_calendar[df_airbnb_calendar.listing_id == 17878].head(100)


# In[5]:


#check if the dataframe was load properly
#
df_airbnb_listings.head()


# In[6]:


#check if the dataframe was load properly

df_airbnb_reviews.head()


# In[7]:


#check if the dataframe was load properly

df_airbnb_reviews_date.head()


# In[8]:


#check the dataframe dimension

df_airbnb_calendar.shape


# In[9]:


#check the dataframe dimension

df_airbnb_listings.shape


# In[10]:


#check the dataframe dimension

df_airbnb_reviews.shape


# In[11]:


#check the dataframe dimension

df_airbnb_reviews_date.shape


# In[12]:


#check the type of each column

df_airbnb_calendar.dtypes


# In[13]:


#check the type of each column

df_airbnb_listings.dtypes


# In[14]:


#check the type of each column

df_airbnb_reviews.dtypes


# In[15]:


#check the type of each column

df_airbnb_reviews_date.dtypes


# ### 2 - Cleaning and treat

# In[16]:


#remove the symbol $/% from columns

df_airbnb_calendar['price']=df_airbnb_calendar['price'].str.strip("$").str.strip(",")
df_airbnb_calendar['adjusted_price']=df_airbnb_calendar['adjusted_price'].str.strip("$").str.strip(",")
df_airbnb_listings['host_response_rate']=df_airbnb_listings['host_response_rate'].str.strip('%')
df_airbnb_listings['host_acceptance_rate']=df_airbnb_listings['host_acceptance_rate'].str.strip('%')
df_airbnb_listings['price']=df_airbnb_listings['price'].str.strip("$").str.strip(",")


# In[17]:


#convert these columns in numbers/date

df_airbnb_calendar['price']=pd.to_numeric(df_airbnb_calendar['price'], errors='coerce')
df_airbnb_calendar['adjusted_price']=pd.to_numeric(df_airbnb_calendar['adjusted_price'],errors='coerce')
df_airbnb_calendar['date']= pd.to_datetime(df_airbnb_calendar['date'], format= '%Y-%m-%d')
df_airbnb_reviews_date['date']= pd.to_datetime(df_airbnb_reviews_date['date'], format= '%Y-%m-%d')

df_airbnb_listings['first_review']= pd.to_datetime(df_airbnb_listings['first_review'], format= '%Y-%m-%d')
df_airbnb_listings['last_review']= pd.to_datetime(df_airbnb_listings['last_review'], format= '%Y-%m-%d')
df_airbnb_listings['price']=pd.to_numeric(df_airbnb_listings['price'], errors = 'coerce')
df_airbnb_listings['host_response_rate']=pd.to_numeric(df_airbnb_listings['host_response_rate'], errors = 'coerce')
df_airbnb_listings['host_acceptance_rate']=pd.to_numeric(df_airbnb_listings['host_acceptance_rate'], errors = 'coerce')


# In[18]:


#check if the type was switch to number type

df_airbnb_calendar.dtypes


# In[19]:


#check if the type was switch to number type

df_airbnb_listings.dtypes


# In[20]:


#check if the type was switch to number type

df_airbnb_reviews_date.dtypes


# In[21]:


#% number of missing per column

df_airbnb_calendar.isnull().sum()/len(df_airbnb_calendar)


# In[22]:


#% number of missing per column

df_airbnb_listings.isnull().sum()/len(df_airbnb_listings)


# In[23]:


#% number of missing per column


df_airbnb_reviews.isnull().sum()/len(df_airbnb_reviews)


# In[24]:


#% number of missing per column

df_airbnb_reviews_date.isnull().sum()/len(df_airbnb_reviews_date)


# ### 3 - Feature engineering

# In[25]:


# select only the columns are likely related with model

select_columns = ['id', 'host_since', 'host_response_rate',                  'host_acceptance_rate', 'host_is_superhost', 'has_availability',                  'availability_30','availability_60', 'availability_90', 'availability_365',                  'host_identity_verified', 'host_total_listings_count',                  'host_has_profile_pic', 'neighbourhood_cleansed', 'latitude', 'longitude',                 'property_type', 'room_type', 'accommodates', 'bathrooms_text','bedrooms', 'beds',                  'amenities','price', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',                 'number_of_reviews', 'review_scores_location', 'review_scores_value',                 'review_scores_rating',
                 'first_review', 'last_review','reviews_per_month']
df_airbnb_listings_sel= df_airbnb_listings[select_columns]


# In[26]:


df_airbnb_listings_sel.dtypes


# In[27]:


# create a dataset Availability

df_availability=df_airbnb_listings_sel[['id','has_availability', 'availability_30',                                        'availability_60', 'availability_90', 'availability_365']]


# In[28]:


#create feature booking

df_availability['%booking30']=1-df_availability.availability_30/30
df_availability['%booking60']=1-df_availability.availability_60/60
df_availability['%booking90']=1-df_availability.availability_90/90
df_availability['%booking365']=1-df_availability.availability_365/365

df_availability.head()


# In[29]:


#create a month number column

df_airbnb_calendar['month_number']=df_airbnb_calendar['date'].dt.month


df_airbnb_calendar.head(100)


# In[30]:


#reshape df_airbnb_calendar to use forward

df_airbnb_calendar_filter=df_airbnb_calendar[['listing_id', 'price', 'adjusted_price', 'month_number']]

df_airbnb_calendar_filter=df_airbnb_calendar_filter.drop_duplicates(subset=['listing_id', 'month_number'])

df_airbnb_calendar_filter.shape


# In[31]:


#join availability with price

#df_availability_join=df_availability.merge(df_airbnb_calendar, left_on ='id', right_on ='listing_id').drop('minimum_nights')

#df_availability_join.head()


# In[32]:


#create a month number column

df_airbnb_reviews_date['month_number']=df_airbnb_reviews_date['date'].dt.month

df_airbnb_reviews_date.head()


# In[33]:


#calcute days between listing id

df_airbnb_reviews_date=df_airbnb_reviews_date.sort_values(['listing_id', 'date'])

df_airbnb_reviews_date['days_between']=np.where(df_airbnb_reviews_date['listing_id']                                                == df_airbnb_reviews_date['listing_id'].shift(-1),                                                df_airbnb_reviews_date.date-df_airbnb_reviews_date.date.shift(+1),0)


# In[34]:


df_airbnb_reviews_date.head()


# In[35]:


#join listing days with pricing calendar

df_airbnb_reviews_date_join=df_airbnb_reviews_date.merge(df_airbnb_calendar_filter,                                                    left_on= ['listing_id', 'month_number'],                                                    right_on= ['listing_id', 'month_number'])
                                                    
                                                       

df_airbnb_reviews_date_join.head()


# In[36]:


#create dataframe reviews price with de availability

df_airbnb_price=df_airbnb_reviews_date_join.merge(df_availability, left_on= 'listing_id', right_on='id')

df_airbnb_price.head()


# In[37]:


#create column stay price

df_airbnb_price.days_between=df_airbnb_price.days_between.astype('timedelta64[D]')

df_airbnb_price['stay_price']=np.where(df_airbnb_price.days_between <= 30,                                       df_airbnb_price.days_between * df_airbnb_price['%booking30'] * df_airbnb_price.price,                                       np.where((df_airbnb_price.days_between > 30)& (df_airbnb_price.days_between <= 60), df_airbnb_price.days_between * df_airbnb_price['%booking60'] * df_airbnb_price.price,                                       np.where((df_airbnb_price.days_between > 60)& (df_airbnb_price.days_between <= 90), df_airbnb_price.days_between * df_airbnb_price['%booking90'] * df_airbnb_price.price, df_airbnb_price.days_between * df_airbnb_price['%booking365'] * df_airbnb_price.price)))



df_airbnb_price.head()


# In[38]:


#create a dataframe for high_season

df_airbnb_price_high_season=df_airbnb_price[(df_airbnb_price.date >= '2019-01-01')
                                             &((df_airbnb_price.month_number ==12)\
                                            |(df_airbnb_price.month_number ==1)\
                                            |(df_airbnb_price.month_number ==2))]

df_airbnb_price_high_season=df_airbnb_price_high_season.groupby('listing_id')['stay_price'].mean().reset_index()

df_airbnb_price_high_season.head(100)


# In[39]:


#check size new dataframe

df_airbnb_price_high_season.shape


# In[40]:


#check missing data

df_airbnb_price_high_season.isnull().sum()/len(df_airbnb_price_high_season)


# In[41]:


#dropna and stay pricing equal or negative

df_airbnb_price_high_season=df_airbnb_price_high_season[df_airbnb_price_high_season.stay_price > 0.0].dropna()


# In[42]:


#check shape dataframe

df_airbnb_price_high_season.shape


# In[43]:


#verify characterisc of data

round(df_airbnb_price_high_season.describe(),0)


# In[44]:


df_airbnb_listings_sel.head()


# In[45]:


df_airbnb_listings_sel.nunique()


# In[46]:


#generate features dummies cardinality <10

dum = pd.get_dummies(df_airbnb_listings_sel, 
                    columns= ['host_is_superhost', 'room_type', 'host_identity_verified', 'host_has_profile_pic'],
                    drop_first=True,
                    prefix = ['host_is_superhost', 'room_type', 'host_identity_verified', 'host_has_profile_pic'],
                    prefix_sep= '_')
dum.head()


# In[47]:


#gerando variáveis dummies para cardinalidade >10

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le_tkt = le.fit_transform(dum['property_type'])
le_tkt_df1 = pd.DataFrame(le_tkt, columns=['Le_property_type'])

le_tkt = le.fit_transform(dum['neighbourhood_cleansed'])
le_tkt_df2 = pd.DataFrame(le_tkt, columns=['LE_neighbourhood_cleansed'],)

df_base_model = pd.merge(dum,le_tkt_df1, left_index=True, right_index=True)
df_base_model = pd.merge(df_base_model,le_tkt_df2, left_index=True, right_index=True)


# In[48]:


df_base_model.head()


# In[49]:


#join listings detail with stay price

df_airbnb_final_base=df_airbnb_price_high_season.merge(df_base_model, left_on = 'listing_id', right_on = 'id')

df_airbnb_final_base.drop(['listing_id', 'id', 'host_since','has_availability', 'neighbourhood_cleansed', 'property_type', 'availability_30', 'availability_60', 'availability_90',
                           'availability_365', 'price', 'latitude', 'longitude', 'amenities', 'bathrooms_text', 'first_review', 'last_review'], axis='columns', inplace=True)




# In[50]:


#check missing data

df_airbnb_final_base.isnull().sum()/len(df_airbnb_final_base)


# In[51]:


#fill missings with 0

df_airbnb_final_base=df_airbnb_final_base.fillna(0)


# In[52]:


columns = df_airbnb_final_base
cor = columns.corr() 
cor.style.background_gradient(cmap='coolwarm')

#cor = wh1.corr() 
#sns.heatmap(cor, square = True) 


# In[53]:


df_airbnb_final_base.isnull().sum()


# ### 4 - MODEL ML

# In[54]:


#split variables target (y) from features (x)

target=df_airbnb_final_base['stay_price']


# In[55]:


#split variables target (y) from features (x)

explicativas=df_airbnb_final_base
explicativas.drop(['stay_price'], axis='columns', inplace=True)


# In[56]:


# Baseado em F_test

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

X_norm = MinMaxScaler().fit_transform(explicativas)

F_selector = SelectKBest(f_regression)
F_selector.fit(X_norm, target)
F_support = F_selector.get_support()
F_feature = explicativas.loc[:,F_support].columns.tolist()
print(str(len(F_feature)), 'selected features')
print(F_feature)


# In[57]:


explicativas_final=explicativas[['host_response_rate', 'host_acceptance_rate', 'accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_value', 'reviews_per_month', 'host_is_superhost_t', 'room_type_Private room']]


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_all = explicativas_final
y_all = target

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.3 ,random_state = 123)

print('Numero de observaçoes do treino:', len(x_train))
print('Numero de observaçoes da teste:',len(x_test))


# In[60]:


#carregando todos os modelo que serão utilizados

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[61]:


# chamando o objeto do modelo de Regressão Linear


LR = LinearRegression()

LR.fit(x_train, y_train)


# In[62]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[63]:


#Avaliando pela Raiz do erro quadrático médio e R2 

MSE_train=mean_squared_error(y_train, LR.predict(x_train))
LR_RMSE_train= MSE_train**0.5
LR_R2_train=r2_score(y_train, LR.predict(x_train))

MSE_test=mean_squared_error(y_test, LR.predict(x_test))
LR_RMSE_test= MSE_test**0.5
LR_R2_test=r2_score(y_test, LR.predict(x_test))


# In[64]:


# parâmetros para  GBR

params={
    'n_estimators':500, 
    'criterion':'mse', 
    'max_depth':4, 
    'min_samples_split':5, 
    'min_samples_leaf':5,
    'min_weight_fraction_leaf':0.0, 
    'max_features':'auto'} 


# In[65]:


# Rodando o modelo GBR

gbr = GradientBoostingRegressor(**params)
gbr.fit(x_train, y_train)


# In[66]:


#Avaliando pela Raiz do erro quadrático médio e R2 

MSE_train= mean_squared_error(y_train, gbr.predict(x_train))
GBR_RMSE_train= MSE_train**0.5
GBR_R2_train=r2_score(y_train, gbr.predict(x_train))

MSE_test= mean_squared_error(y_test, gbr.predict(x_test))
GBR_RMSE_test= MSE_test**0.5
GBR_R2_test=r2_score(y_test, gbr.predict(x_test))


# In[67]:


# parâmetros para  RFR

params={
    'n_estimators':500, 
    'criterion':'mse', 
    'max_depth':4, 
    'min_samples_split':5, 
    'min_samples_leaf':5,
    'min_weight_fraction_leaf':0.0, 
    'max_features':'auto'} 


# In[68]:


#Rodando modelo RFR

rfr = RandomForestRegressor(**params)
rfr.fit(x_train, y_train)


# In[69]:


#Avaliando pela Raiz do erro quadrático médio e R2 

MSE_train= mean_squared_error(y_train, rfr.predict(x_train))
RFR_RMSE_train= MSE_train**0.5
RFR_R2_train=r2_score(y_train, rfr.predict(x_train))

MSE_test= mean_squared_error(y_test, rfr.predict(x_test))
RFR_RMSE_test= MSE_test**0.5
RFR_R2_test=r2_score(y_test, rfr.predict(x_test))


# In[70]:


columns=['LinearRegression_train', 
         'GradientBoostingRegressor_train',  
         'RandomForestRegressor_train',
         'LinearRegression_test', 
         'GradientBoostingRegressor_test',  
         'RandomForestRegressor_test' 
         ]

Modelos=pd.DataFrame({'R2': [LR_R2_train, GBR_R2_train, RFR_R2_train, LR_R2_test, GBR_R2_test, RFR_R2_test],
                     'RMSE': [LR_RMSE_train, GBR_RMSE_train, RFR_RMSE_train, LR_RMSE_test, GBR_RMSE_test, RFR_RMSE_test]
                     },
index = columns)


# In[71]:


#selecionando os três melhores modelos

Modelos.sort_values(['R2', 'RMSE'], ascending= False)

