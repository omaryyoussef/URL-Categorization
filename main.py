
# coding: utf-8

# In[1]:

import csv
import nltk
import numpy as np
#from urllib import urlopen
from bs4 import BeautifulSoup
from langdetect import detect
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import requests
from requests.exceptions import ConnectionError
import time
from collections import Counter
import urllib.request
from bs4 import BeautifulSoup
from dict_site import dict_sites


# In[3]:

mydata = 'Dataset.csv'
dataset = pd.read_csv(mydata)
dataset.shape


# In[4]:

dataset.index


# In[5]:

dataset['link'][1:10]
# Better construction, you are already in pandas
dataset.head(10)['link']


# # Dropping non working links from the dataset

# In[6]:

dataset_columns = np.append(dataset.columns.values,'source_site')
filtered_dataset = pd.DataFrame(columns=dataset_columns)


# In[7]:

def get_source_name(url):
        ''''
        This function takes a url and returns the main domain. For example, if the input
        is http://krmalk.tv/video/watch.php?vid=8e54b1d51, it returns krmalk.tv.
        
        @params:
            url: A string the contains the url
        @returns:
            The main domain
        ''''
    return url.split('/')[2]
    


# In[8]:

get_error = []
broken_links = []

# Iterating through the links in the dataset
for index, row in dataset.iterrows():
    
    # The try except is used to catch a connection peer error that happens for some websites.
    try:
        # Get request for the link
        response = requests.get(row['link'], allow_redirects=True)
    except:
        get_error.append(row['link'])
        continue
    
    # Conditioning if the status code is 200 which means that the request has succeeded
    if response.status_code == 200:
        
        source_site = get_source_name(row['link'])
        filtered_dataset = filtered_dataset.append({'link': row['link'],'class':row['class'],'source_site':source_site}, ignore_index=True)
        continue
        
    # If the request did not succeed, save those links and the their status codes  
    else:
        broken_links.append([row['link'],response.status_code])
        print(response.status_code)
    


# ### Taking only the websites that have a frequency of 5 or more

# In[10]:

site_count = filtered_dataset.groupby(['source_site']).count().sort_values(by='class',ascending=False)


# In[11]:

sites_with_atleast_5 = site_count.loc[site_count['class']>=5].reset_index()['source_site'].unique()


# In[12]:

final_filtered_dataset = filtered_dataset.loc[filtered_dataset['source_site'].isin(sites_with_atleast_5)]


# ### Checking the new balance of the final filtered dataset

# In[14]:

Counter(final_filtered_dataset['class'])


# In[19]:

print("The percentage amount of data taken when thresholding by 5 is : ", len(final_filtered_dataset)/len(filtered_dataset)*100)


# ## Scraping process

# All the information that are required to scrape the website are inside the **dict_sites** dictionary

# To extract the features from the websites, I decided to scrape the website directly, regardless whether the URL displays some information or not, this is because for some websites the URL is not consistent.
# >For example we might have:
#     - ramdan.video/video/rahim+episode+5
#     - ramdan.video/video/123we141
#     
# So either way I am going to scrape them, to extract information for the second type of format, hence using scraping for all the domains is a better and safer approach.

# The domain **www.wataan.com** is troublesome because it returns weird symbols because of page encoding issues. It is most probably not set to UTF-8 and so it is dropped.

# In[23]:

final_filtered_dataset = final_filtered_dataset.loc[~(final_filtered_dataset['source_site']=='www.wataan.com')]
series_variations = ['raheem', 'rahim', 'ra7em', 'ra7eem', 'ra7im', 'رحيم', 'r7eem', 'rahiem','ra7iem', 'ر7يم']


# In[72]:

final_filtered_dataset = final_filtered_dataset.loc[~(final_filtered_dataset['source_site']=='medium.com')]
final_filtered_dataset = final_filtered_dataset.loc[~(final_filtered_dataset['source_site']=='www.youbox7.com')]
final_filtered_dataset = final_filtered_dataset.loc[~(final_filtered_dataset['source_site']=='www.halacima.net')]


# In[31]:

final_filtered_dataset.head()


# In[73]:

a = set(list(final_filtered_dataset['source_site'].unique()))
b = set(list(dict_sites.keys()))
a-b


# The above sites should be in the if condition

# In[44]:

titles = []

for index,row in final_filtered_dataset.iterrows():
    
    source_site = row['source_site']
    
    # Requesting the site and scraping
    r = requests.get(row['link'])
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Special cases for scraping
    
    if source_site == 'tv.alarab.com':
        try:
            title = soup.find('div', attrs={'id':'banner_slider'}).find('h2').text
            titles.append(title)
            continue
        except:
            titles.append('Cant scrape')
            continue
        
    elif source_site == 'www.elrdar.com':
        try:
            title = soup.find('div', attrs={'class':'col-xs-12 col-sm-12 col-md-10'}).find('h1').text
            titles.append(title)
            continue
        except:
            titles.append('Cant scrape')
            continue
            
    elif source_site == 'www.mosalslat.video':
        try:
            title = soup.find('h1', attrs={'class':'entry-title'}).text
            titles.append(title)
            continue
        except:
            titles.append('Cant scrape')
            continue
            
    elif source_site == 'www.kelmeten.com':
        try:
            title = soup.find('div', attrs={'class':'row pm-video-heading'}).find('h1').text
            titles.append(title)
            continue
        except:
            titles.append('Cant scrape')
            continue
            
    elif source_site == 'shahidline.com':
        try:
            title = soup.find('div', attrs={'class':'on-episode-postinfo'}).find('h1').text
            titles.append(title)
            continue
        except:
            titles.append('Cant scrape')
            continue
            
    # Used to scrape for each site
    attr_class = dict_sites[source_site]['attr_class']
    attr_type = dict_sites[source_site]['attr_type']
    attr_value = dict_sites[source_site]['attr_value']
    
    try:
        title = soup.find(attr_type, attrs={attr_class:attr_value}).text
        titles.append(title)
    except:
        titles.append('Cant scrape')


# In[ ]:

# If any of the words is present in the title, return 1, else return 0 
is_present = []
for title in titles:
    if any(word in title for word in series_variations):
        is_present.append(1)
    else:
        is_present.append(0)


# In[ ]:

final_filtered_dataset['titles'] = titles
final_filtered_dataset['is_present'] = is_present


# In[ ]:

abs(final_filtered_dataset['is'])


# In[ ]:

get_ipython().system(u' jupyter nbconvert --to=python introducing_Tensors.ipynb')

