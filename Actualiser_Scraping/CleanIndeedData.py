# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:18:38 2020

@author: utilisateur
"""

#import modules
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import FrenchStemmer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from gensim import matutils, models
import scipy.sparse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import warnings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.simplefilter(action='ignore')
# parsing date and extract the date in which the job offer has been published
from datetime import datetime, timedelta
import spacy
from spacy import displacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from ExportCleanedData_MongoDB import ExportCleanedScrapedData
from IPython.display import display

stemmer = FrenchStemmer()
tokenizer = nltk.RegexpTokenizer(r'\w+')
stop_fr = nltk.corpus.stopwords.words('french')
stop_uk = nltk.corpus.stopwords.words('english')
stop_spacy_fr = list(fr_stop)

def parse_date(date):   
    try:
        N = int(''.join(filter(lambda x: x.isdigit(), date)))
        date_N_days_ago = datetime.now() - timedelta(days=N)
        date_N_days_ago = date_N_days_ago.strftime('%Y-%m-%d')
        return date_N_days_ago
    except:
        pass 
    
def avg_sal(sal):
    try:
        splt = sal.split(' - ', 1)
        first = float(splt[0])
        second = float(splt[1])
        return (first + second)/2
    except:
        return float(sal)

def min_sal(sal):
    try:
        splt = sal.split(' - ', 1)
        first = float(splt[0])
        return first
    except:
        return float(sal)
    
def max_sal(sal):
    try:
        splt = sal.split(' - ', 1)
        second = float(splt[1])
        return second
    except:
        return float(sal)
    
import re
def ParseLocation(x):
    # delete the parentese ( ) from location
    x=re.sub(r"\(.*\)","",x)
    # delete the digits with letters from location as paris 1e, lyon 5e 
    x=re.sub(r'\d\w+','',x).strip()
    return x

def parse_text(x,location):
    text=str(x)
    # tokenize the text
    text = tokenizer.tokenize(text.lower())
    # delete english and frensh stop words
    text = [word for word in text if not word in stop_fr]
    text = [word for word in text if not word in stop_uk]
    text = [word for word in text if not word in stop_spacy_fr]
    # delete not wanted chars/words
    text = [word for word in text if word !=location.lower()]
    # delete not chars
    text=[word for word in text if word.isalpha()]
    # keep words with length more than 2 chars
    text=[word for word in text if len(word)>2]
    
    return text

def parseContract(x):
    text=str(x)
    # tokenize the text
    text = tokenizer.tokenize(text.lower())
    # delete not chars
    text=[word for word in text if word.isalpha()]
    # keep words with length more than 2 chars
    text=[word for word in text if len(word)>2]
    return text

def FindJobType(x, desc):
    ''' input : contract type / job description
        search for correct job type'''
    # verify if the job type is cdi / temps plein / temps partiel / unknown type
    if len(desc)<=2:
        # check if the job type is included in desc
        if all(item in x for item in desc):
            return 1
        else:
            return 0
    # if the job type is cdd, then we verify that there is at least one description of it in the contract
    elif any(w in desc for w in x):
        return 1
    else: return 0

# This function creates five columns in the dataframe df (keys in dictionary Dec), and assign the value 1 (exist) or 0 if not in the column contract
def DummyJobType(df,Dic):
    for Jtype, desc in Dic.items():
        for index ,row in df.iterrows():
            df.at[index, Jtype]=FindJobType(row['contractWords'],desc)
    return df

def getRegion(df):
    REGIONS = {'Auvergne-Rhône-Alpes': [1, 3, 7, 15, 26, 38, 42, 43, 63, 69, 73, 74],
    'Bourgogne-Franche-Comté': [21, 25, 39, 58, 70, 71, 89, 90],
    'Bretagne': [35, 22, 56, 29],
    'Centre-Val de Loire': [18, 28, 36, 37, 41, 45],
    'Corse': [2],
    'Grand Est': [8, 10, 51, 52, 54, 55, 57, 67, 68, 88],
    'Dom': [971, 972, 973, 974],
    'Hauts-de-France': [2, 59, 60, 62, 80],
    'Île-de-France': [75, 77, 78, 91, 92, 93, 94, 95],
    'Normandie': [14, 27, 50, 61, 76],
    'Nouvelle-Aquitaine': [16, 17, 19, 23, 24, 33, 40, 47, 64, 79, 86, 87],
    'Occitanie': [9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82],
    'Pays de la Loire': [44, 49, 53, 72, 85],
    'Provence-Alpes-Côte d\'Azur': [4, 5, 6, 13, 83, 84]}
    
    df['Regions']=""
        #print(region,deps)
    for index ,row in df.iterrows():
        if row['Dept']==0:
            #print(row['Dept'])
            df.at[index,'Regions']="Non renseigné"
        else:
            for region,deps in REGIONS.items():
                if row['Dept'] in deps:
                #print('true')
                   df.at[index,'Regions']=region
                else:
                    pass
            #df.at[index,'Regions']=FindRegion(row['Dept'],region,deps)
    return df




def CleanScrapedData(df):

# 1. Import the scraped data from indeed website
    df=df.replace('None', np.nan)
    print('initial data shape is ',df.shape)
    df = df[~df.duplicated(keep='first')] 
    print('data shape after deleting duplicate ',df.shape)
    # only with salary data present
    df_salary = df.dropna(subset=['Salary'])
    print('data with salary ',df_salary.shape)
    # rest of the dataframe without salary info
    df = df[~df.Salary.isin(df_salary.Salary)]
    print('Data without salary is ',df.shape)
    
### 2. Deal with salary 
    df_salary["Salary"] = df_salary["Salary"].str.replace("\n", "")
    df_salary["Salary"] = df_salary["Salary"].str.replace(",", "")
    df_salary["Salary"] = df_salary["Salary"].str.replace("€", "")
    df_salary["Salary"] = df_salary["Salary"].str.replace("\xa0", "")
    display(df_salary.head(2))
    
    year_salaries = df_salary[df_salary["Salary"].str.contains("an")]
    month_salaries = df_salary[df_salary["Salary"].str.contains("mois")]
    day_salaries = df_salary[df_salary["Salary"].str.contains("jour")]
    hour_salaries = df_salary[df_salary["Salary"].str.contains("heure")]
    print(year_salaries.shape)
    print(month_salaries.shape)
    print(day_salaries.shape)
    print(hour_salaries.shape)
    # removing string values("par an", " par mois", etc. from salary dfs)

    year_salaries["Salary"] = year_salaries["Salary"].str.replace(" par an", "")
    month_salaries["Salary"] = month_salaries["Salary"].str.replace(" par mois", "")
    day_salaries["Salary"] = day_salaries["Salary"].str.replace(" par jour", "")
    hour_salaries["Salary"] = hour_salaries["Salary"].str.replace(" par heure", "")
    
    # min salary

    year_salaries["min_salary"] = year_salaries["Salary"].apply(min_sal)
    month_salaries["min_salary"] = month_salaries["Salary"].apply(min_sal)
    day_salaries["min_salary"] = day_salaries["Salary"].apply(min_sal)
    hour_salaries["min_salary"] = hour_salaries["Salary"].apply(min_sal)
    # max salary

    year_salaries["max_salary"] = year_salaries["Salary"].apply(max_sal)
    month_salaries["max_salary"] = month_salaries["Salary"].apply(max_sal)
    day_salaries["max_salary"] = day_salaries["Salary"].apply(max_sal)
    hour_salaries["max_salary"] = hour_salaries["Salary"].apply(max_sal)
    # average salary

    year_salaries["avg_salary"] = year_salaries["Salary"].apply(avg_sal)
    month_salaries["avg_salary"] = month_salaries["Salary"].apply(avg_sal)
    day_salaries["avg_salary"] = day_salaries["Salary"].apply(avg_sal)
    hour_salaries["avg_salary"] = hour_salaries["Salary"].apply(avg_sal)
    
    # converting to yearly salary

    month_salaries["min_salary"] = month_salaries["min_salary"] * 12
    day_salaries["min_salary"] = day_salaries["min_salary"] * 230
    hour_salaries["min_salary"] = hour_salaries["min_salary"] * 1607
    
    month_salaries["max_salary"] = month_salaries["max_salary"] * 12
    day_salaries["max_salary"] = day_salaries["max_salary"] * 230
    hour_salaries["max_salary"] = hour_salaries["max_salary"] * 1607
    
    month_salaries["avg_salary"] = month_salaries["avg_salary"] * 12
    day_salaries["avg_salary"] = day_salaries["avg_salary"] * 230
    hour_salaries["avg_salary"] = hour_salaries["avg_salary"] * 1607
    # Put all the salary dataframes together
    
    df_salary = pd.concat([year_salaries, month_salaries, day_salaries, hour_salaries], sort=False)
    
    print(df_salary.shape)
    
    #rejoining salary data into main scrape_data df
    df = pd.concat([df, df_salary])
    print(df.shape)
    df.drop('Salary',axis=1,inplace=True)
    print(df.info())
    # Date manuplation
    df['Date'] = df['Date'].apply(parse_date)
    df['Date'] = df['Date'].fillna(value=datetime.now().strftime('%Y-%m-%d'))
    df.isna().mean()
    
    ## Extract the department from location, then delete it from location column
    
    # extract department numbers in location
    df['Dept']=df['Location'].str.extract('(\d+)')
    
    # deal with NAN values
    df['Dept'] = df['Dept'].fillna(0)
    df['Dept']=df['Dept'].astype(int)
    df.info()
    
    # Deal with location 
    
    df.Location = df.apply(lambda x: ParseLocation(x.Location), axis=1)
    
    # Extract BagOfWords from title and summary column and save them in a new columns
    
    df['title_words'] = df.apply(lambda x: parse_text(x.Title,x.Location), axis=1)
    df['title_words']=df['title_words'].apply(lambda x: " ".join(x))
    df['summary_words'] = df.apply(lambda x: parse_text(x.Description,x.Location), axis=1)
    
    ## Get the number of opinions from the count column
    
    df.Count=df.Count.str.extract('(\d+)')
    
    ### Get unique job offers from contract
    
    df.Contract = df.Contract.replace(re.compile('^[0-9]'), np.nan, regex=True)
    
    print('unique contract ', len(df.Contract.unique()))
    
    # define a dictionary with the most commun job types
    dic = {'CDI': ['cdi'], 'CDD': ['cdd', 'stage', 'apprentissage', 'contrat pro', 'intérim'], 'Freelance': ['freelance', 'indépendant'],
       'Temps partiel': ['temps', 'partiel'], 'Temps plein': ['temps', 'plein'], 'Unknown': ['nan']}
    
    df['contractWords']= df.apply(lambda x: parseContract(x.Contract), axis=1)
    print(df['contractWords'].head(5))
    df=DummyJobType(df,dic)
    """ Dealing with NaNs
Description and Company have very few missing values so we decide to drop rows. For Rating, we fill with the neutral rating of 3, and for Count, the obvious choixe is 0"""

    df.dropna(subset=['Description', 'Company'], inplace=True)
    df['Rating'].fillna(3, inplace=True)
    df['Count'].fillna(0, inplace=True)
    df['Contract'].fillna('Non renseigné', inplace=True)
    
   ### Location features
    #geolocator = Nominatim(user_agent='BANFrance')
    #df['geocode'] = df['Location'].apply(lambda x: geolocator.geocode(x, country_codes='FR',timeout=10000))
    #df['lat-long'] = df['lat-long'].apply(lambda x: x[:-1] if x else None)
    #position = pd.DataFrame(df['lat-long'].tolist(), columns=['lat','long'], index=df['lat-long'].index)
    #df.reset_index(drop=True, inplace=True)
   # position.reset_index(drop=True, inplace=True)
    #df = pd.concat([df, position], axis=1)
    # creating an extra feature to orient the model focus
   # df['lat*long'] = df['lat']*df['long']
   
   
    # Extract the region of each job offer
    REGIONS = {
    'Auvergne-Rhône-Alpes': [1, 3, 7, 15, 26, 38, 42, 43, 63, 69, 73, 74],
    'Bourgogne-Franche-Comté': [21, 25, 39, 58, 70, 71, 89, 90],
    'Bretagne': [35, 22, 56, 29],
    'Centre-Val de Loire': [18, 28, 36, 37, 41, 45],
    'Corse': [2],
    'Grand Est': [8, 10, 51, 52, 54, 55, 57, 67, 68, 88],
    'Dom': [971, 972, 973, 974],
    'Hauts-de-France': [2, 59, 60, 62, 80],
    'Île-de-France': [75, 77, 78, 91, 92, 93, 94, 95],
    'Normandie': [14, 27, 50, 61, 76],
    'Nouvelle-Aquitaine': [16, 17, 19, 23, 24, 33, 40, 47, 64, 79, 86, 87],
    'Occitanie': [9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82],
    'Pays de la Loire': [44, 49, 53, 72, 85],
    "Provence-Alpes-Côte d'Azur": [4, 5, 6, 13, 83, 84]}
    regions = {}
    for k, v in REGIONS.items():
        for x in v:
            regions[x] = k 
    df['Region'] = df.Dept.map(regions)
    df['Region'].fillna('Non renseigné', inplace=True)
    #df=getRegion(df)
    #uniqueRegion=df.Region.unique
    print('unique Region',len(df.Region.unique()))
    
    ### Spliting and saving dataframes
    

    
    df_salary = df[df['avg_salary'].notna()]
    print('salary size ',df_salary.shape)
    
    
    df_nan = df[df['avg_salary'].isna()]
    #df_nan = df[~df.avg_salary.isin(df_salary.avg_salary)]
    print('Nan salary size ',df_nan.shape)
    display(df.info())
    df_salary = df_salary[df_salary['avg_salary'].astype(int) < 200000]
    df = pd.concat([df_salary, df_nan])      
### stock all in csv file 
    df_salary.to_csv('DATA_WITH_SALARY.csv', index=False)
    df_nan.to_csv('DATA_WITHOUT_SALARY.csv', index=False)
    df.to_csv('DATA_FULL.csv', index=False)
    #### stock cleaned data into MongoDB
    ExportCleanedScrapedData(df_salary,df_nan,df)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   

