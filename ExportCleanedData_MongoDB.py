# -*- coding: utf-8 -*-
"""
Created on Sun May 31 02:47:49 2020

@author: utilisateur
"""

from pymongo import MongoClient
import json
import pymongo
import pandas as pd


def updateDocument(myCollection,jsonFile):
    for doc in jsonFile:
        try:
            myCollection.update_one(doc, upsert=True) 
        except:
            pass
        
def ExportCleanedScrapedData(salary,NanSalary,AllData):
    # connect to MongoDB
    client = MongoClient("mongodb+srv://root:root89@reemhasan-e3kq8.gcp.mongodb.net/test?retryWrites=true&w=majority")
    db = client.test
    print(db)
    
    # connect to database
    myDB=client['IndeedScrapedData']
    # connect to the tables in this database
    SalaryCollection= myDB['Data_withSalary']
    print('Salary table size before adding new data ',SalaryCollection.count_documents({}))
    
    NanSalaryCollection=myDB['Data_withoutSalary']
    print('NonSalary table size before adding new data ',NanSalaryCollection.count_documents({}))
    
    allCollection=myDB['Scraped_all']
    print('AllData table size before adding new data ',allCollection.count_documents({}))
    ### Transform the dataframe into Json file to upload it into MongoDB
    jsonsalary=salary.to_dict('records')
    jsonNanSalary=NanSalary.to_dict('records')
    jsonAllData=AllData.to_dict('records')
    
    # update all the documents in the dataset
    #SalaryCollection.update_one(jsonsalary, upsert=True)
    updateDocument(SalaryCollection,jsonsalary)
    print('Salary table size after adding new data ',SalaryCollection.count_documents({}))
    #NanSalaryCollection.update_one(jsonNanSalary, upsert=True)
    updateDocument(NanSalaryCollection,jsonNanSalary)
    print('NonSalary table size after adding new data ',NanSalaryCollection.count_documents({}))
    updateDocument(allCollection,jsonAllData)
    #allCollection.update_many(jsonAllData, upsert=True)
    print('AllData table size after adding new data ',allCollection.count_documents({}))
    