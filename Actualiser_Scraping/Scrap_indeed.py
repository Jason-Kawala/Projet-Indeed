# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:14:24 2020

@author: utilisateur
"""

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotVisibleException, TimeoutException
from selenium.webdriver import ActionChains
import pandas as pd
import time
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from CleanIndeedData import CleanScrapedData

options = webdriver.chrome.options.Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=options)
driver = webdriver.Chrome("C:/Users/utilisateur/Desktop/Simplon/Python/ML/Projet-Indeed/chromedriver")
driver.maximize_window()

#jobs = ['big+data', 'data+scientist', 'data+analyst', 'data+engineer', 'machine+learning','business+intelligence', 'web+developper', 'software+engineer', 'devops']
jobs=['data+scientist']

for j in jobs:
    driver.delete_all_cookies()
    time.sleep(4)

    data = pd.DataFrame(columns=[
        'Title', 'Location', 'Date', 'Company', 'Rating', 'Count', 'Salary',
        'Contract', 'Description'
    ])

    for i in tqdm(range(0, 1000, 10)):
        driver.get('https://www.indeed.fr/emplois?q=' + str(j) +
                   '&l=France&start=' + str(i))

        for h, job in enumerate(driver.find_elements_by_class_name('result')):

            soup = BeautifulSoup(job.get_attribute('innerHTML'), 'html.parser')

            try:
                title = soup.find("a",
                                  class_="jobtitle").text.replace("\n",
                                                                  "").strip()
            except:
                title = 'None'

            try:
                location = soup.find(class_="location").text
            except:
                location = 'None'

            try:
                company = soup.find(class_="company").text.replace("\n",
                                                                   "").strip()
            except:
                company = 'None'

            try:
                salary = soup.find(class_="salary").text.replace("\n",
                                                                 "").strip()
            except:
                salary = 'None'

            try:
                date = soup.find(class_="date").text
            except:
                date = 'None'

            try:
                rating = soup.find(class_="ratingsContent").text.replace(
                    "\n", "").strip()
            except:
                rating = 'None'

            driver.implicitly_wait(2)
            sum_div = job.find_element_by_css_selector("a.jobtitle")

            try:
                ActionChains(driver).move_to_element(sum_div).click(
                    sum_div).perform()
            except:
                close_button = driver.find_elements_by_class_name(
                    'popover-x-button-close')[0]
                close_button.click()
                ActionChains(driver).move_to_element(sum_div).click(
                    sum_div).perform()

            try:
                rating_count = driver.find_element_by_class_name(
                    'slNoUnderline').text
            except:
                rating_count = 'None'

            try:
                contract = driver.find_element_by_css_selector(
                    '.jobMetadataHeader > div:nth-child(2)').text
            except:
                contract = 'None'

            try:
                job_desc = driver.find_element_by_id('vjs-desc').text.replace(
                    "\n", "").strip()
            except:
                job_desc = 'None'

            data = data.append(
                {
                    'Title': title,
                    'Location': location,
                    'Date': date,
                    'Company': company,
                    'Rating': rating,
                    'Count': rating_count,
                    "Salary": salary,
                    'Contract': contract,
                    "Description": job_desc
                },
                ignore_index=True)

    data.to_csv(str(j) + ".csv", index=False)

# Show the dimention of the scraped data
print('data shape with duplicate ',data.shape)
data.drop_duplicates(inplace=True)
print('Data shape after deleting duplicated ',data.shape)

CleanScrapedData(data)
