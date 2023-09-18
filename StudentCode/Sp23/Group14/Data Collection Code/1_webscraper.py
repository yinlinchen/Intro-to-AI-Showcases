# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:41:46 2023

@author: kents
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/#y22")  

#Give site a chance to load
time.sleep(2)

#2011 
#   //*[@id="fhwacontent"]/div[2]/ul[2]
#2022 
#   //*[@id="fhwacontent"]/div[2]/ul[13]

#2022 months i=1 is station data
# //*[@id="fhwacontent"]/div[2]/ul[13]/li[1]
# //*[@id="fhwacontent"]/div[2]/ul[13]/li[13]

#//*[@id="fhwacontent"]/div[2]/ul[13]/li[13]/a

for i in range(13, 14):
    for j in range(9, 14):
        
        #Missing data for May 2022
        if i == 13 and j == 7:
            continue
        
        print(i, j)
        month = driver.find_element('xpath', f'//*[@id="fhwacontent"]/div[2]/ul[{i}]/li[{j}]/a')
        month.click()
        time.sleep(15)
        

driver.close()