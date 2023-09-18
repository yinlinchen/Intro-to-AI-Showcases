# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:12:09 2023

@author: kents
"""


import os
import zipfile
import re
import shutil


month_dict = {'january': '01', 'jan': '01',
              'february': '02', 'feb': '02',
              'march': '03', 'mar': '03',
              'april': '04', 'apr': '04',
              'may': '05',
              'june': '06', 'jun': '06',
              
              'july': '07', 'jul': '07',
              'august': '08', 'aug': '08',
              'september': '09', 'sep': '09',
              'october': '10', 'oct': '10',
              'november': '11', 'nov': '11',
              'december': '12', 'dec': '12',
              }


directory = r"C:\Users\kents\OneDrive\Desktop\Traffic Data"
new_directory = r"C:\Users\kents\OneDrive\Desktop\Traffic Data\VA"

# for folder_name in os.listdir(directory):
#     print(folder_name)
    
#     if folder_name.endswith('.zip'):
        
#         x = folder_name.split('_')
        
#         month = x[0]
#         year = x[1]
        
#         with zipfile.ZipFile(os.path.join(directory, folder_name), 'r') as zip_ref:
#             zip_ref.extractall(os.path.join(directory, f'{year}_{month_dict[month]}' ))
   

for folder_name in os.listdir(directory):  
    print(folder_name)
    
    folder = os.path.join(directory, folder_name)
    
    for subfolder_name in os.listdir(folder):
        if subfolder_name == 'VA':
            continue
        
        print(subfolder_name)
        subfolder = os.path.join(directory, folder_name, subfolder_name)
        
        for file_name in os.listdir(subfolder):
            
            if file_name.startswith('VA_'):
                print('\t' + file_name)
                file = os.path.join(directory, folder_name, subfolder_name, file_name)
                shutil.copy(file, os.path.join(new_directory, file_name))