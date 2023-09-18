# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:37:12 2023

@author: kents
"""

import os
import pandas as pd

directory = r"C:\Users\kents\OneDrive\Desktop\Traffic Data\VA_converted"

df = pd.DataFrame()

stations = pd.read_csv(r"C:\Users\kents\OneDrive\Desktop\Traffic Data\VA_2022 (TMAS).STA", sep = '|')


for folder_name in os.listdir(directory):
    print(folder_name)
    
    file = os.path.join(directory, folder_name)
    
    temp = pd.read_csv(file).query("Station_Id == 90106")

    print(temp.size)
    
    df = pd.concat([df, temp], axis = 0)
    
df = df.sort_values(by = ['Year_Record', 'Month_Record', 'Day_Record', 'Hour_Record']).reset_index(drop = True)