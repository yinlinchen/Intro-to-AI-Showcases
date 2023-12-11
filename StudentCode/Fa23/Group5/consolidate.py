import pandas as pd
column_names = [
"Taxpayer Number",
"Taxpayer Name",
"Taxpayer Address",
"Taxpayer City",
"Taxpayer State",
"Taxpayer Zip",
"Taxpayer County",
"Taxpayer Phone",
"Location Number",
"Location Name",
"Location Address",
"Location City",
"Location State",
"Location Zip",
"Location County",
"Location Phone",
"Unit Capacity",
"Responsibility Begin Date (YYYYMMDD)",
"Responsibility End Date (YYYYMMDD)",
"Obligation End Date (YYYYMMDD)",
"Filer Type",
"Total Room Receipts",
"Taxable Receipts",
]
unwanted_columns = [
    "Taxpayer Number",
    "Taxpayer Name",
    "Taxpayer Address",
    "Taxpayer City",
    "Taxpayer State",
    "Taxpayer Zip",
    "Taxpayer County",
    "Taxpayer Phone",
    "Location Number",
    "Location Address",
    "Location State",
    "Location Phone",
    "Responsibility Begin Date (YYYYMMDD)",
    "Responsibility End Date (YYYYMMDD)",
    "Taxable Receipts"
    ]
populationDataFile = "texasStatePopulationData.csv"
statePopData = pd.read_csv(populationDataFile, index_col = 0)
populationDataIndex = 0
populationGrowthDataIndex = 1
q1Date = 331
q2Date = 630
q3Date = 930

for i in range(0, 16):
    if i!= 1:
        year = "20"
        if i>= 10:
            year = year + str(i)
        else:
            year = year + "0" + str(i)
        dataFile = "LoanData/HOTEL" + year + ".csv"
        df = pd.read_csv(dataFile)
        df.columns = column_names
        df.drop(columns=unwanted_columns, inplace=True)
        df['Year'] = int(year)
        df['Population'] = statePopData.loc[int(year)][populationDataIndex]
        df['Population Growth Rate'] = statePopData.loc[int(year)][populationGrowthDataIndex]
        df['Quarter'] = " "
        modularYearValue = int(year)*10000
        df.loc[df["Obligation End Date (YYYYMMDD)"] - modularYearValue <= q1Date, 'Quarter'] = "Q1"
        df.loc[(q1Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue)) &
               ((df["Obligation End Date (YYYYMMDD)"] - modularYearValue) <= q2Date), 'Quarter'] = "Q2"
        df.loc[(q2Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue)) &
               ((df["Obligation End Date (YYYYMMDD)"] - modularYearValue) <= q3Date), 'Quarter'] = "Q3"
        df.loc[q3Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue), 'Quarter'] = "Q4"
        if i == 0:
            df.to_csv('TrainingLoanData.csv', mode='a', index=False, header=True)
        else:
            df.to_csv('TrainingLoanData.csv', mode='a', index=False, header=False)
'''
for i in range(16, 23):
    if i!= 1:
        year = "20"
        if i>= 10:
            year = year + str(i)
        else:
            year = year + "0" + str(i)
        print(year)
        dataFile = "LoanData/HOTEL" + year + ".csv"
        df = pd.read_csv(dataFile)
        df.columns = column_names
        df.drop(columns=unwanted_columns, inplace=True)
        df['Year'] = int(year)
        df['Population'] = statePopData.loc[int(year)][populationDataIndex]
        df['Population Growth Rate'] = statePopData.loc[int(year)][populationGrowthDataIndex]
        df['Quarter'] = " "
        modularYearValue = int(year)*10000
        df.loc[df["Obligation End Date (YYYYMMDD)"] - modularYearValue <= q1Date, 'Quarter'] = "Q1"
        df.loc[(q1Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue)) &
               ((df["Obligation End Date (YYYYMMDD)"] - modularYearValue) <= q2Date), 'Quarter'] = "Q2"
        df.loc[(q2Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue)) &
               ((df["Obligation End Date (YYYYMMDD)"] - modularYearValue) <= q3Date), 'Quarter'] = "Q3"
        df.loc[q3Date < (df["Obligation End Date (YYYYMMDD)"] - modularYearValue), 'Quarter'] = "Q4"
        if i == 16:
            df.to_csv('TestingLoanData.csv', mode='a', index=False, header=True)
        else:
            df.to_csv('TestingLoanData.csv', mode='a', index=False, header=False)
'''
