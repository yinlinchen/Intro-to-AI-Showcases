import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as ws

ws.filterwarnings("ignore")

data = pd.read_csv("BankNote_Authentication.csv")

# data
print()
print(data.head())

# sample size
print()
print(data.shape)

# data type
print()
data.info()

# statistical details
print()
print(data.describe())

# finding null values
print()
print(data.isna().sum())

# countplot
plt.figure()
countplot = sns.countplot(data=data, y="class")
plt.savefig("countplot.png")

# correlation heatmap
plt.figure()
heatmap = sns.heatmap(data.corr(), annot=True, linewidths=0.2)
plt.savefig("heatmap.png")

# pairplot
plt.figure()
pairplot = sns.pairplot(data, diag_kind="hist", hue="class")
plt.savefig("pairplot.png")
