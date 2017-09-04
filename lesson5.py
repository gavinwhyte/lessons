# Real- Valued Predicitions using Real-Valued Attributes

# Calculate how your wine tastes.

# The wine taste data set contains data for approximately 1500 red wines
# For each wine there are a number of measurements of chemical compositions
# Including things like alcohol content, volatile acidity  and sulphites.
# Each wine has a taste score determined by averaging the scores given by three professional wine
# tasters

# The problem statement is to build a model that will incorporate the chemical measurements and
# predict taste scores to match those given by human tasters

# Below is a numeric summary of the data
# The code generates a box plot od the normalised variables so that you can visualise the outliers

# The numerical summaries and plots indicates numerous outlying values.


# When analysing the performance of your trained models, these outlying examples will be one place
# to look to understand the source of errors in your model


__author__ = 'gavin whyte'


import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import exp

target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/wine-quality/winequality-red.csv"

wine = pd.read_csv(target_url, header=0, sep=";")


print(wine.head())

#generate statistical summaries

summary = wine.describe()

print(summary)

winenormalised = wine

ncols = len(winenormalised.columns)

for i in range(ncols):
    mean = summary.iloc[1,i]
    sd = summary.iloc[2,i]
    winenormalised.iloc[:, i:(i + 1)] = (winenormalised.iloc[:, i:(i+1)] - mean) / sd

array = winenormalised.values
plot.boxplot(array)

plot.xlabel("Attribute Index")

plot.ylabel(("Quartile Ranges - Normalized "))
plot.show()

# Producing a color-coded parallel coordinates plot for the wine data will give some idea of
# how well correlated the attributes are with the targets

#
#
# nrows = len(wine.index)
#
# tasteCol = len(summary.columns)
#
# meanTaste = summary.iloc[1, tasteCol - 1]
# sdTaste = summary.iloc[2, tasteCol - 1]
#
# nDataCol = len(wine.columns) - 1
#
# for i in range(nrows):
#     #plot rows of data as if they were series data
#     dataRow = wine.iloc[i, 1:nDataCol]
#     normTarget = (wine.iloc[i, nDataCol] - meanTaste) / sdTaste
#     labelColor = 1.0/(1.0 + exp(-normTarget))
#     dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
#
# plot.xlabel("Attribute Index")
# plot.ylabel("Attribute Value")
#
# plot.show()
#
# # Try again with Normailsed values
#
# winenormalised2 = wine
#
# ncols2 = len(winenormalised2.columns)
#
#
# for i in range(ncols2):
#     mean1 = summary.iloc[1, i]
#     sd1 = summary.iloc[2, i]
#     winenormalised2.iloc[:, i:(i + 1)] = (winenormalised2.iloc[:, i:(i+1)] - mean1) / sd1
#
# for i in range(nrows):
#     #plot rows of data as if they were series data
#     dataRow = winenormalised2.iloc[i, 1:nDataCol]
#     normTarget = winenormalised2.iloc[i, nDataCol]
#     labelColor = 1.0/(1.0 + exp(-normTarget))
#     dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
#
# plot.xlabel("Attribute Index")
# plot.ylabel("Attribute Value")
# plot.show()
