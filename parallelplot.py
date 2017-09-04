

__author__ = 'gavin whyte'


import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot
from math import exp

target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/wine-quality/winequality-red.csv"

wine = pd.read_csv(target_url, header=0, sep=";")



#generate statistical summaries

summary = wine.describe()
nrows = len(wine.index)
tasteCol = len(summary.columns)
meanTaste = summary.iloc[1, tasteCol - 1]
sdTaste = summary.iloc[2, tasteCol - 1]
nDataCol = len(wine.columns) - 1


for i in range(nrows):
    # plot rows of data as if they were series data
    dataRow = wine.iloc[i, 1:nDataCol]
    normTarget = (wine.iloc[i, nDataCol] - meanTaste)/sdTaste
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")

plot.ylabel(("Attributes Values "))
plot.show()



# Producing a color-coded parallel coordinates plot for the wine data will give some idea of
# how well correlated the attributes are with the targets


# Try again with Normailsed values

winenormalised = wine

ncols = len(winenormalised.columns)


for i in range(ncols):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    winenormalised.iloc[:, i:(i + 1)] = (winenormalised.iloc[:, i:(i+1)] - mean) / sd

for i in range(nrows):
    # plot rows of data as if they were series data
    dataRow = winenormalised.iloc[i, 1:nDataCol]
    normTarget = winenormalised.iloc[i, nDataCol]
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()

# The plot of the normalized wine data gives a between simultaneous view of the correlation
# with the targets along all coordinate directions.

# On the far right of the plot dark blue line (high taste scores) aggregate at high values of volatile
# acidity.

# the parallel coordinates plot show that high levels of alcohol go with high taste scores

