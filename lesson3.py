# We have left off the last lesson on quantile boundaries

# Today our focus is on the rocks vs mine dataset using Pandas to read and Summarize Data

__author__ = 'gwhyte'

import urllib2
import sys
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

# read data from UCI repository
# url "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"


target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/undocumented/connectionist-bench/sonar/sonar.all-data"



rocksVMines = pd.read_csv(target_url, header=None, prefix="V")

# Print the head and tail of the data frame

print(rocksVMines.head())

print(rocksVMines.tail())

# print the Summary of data frame

summary = rocksVMines.describe()

print(summary)

# After reading the file the first section of the program prints out head and tail.

# Notice all the heads have R labels and the tails have the m labels

# The last bit of the code snippet prints out summaries of the real valued columns in the
# data set

# Pandas makes it possible to automate the steps of calculating mean, variance and quantiles
# Notice that the summary produced by the describe function is iteself
# a data frame so that you can automate the process of screening for attributes that
# have outliers.

# How do we do this ?

# Compare the differences between the various quantiles and raise a flag if any of
# the differences for an attributes are out of scale with other differences for the
# same attributes

# It may be worth looking to detrmine how many rows are involved int he outliers.
# They all may come from a handful of examples.


 # lets contruct a parrallel coordinates plot to represent potential outliers

for i in range(208):
    if rocksVMines.iat[i,60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
    dataRow = rocksVMines.iloc[i, 0:60]
    dataRow.plot(color=pcolor)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

# The lines are color coded accordingly to their labels: blue for R and red for M
# Some times a [plot of this type show clear areas of separation between the classes

# For the rocks vs mines no extremely clear separation is evident in the line plot
# But there are some areas where the blues and red are separated.

# along the bottom of the plot the blues stand out a bit and in the range of attribute indices
# from 30 to 40 , the blues are somewhat higher than the reds.

# These kind of insights can help in interpreting and confirming predicitions made by your trained
# model

