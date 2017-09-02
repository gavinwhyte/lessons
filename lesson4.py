# Pearson Correlation Calculations

# Given the Rocks Vs Mines Dataset lets look at correlations between columns.

__author__ = "Gavin Whyte"


import pandas as pd

from pandas import DataFrame

from math import sqrt

import sys


target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/undocumented/connectionist-bench/sonar/sonar.all-data"

# Multiple Comprisons between columns

rocksvsmines = pd.read_csv(target_url, header=None, prefix="V")

# Calculate correlation between real-values attributes

dataRow2 = rocksvsmines.iloc[1, 0:60]

dataRow3 = rocksvsmines.iloc[2, 0:60]

dataRow21 = rocksvsmines.iloc[20, 0:60]

mean2 = 0.0

mean3 = 0.0

mean21 = 0.0

numElt = len(dataRow2)

for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i]/numElt
    mean21 += dataRow21[i]/numElt

var2 = 0.0

var3 = 0.0

var21 = 0.0

for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2) / numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3) / numElt
    var21 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3) / numElt


corr23 = 0.0
corr221 = 0.0

for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * (dataRow3[i] - mean3) / (sqrt(var2 * var3) * numElt)
    corr221 += (dataRow2[i] - mean2) * (dataRow21[i] - mean21) / (sqrt(var2 * var21) * numElt)


sys.stdout.write("Correleation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")

sys.stdout.write("Correleation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")


# In the process of understanding the rocks versus mines data set, this section
# has introduced a number of tools for you to use to gain understanding
# and intuition about your datasets


# Real-Values Predictions with Factor Variables:
# Most of the tools you'ue seen used for the understanding the problem of detecting
# unexploded mines can be applied to regression problems.

# Next predicting the age of an abalone, given physical measurements provides an example
# of such problem.

# The abalone data set poses the problem of predicting the age of an abalone by taking several
# measurements

# It is possible to get a precise reading on the age of an abalone by slicing the shell
# and counting the growth rings
# Its expensive and time consuing to slice the sheels and count the rings under a microscope.

# It will be more convenient and economical to be able to make simple physical
# measurements and make an accurate determination of age of an abalone

# Read and Summarize The Abalone Data set

__author__ = 'gavinwhyte'

import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

abalone = pd.read_csv(target_url, header=None, prefix="V")


abalone.columns = ['Sex', 'Length', "Diameter", 'Height', 'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']

print(abalone.head())

print(abalone.tail())

# print summary of data frame

summary = abalone.describe()
print(summary)

# Box plot the real-valued attributes

# Convert to array for plot routine

array = abalone.iloc[:, 1:9].values
plot.boxplot(array)

plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges"))

plot.show()

# The last column (rings) is out of scale with the rest

array2 = abalone.iloc[:, 1:8].values
plot.boxplot(array2)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges"))
plot.show()

# Removing is ok at times but normalizing the variables generalizes better
# normalize columns to Zero Mean and unit standard devaition
# this is a common normalisation and desirable for other operations
# (like k-means clustering or k-nearest neighbours)


abaloneNormalised = abalone.iloc[:, 1:9]

for i in range(8):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    abaloneNormalised.iloc[:, i:(i + 1)] = (abaloneNormalised.iloc[:, i:(i + 1)] - mean) / sd

array3 = abaloneNormalised.values
plot.boxplot(array3)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges - Normalised"))
plot.show()

# What is the box and Whisker plots.
# These plots hsow a small rectangle with a red line through it.
# The red line marks the median value (or the 50th percentile) for the column of data
# The to mark the 25 th percentile and the bottom the 75 th percentile
# Above and below the box you will see small horizontal ticks, the so called wiskers.
# These are draw at levels that are 1.4 times the interquartile spacing above and below
# the box. Interquartile spacing is the difference between the 75th percentile and the 25 th
# percentile. The space between the top of the top and the upper whisker is 1.4 times the height
# of the box

# You will notice in some cases then whiskers are closer than the 1.4x spacing
# For these cases the data values do not extend all the way to the calculated whisker locations
# The data can extend beyond the calculated whisker locations
# These points can be considered outliers.

# The last section of the code normalizes all the data columns before box plotting.
# Normalization in this case means centering and scaling each column so that a unit
# of attribute number 1 means the same thing as a unit of attribute number 2.

# A number of alogirthms and operations in data science require this type of normalization.

# For example K-MEans clustering builds clusters based on vector distance between rows of
# data. Distance is measured by subtracting one point from another and squaring. If the units
# are different , the numeric distances are different.

# The normalisation adjusts variables so that they have a 0 mean and a Standard deviation
# of 1. This is very common normalization.

# The calculations for the normalization make use of the numbers generated by the summary

# It more or less places the lower and upper edges of the boxes at -1.0 and +1.0, but much
# of the data are outside these boundaries.









