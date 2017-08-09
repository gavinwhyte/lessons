# Statistical Terminology for Model Building

# Statistics is the branch of mathematics dealing collection, analysis,
# interpretation, presentation and organisation of numerical data

# Statistics are mainly classified into two sub categories

# 1. Descriptive statistics
# These include
# 1.1. Mean
# 1.2. Medium
# 1.3. standard deviation for continous types(such as age)
# 1.4. Whereas frequency and percentage are useful for categorical variables
# such as gender

# 2. Inferential statistics
# 2.1 Many times a collection of the entire data is impossible,
# hence a subset of the data points is collected, also called a sample
# and conclusions about the entire population will be drawn.
# Inferences are drawn using hypothesis testing, the estimation of numercial
# characteristics, correlation of relationships within data etc.

# Statistical modelling is applying statics on data to find underlying
# the hidden relationships by analyzing the significance of the variables.

# Machine Learning

# Is the branch of computer science that utilises past experience
# to learn from and use its knowledge to make future decisions
# Machine learning is at the intersection of computer science, engineering,
# statistics

# Machine Learning is broadly classified in three categories.
# Supervised Learning
# This is teaching machines to learn the relationship between
# other variables and a target variable. The major segments within
# supervised learning is
# 1. Regression
# 2. Classification

# Unsupervised Learning
# Algorithms learn by themselves with any supervision or without any
# target variable provided.
# It is a question of finding hidden patterns and relations in the data
# The categories un unsupervised learning are as follows
# Dimensionality reduction
# Clustering

# Reinforcement Learning
# This allows the machine or agent to learn its behaviour based on feedback
# from the environment
# In reinforcement learning the agent takes a series of decisive actions
# without supervision and in the end a reward will be given
# Based on the reward the agent reevaluates its path

# In some cases we initially perform unsupervised learning to reduce the
# dimensions allowed by supervised learning when the number of variables
# is very high.

# Similarly in some AI applications, supervised learning combined with
# reinforcement learning could be utilized for solving a problem; an
# example is self driving cars in which initially images are converted to
# numeric format using supervised learning and combined with driving
# actions (left , forward, right and backward)

# Major Differences between statistical modelling and machine Learning
# Stats Modelling                       Machine Learning

# 1.Formalisation of relationships      1. Algorithm that can learn from the
# between variables in the form of      data without relying on rule based
# mathematical equations                programming

# 2. Required to assume the shape       2. Does not need to assume the
# of the model curve prior to           underlying shape, as ML algorithms
# perform model fitting on the data     can learn complex patterns
# (linear, polynomial and so on)        automatically based on data

# 3. Data will be split into 70%        3. Data will be into 50 % 25% 25%
# 30 % to create traning and test data  to create traning, validation and test
#                                       data. Models developed on training
#                                       and hyperparameters are tuned on
#                                       validation data and finally get
#                                       evaluated against test data


# Steps on the machine Learning model development
# The development and deployment of machine learning models involves a
# series of steps

# 1. Collection of data
# Data for ML is collected directly from structured data and unstructured
# data(ie chat etc)

# 2. Data prep and missing outlier treatment:  Data is to be formatted
# as per chosen machine learning algorithm; also missing value treatment
# needs to be performed by replacing missing and outlier values with
# mean/median and so on.

# 3. Data analysis and feature engineering:  Data needs to be analysed in
# order to find any hidden patterns and relations between variables.
# Correct feature engineering with appropriate business knowledge
# will solve 70% of the problems

# 4. Train algorithm on training and validation data.
# Post feature engineering data will be divided into three chunks
# (Training, Validation, Test data) rather than two train and test.
# Machine learning are applied on training data and the hyper-parameters
# of the model are tuned based on validation data to avoid over-fitting.

# 5. Test the Algorithm on test data
# Once the model has shown a good enough performance on train and validation
# data, its performance will be checked on unseen test data.
# If the performance is still good we can proceed to next step.

# 6. Deploy the algorithm on live streaming sata to classify the outcomes.
# Eg a recommender on a web site.



# Statistical fundamentals and terminology for model building and validation
# The following definitions decrive various fundamentals

# Population :  The is the totality, the complete list of observations
# or all the data points and the subject under study

# sample: A subset of the population , usually a small portion of the
# population that is being analyzed

# Mean This si simple arithmetic average, which is computed by taking
# the aggregated sum of values dived by the count of those values.
# The mean is sensitive to outliers
# An outlier is the value of a set or column that is high deviant from
# the may other values in the same data.
# It usually has very high or low values.

# Median : It is the midpoint of the data

# Mode:  This is the most repetitive data point in the data


# Measure of variance : Dispersion in the variation of the data and
# measures the inconsistencies in the value of variables in  the data
# Dispersion actually provides an idea about the spread than central values

# Range:  Difference between the max and min of the value

# Variance   This mean of squared deviations from the mean(xi = data points,
# U = mean of the data , N = number of data points) The dimension of
# variance is the square of the actual values. The reason to use
# denominator N - 1 for a sample instead of N in the population is due
# the degree of freedom. 1 degree of freedom lost in a sample
# by the time of calculating variance is due ti extraction of substitution
# of sample
# Standard deviation:  This is the square root of variance. By applying
# Square root of variance, we measure the despersion with respect to
# the original variable rather than the square of the dimension

# Demonstrating the functionality of numpy to calculate mean

import numpy as np
from scipy import stats

data = np.array([4, 5, 1, 2, 7, 2, 6, 9, 3])

# Calculate Mean

dt_mean = np.mean(data)

print("Mean : ", dt_mean)

# Calc Medium

dt_medium = np.median(data)

print("Median :", dt_medium)

# Cal Mode

dt_mode = stats.mode(data)

print("Mode :", dt_mode[0][0])


# Statistical Summaries of the Rocks vs Mines Data Set

# After determining which attributes are categorical and which
# are numeric, you'll want some descriptive statistics for the
# numeric variables and a count of the unique categories in
# each categorical attribute


__author__ = 'gwhyte'

import urllib2
import sys

# read data from UCI repository
# url "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"


target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/undocumented/connectionist-bench/sonar/sonar.all-data"

data = urllib2.urlopen(target_url)


# arrange data into a list for labels and lists of list for attributes

xList =[]
labels = []

for line in data:
    # split on comma
    row = line.strip().split(",")
    xList.append(row)

nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []

# generate summary statistics for col 3
col = 3
colData = []

for row in xList:
    colData.append(float(row[col]))

colArray = np.array(colData)

colMean = np.mean(colArray)

colsd = np.std(colArray)

sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                 "Standard Deviation = " + '\t ' + str(colsd) + "\n")

# Calculate quantile Boundaries

ntiles = 4

percentBdry = []

for i in range(ntiles + 1):
    percentBdry.append(np.percentile(colArray, i * (100)/ntiles))


sys.stdout.write("Boundaries for 4 equal percentiles \n")

print (percentBdry)

sys.stdout.write("\n")

ntiles = 10

percentBdry = []

for i in range(ntiles + 1):
    percentBdry.append(np.percentile(colArray, i * (100)/ntiles))


sys.stdout.write("Boundaries for 10 equal percentiles \n")

print (percentBdry)

sys.stdout.write("\n")

# The last column contains categorical values

col = 60

colData = []

for row in xList:
    colData.append(row[col])

unique = set(colData)

sys.stdout.write("Unique Label Values \n")
print(unique)

# count up the number of elements have each value

catDict = dict(zip(list(unique), range(len(unique))))

catCount = [0] * 2

for elt in colData:
    catCount[catDict[elt]] += 1

sys.stdout.write("\n Counts for Each Value of Categorical Label \n")

print(list(unique))

print(catCount)

# The first section of the code picks up one column of numeric data
# and then generates some statistics for it.
# The first step is to calculate the mean and St Deviation for the chosen attribute
# Knowing these will undergrind your intuition.

# The next section of the code looks for outliers, here's how that works
# Suppose that you're trying to determine whether you got outliers
# for the following set of numbers
# [0.1, 0.15, 0.2,0.25,0.3,0.35,0.4,4]
# This example constructed has an outlier, as the last number 4 is clearly
# out of scale.


# One way to reveal this sort opf mismatch is to divide a set of numbers
# into percentiles for example the 25th percentile contains the smallest 25
# percent of the data. The 50th percent contains the smallest 50 percent of
# data.

# Note the upper quartile is much larger than the others.

# Visulisation of outliers using quantile-quantile plot






