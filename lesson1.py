# Understand the problem by understanding the data

# this course is about opening the data set and understanding the data

# Its about getting appreciation for what you'll be able to do with the the data.

# You will start thinking about model building

# The purpose of this lecture is to start to understand the data set that will be used
# later as examples of different types of problems to be solved using
# 1. Penalised Linear Regression
# 2. Ensemble Methods


# I will demonstrate tools available in Python for data exploration.

# Lets talk about installing Python on Linux and Unix systems

# Firstly I will show you what happens on a mac

# Conda

# Setting up the environment - Pycharm

# Git source control


# The anatomy of a new problem

# This course  start with a matrix full of numbers and perhaps some character variables

# User ID  Attribute 1, Attribute 2 ,  Attribute 3  Labels
# 001          6.5         Male           12          $120
# 004          4.2         Female         17          $270
# 007          5.7         Male           3           $75
# 008          5.8         Female         8           $600


# Data in the above set is arranged in rows and columns
# Each row is an individual case(also called an instance, example or observation)
# The columns are designated as attributes will be used to make predictions,
# of the dollars spent on books
# The column designated as labels you will see how much customer spent on books last year.
# a row represents an individual customer
# And the data in that row all pertain to an individual customer.
# # The first column is called an Unique identifier may or may not be present in your problem.
# Web sites typically tag site visits with a unique id with each visit

# An id is usually assigned to each observation
# Columns 2,3,4 represents height, gender, no books purchased
# The point is to highlight there role in the prediction process.
# Attributes are data available in the prediction process.
# Labels are things you want to predict

# In this example Userid is a simple number, Attribute 1 , is height, Attribute 2 is Gender
# Attribute 3 is how many books the person read last year.

# the Columns under labels contains how much money the individuals spent on books online
# last year

# Question
# What role that these different categories of variables play.
# What use does a machine learning algorithm make of user id, attributes and labels ?
# The short answer is You ignore the user id. You use the attributes to predict the
# labels

# The unique id is for bookeeping purposes and allows you to refer back to other data
# available for the specific case.
# Generally a unique id does not get used directly in ML algorithm

# Labels are the observed outcomes that ML algorithm will use to build a predictive model.
# User id doesnt get used in make predictions as it is too specific, it will
# only pertain to single example
# The trick is to build a model to generalise to new cases(not merely memorising past
# cases

# to archive that the algorithm must be derived so that it is forced to pay attention
# to more than one row of data.
# one exception to the excluding user id is when the user id is numeric and assigned in the
# order they are signed up . Basically its indicating sign up date in that case it can be
# useful because users with close IDs signed up at similar times can be considered
# as a group on that basis

# the process of building a predictive model is called training

# the algorithm postulates a predictive relationship between the attributes and labels
# observes the mistakes and makes ome correction and then iterates on that process until a
# a sound model is achieved.

# Different types pf Attributes and Labels Drive Modelling Choices

# The attributes shown come in two different types, numeric variables and categorical
# or Factor variables. Attribute 1 (height is numeric) Attribute 2 is gender and indicated
# by male or female. The type of attribute is called a categorical or factor variable.
# Categorical variables have the property that there's no order relation between
# the values.
# There is no sense in Male <  Female.
# Categorical variables can be two-valued. like Male and Female or like states (AL, AK)
# Other distinctions can be drawn regarding attributes(integer vs float), but they do
# have the same impact on machine learning algorithm.
# The reason for this is that many machine learning algorithms take numeric
# attributes only; They cannot handle categorical or factor variables.
# Penalised linear regression cannot handle categorical variables.

# Penalised linear regression only handles numeric variables.
# The same is true fi Support Vector Machine, kernel methods and K nearest neighbours
# We will cover methods for converting categorical variables to numeric variables
# The nature of the variables will shape your algorithm
# choices and the direction you take in developing a predictive model.
# It s one of the things you need to pay attention to when you face a new problem

# A similar dichotomy arises for labels.

# the labels shown in the above example are numeric.
# In Other problems the labels may also be categorical.
# Eg if the job was to predict which individuals would earn more than $200 next year
# the problem would change.

# the new problem of predicting customers that would spend more than $200 would be

# Labels        >$200
# 120           False
# 270           True
# 75            False
# 600           True

# When the labels are numeric the problem is called a regression problem
# When the label are categorical the problem is called a classification problem

# If the categorical target takes only two values, the problem is called a
# binary classification problem.
# If it takes more than two values it is called a multiclass classification problem

# In many cases the choice of the problem is up to the designer
# As you have seen this example can be changed from regression problem to classification
# problem by a change of labels, These are the tradeoff as as an ML designer you make.

# Items to check about your new dataset

# You want to ascertain the number of other features of the data set as part of your
# initial inspection of the data.
# The following is a check list and a sequence of things to learn
# about your data set

# 1. Number of rows and columns
# 2. Number of categorical variables and number of numeric values for each
# 3. Missing values
# 4. Summary stats for attributes and labels

# One of the first thing to check is size and shape of data
# read the data into a list of lists
# the dimension of the outer list is the rows, and the dimension of the inner list
# is the number columns

# the next step in the in the process is to determine how many missing values there are
# in each row.
# The reason for doing this on a row by row basis is that the simplest way
# to deal with this is to throw away instances that aren't complete.

# In many situations this can bias the results , but just a few incomplete examples
# will not make a material difference

# By counting the rows of missing you will know how much of the data set to discard
# If you are working on biological problem where the data is expensive and you have many
# attributes you may not be able to afford to throw out data and you may be able to
# fill in missing values

# The process of filling in missing values is called imputation
# The easiest way is to use the average values of entries in each row.
# A more sophisticated method is to use one of the predictive methods


# Classification problems detecting Unexploded Mines using Sonar

# this section steps though several check that you might make on classification problem
# It starts with measurement of size and shape
# reporting data types and counting missing values so forth
# then it moves on to statistical properties of the data and interrelationships between
# attributes and the labels.

# the ste comes from the UC Irvine Data Repository
# The data result from some experiments to determine if sonar can be used to detect
# unexploded mines left in harbours subsequent to military actions.
# the sonar is a chirped signal
# The means the signal rises or falls in frequency over the duration of the sound pluse.

# The measurements in the data set represent the power measurements collected in the
# sonar receiver at different points in the returned signal.

# for roughly half the example the sonar is illuminating a rock and for the other half
# a metal cylinder having the shape of the mine, hence ROcks vs Mine

# Lets look at the physical characteristic of the Rocks vs Mine data set

# 1. The first thing to do is too determine its size and shape


# The process for determining the number of rows and columns is pretty simple.

# The file is comma delimited with the data for experiment occupying one line of text
# This makes it simple matter to read the line and slit on comma delimiters
# and stack the resulting lists into an outer list containing the whole data set.


__author__ = 'gwhyte'

import urllib2
import sys

# read data from UCI repository
# url "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"


target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases" \
             "/undocumented/connectionist-bench/sonar/sonar.all-data"

data = urllib2.urlopen(target_url)


# arrange data into list for labels and list of lists for attributes

xList = []
labels = []

for line in data:
    #split on comma
    row = line.strip().split(",")
    xList.append(row)

sys.stdout.write("Number of Rows of data = " + str(len(xList)) + '\n')

sys.stdout.write("Number of columns of data = " + str(len(xList[1])))


# As you can see in the sample output, this dataset has 208 rows (lines) and 61 columns
# What difference does  this make

# the number of rows and columns has several impacts on how you proceed

# First the overall size gives you a rough idea of how your training times are going to
# be. For a small data set for rocks vs mines training time will be less than a minute
# If your data set grows to 1000 x 100 the training times will grow to a fraction of a minute
# for penalised linear regression and a few minutes for ensemble

# As the data set gets to several tens of thousands of rows and columns the training
# times will expand to 3 to 4 hours for penalised linear regression and 12 to 24 hours
# for an ensemble method.

# the larger times will have an impact on development times because you will iterate
# a number of times.

# The second most important observation regarding row and column counts is that if
# the data set has many more columns than rows you are more likely to get the best
# prediction with penalised linear regression.

# In nest few lessons balancing performance complexity and big data will give you
# a better understanding of why that is true

# the next step on the check list is to determine how many columns of the data are
# numeric vs categorical

# the code below runs down each colum and adds up the number of entries that are
# nonempty string  , numeric(int or float) and the number that are empty

for line in data:
        # split on comma
        row = line.strip().split(",")
        xList.append(row)

nrow = len(xList)
ncol = len(xList[1])

type = [0] * 3
colcounts = []

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a,float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1]  += 1
            else:
                type[2] +=1
    colcounts.append(type)
    type = [0] * 3

sys.stdout.write("Col#" + '\t'  + "Number" + '\t' + "Strings" '\t' + "Other\n")

iCol = 0

for types in colcounts:
    sys.stdout.write(str(iCol) +
                     '\t\t' + str(types[0]) + '\t\t' + str(types[1]) +
                     '\t\t' + str(types[2]) + "\n")
    iCol += 1

