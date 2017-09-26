
# Predictive Model Building Balancing Performance, Complexity and Big Data
# The lesson provides the start of the technical definitions of performance for
# different types of machine learning problems.

# In an e-commerce application good performance means returning correct search results
# or presenting ads that site visitors frequently click.
# In a genetic problem it means isolating a few genes responsible for a
# heritable condition.
# The lesson is the start of what describes relevant performance
# measures for these different problems

# The goal of selecting and fitting a predictive algorithm is to achieve the best
# possible performance

# Achieving performance goals involves three factors:
# complexity of the problem, complexity of the algorithmic model employed,
# and the amount and richness of the data available.

# The basic problem understanding function approximation

# The algorithms covered in these lessons address a specific class of predictive problems
# The problem statement for these problems has two types of variables

# The variable you are attempting to predict(for example whether a visitor to a
# website will click an ad)

# Other Variables (for example, the visitors demogrpahics or past behavior on the site)
# that you can use to make the prediction

# Problem of this type are referred to as function approximation problems because the
# goal is to construct a model generating predictions of the first of these as a
# function of the second

# In a function approximation problem, the designer starts with a collection of
# historical example for which the correct answer is known.
# For example, historical web logs files will indicate whether a visitor clicked an ad
# when shown the ad.

# The data scientist might try using other pages that the vistor viewed before seeing the
# ad.

# If the user is registered with the site, data on past purchases or pages viewed
# might be available for making a prediction

# The variable being predicted refers to a number of names, such as target,label
#  and outcome

# The variables used to make the predictions are variously called predictors,
# regressors, features and attributes.

# These terms are used interchangeably in this text.
# Determing what attributes to use for making predictions is called feature
# Engineering

# Feature Engineering usually requires a manual iterative process for selecting
# features, determining optimal potential, and experimenting with Different combinations
# of features.

# The algorithms covered in this book assign numeric importance to each attribute,
# These values indicate the relative importance of attributes in making predictions
# That information speeds up the feature engineering process.

# Working with Training data
# The datascientist starts algorithm development with a training set.
# The training set consists of outcome examples and the assemblage of features chosen
# by the data scientist

# The training set comprises two types of data:

# The outcomes you want to predict
# Features available for making the prediction

# Example of a training set

# Feature 1 - Gender Feature  2 Money spent on site Feature 3 Age Outcomes Clicked
#           M                         0                   25                 Yes
#           F                         250                 32                 No
#           F                         12                  17                 Yes

# The predictor values (AKA Features, Attributes etc) can be arranged in the form
# of a matrix.

# A proper matrix contains variables that are all the same type. Predictors however
# may not all be the same type of variable.
# Using the example above about predicting advertising clicks , the predictors may at
# times include demographic data about the site visitor
# Data could include martial status, yearly income, among other things.
# YEarly income is a real number, marital status is a categorical variable.
# That means that marital status does not admit arithmetic operations such as
# addition or multiplication and that no order relation exists between single,
# married and divorced.
# The entries in a column from X all have the same type, but the type may vary from
# one column to the next

# Attributes such as marital status, gender or the state of residence go by several
# different designations.
# They are called factor or categorical. Attributes like age or income that
# are represented by numbers are called numeric or real-valued. The distinction
# between these 2 types of attributes are important because some algorithms may
# not handle one type or the other.

# For example, linear methods require numeric attributes.
# When we cover penalized linear regression, i will show methods for converting
# categorical variables to numeric in order to apply linear methods to problems with
# categorical variables.

# The targets corresponding to each row in X are arranged in a column vector Y

# Targets may be of several different forms.
# For example they may be real numbers, like if the objective were to predict how much
# a customer will spend
# When the targets are real numbers, the problem is called a regression problem.

# Linear regression implies using a linear method to solve a regression problem
# This lesson covers linear and non-linear regression methods.

# If the targets are two valued the problem is called a binary classification problem.
# Predicting whether a customer will clcik an advertisement is a binary
# classification problem.
# If the target contains several discrete values, the problem is called a
# multiclass classification problem.

# Predicting which of several ads a customer will click will be a multiclass
# classification problem. The basic problem is to find a prediction function pred()
# that uses attributes to predict outcomes

# function pred() uses the attribute Xi to predict yi. This course describes some
# of the very best current methods for producing the function pred()


# Assessing Performance of Predictive Models
# Good performance means using the attributes xi to generate a prediction thats
# close to yi but close has different meanings for different problems.

# for a regression problem where yi is a real number







