# In this lesson we are taking a few steps back to look at linear regression,
# from a simple idea of fitting a straight line through data to ridge regression
# on our next lesson



# The boston dataset is perfect to play around with regression.
# The boston dataset has the median home price of several areas in Boston.
#  It also has other factors that might impact housing prices,
# for example, crime rate.

# First Import the datasets


from sklearn import datasets
boston = datasets.load_boston()

print(boston.data)

print(boston.target)

# First, import the LinearRegression object and create an object:



from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# Now, it's as easy as passing the independent
# and dependent variables to the method of LinearRegression:

lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

# Now, to get the predictions, do the following:
predictions = lr.predict(boston.data)

# So, going back to the data, we can see which factors have a negative
# relationship with the outcome, and also the factors that have a
# positive relationship. For example, and as expected, an increase
# in the per capita crime rate by town has a negative relationship
# with the price of a home in Boston. The per capita
# crime rate is the  first coefficient in the regression.

print("The Coefficients" , lr.coef_)

print(predictions)

# we minimize the error term. This is done by minimizing
# the residual sum of squares.

# The LinearRegression object can automatically normalize (or scale) the inputs:

lr2 = LinearRegression(normalize=True)
lr2.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
predictions2 = lr2.predict(boston.data)


# In this section  , we'll look at how well our regression
# fits the underlying data.
# We  look at a regression in the last display , but didn't pay much attention
# to how well we actually did it.
#  The  first question after we  fit the model was clearly
# "How well does the model  fit?" In this section, we'll examine this question.
# Evaluation the linear Regression Model


# The lr object will have a lot of useful methods now that the model has been  fit.

import matplotlib.pyplot as plt
import numpy as np


f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)
ax.hist(boston.target - predictions, bins=50)
ax.set_title("Histogram of Residuals.")

# The error terms should be normal, with a mean of 0.
# The residuals are the errors; therefore, this plot should be approximately
# normal. Visually, it's a good  t, though a bit skewed. We can also look
# at the mean of the residuals, which should be very close to 0:

plt.show()

# print("The mean is ", np.mean(boston.target - predictions))


# Q-Q plot.

# Here, the skewed values we saw earlier are a bit clearer.

from scipy.stats import probplot
f = plt.figure(figsize=(7, 5))
ax = f.add_subplot(111)
probplot(boston.target - predictions, plot=ax)

plt.show()


# We can also look at some other metrics of the  t;
# mean squared error (MSE) and mean absolute deviation (MAD) are
# wo common metrics.
# Let's de ne each one in Python and use them.


def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


print("MSE is ", MSE(boston.target, predictions))
print("MSE after normalisation ", MSE(boston.target, predictions2))


