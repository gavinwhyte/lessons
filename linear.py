import numpy as np
import matplotlib.pyplot as plt

# X represents the features of our training data, the diameters of the pizzas.
# A scikit-learn convention is to name the matrix of feature vectors X.
# Uppercase letters indicate matrices, and lowercase letters indicate vectors.

X = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1)
y = [7, 9, 13, 17.5, 18]  # y is a vector representing the prices of the pizzas.
# plt.figure()
# plt.title('Pizza price plotted against diameter')
# plt.xlabel('Diameter in inches')
# plt.ylabel('Price in dollars')
# plt.plot(X, y, 'k.')
# plt.axis([0, 25, 0, 25])
# plt.grid(True)
# plt.show()

# In[2]:
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # Create an instance of the estimator
model.fit(X, y)  # Fit the model on the training data

# Predict the price of a pizza with a diameter that has never been seen before

test_pizza = np.array([[15]])

predicted_price = model.predict(test_pizza)[0]

print('A 11" pizza should cost: $%.2f' % predicted_price)
# print('R-squared: %.2f' % model.score(X, y))


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 3]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()

model.fit(X, y)

X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
       print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
       print('R-squared: %.2f' % model.score(X_test, y_test))
