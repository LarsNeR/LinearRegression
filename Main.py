import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, model_selection

# Load dataset (you can also load your own with pandas but sklearn offers a range of different datasets)
(X, y) = datasets.load_boston(return_X_y=True)

# Preprocess it with sklearn (not necessary, but improves gradient descent)
X = preprocessing.scale(X)

# Divide dataset into train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65)

# Instantiate a new LinearRegression and call its fit-method with the train data
lr = LinearRegression()
lr.fit(X=X_train, y=y_train)

# Predict the result with the test data and calculate the absolute costs between the prediction and the target values
y_pred = lr.predict(X_test)
costs = np.absolute(y_pred - y_test).sum()

# Print interesting information
print(costs)
print(costs / len(X_test))

# Plot costs per iteration
plt.plot(lr.costs_per_iter, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total costs')
plt.show()
