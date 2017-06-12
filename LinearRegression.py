import numpy as np


class LinearRegression(object):
    """
    Basic LinearRegression

    Parameters
    ----------
    n_iter : int, default 1000
        number of iterations the gradient descent should process

    eta : int, default 0.1
        the learning rate.
        High learning rate: might overshoot the minimum
        Low learning rate: gradient descent could take too long (does not reach minimum within n_iter)

    Attributes
    ----------
    weights : array, shape(n_features, )
        Coefficients for the hypothesis. The weights will be 'fitted' during the gradient descent

    costs_per_iter : array, shape(n_iter, )
        Contains the sum of costs for every iteration
    """

    def __init__(self, n_iter=1000, eta=0.1):
        self.n_iter = n_iter
        self.eta = eta
        self.weights = []
        self.costs_per_iter = []

    def fit(self, X, y):
        """
        Fit LinearRegression with gradient descent

        Parameters
        ----------
        X : numpy array, shape(m_samples, n_features)
            Training data

        y : numpy array, shape (m_samples, )
            Target values
        """
        X = self.add_column_with_ones(X)

        self.weights = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            # Gradient descent
            self.weights -= self.eta * np.dot(X.T, self.costs(X, y)) / len(y)

            # Append current costs (just for visualization purposes)
            self.costs_per_iter.append(np.absolute(self.costs(X, y)).sum())

    def costs(self, X, y):
        return self.X_dot_weights(X) - y

    def X_dot_weights(self, X):
        return np.dot(X, self.weights)

    def predict(self, X):
        """
        Predict the target value(s)

        Parameters
        ----------
        X : numpy array, shape(m_samples, n_features)
            Data to use for prediction

        Returns
        -------
        prediction : numpy array, shape(m_samples, )
            Prediction for every sample in X
        """
        X = self.add_column_with_ones(X)
        return self.X_dot_weights(X)

    def add_column_with_ones(self, X):
        # Add one column with ones (for weight_0, the 'bias')
        # Otherwise you have to calculate weight[0] and weight[1:] separately
        # If you want to use that check the history (before 30.05.2017)
        return np.concatenate([np.ones((len(X), 1)), X], axis=1)
