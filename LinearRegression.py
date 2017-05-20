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

        self.weights = np.zeros(1 + X.shape[1])
        weights_tmp = np.zeros(1 + X.shape[1])
        m_samples = len(y)

        for _ in range(self.n_iter):
            # Weight 0 has to be calculated separately
            weights_tmp[0] = self.weights[0] - self.eta / m_samples * self.sigma(X, y, np.ones(X.shape[0]))

            # Update all weights with gradient descent (in weights_tmp)
            for j in range(len(self.weights) - 1):
                weights_tmp[j + 1] = self.weights[j + 1] - self.eta / m_samples * self.sigma(X, y, X[:, j])

            # Update all weights simultaneously
            self.weights = weights_tmp

            # Append current costs (not necessary, just to visualize gradient descent)
            self.costs_per_iter.append(np.absolute(self.costs(X, y)).sum())

    def sigma(self, X, y, x_feature):
        """
        The sigma in gradient descent (derivative)
        """
        return (np.dot(self.costs(X, y), x_feature)).sum()

    def costs(self, X, y):
        return self.predict(X) - y

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
        return np.dot(X, self.weights[1:]) + self.weights[0]
