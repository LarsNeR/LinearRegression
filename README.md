# Linear Regression
A simple Linear Regression with Python

#### Main.py
Shows how to use **LinearRegression.py**

#### LinearRegression.py
Contains the Linear Regression using gradient descent

## How to use
You can pull the repository the way it is. Running **Main.py** calls Linear Regression with the boston dataset. If you just want to use the Linear Regression you have to call `fit()` with a Numpy Array `X` (m_samples, 1 + n_features) and a Numpy Array `y` (m_samples). 
(1 + n_features) because I am working with vectorized gradient descent to update all weights simultaneously. So the first column has to be only ones:

`X = [
[2 3]
[5 6]
[9 3]}
`

has to be

`X = [
[1 2 3]
[1 5 6]
[1 9 3]}`

Check **Main.py** line 16 to see how this can be done.

## Contribution
If you have an idea how to improve this Linear Regression keeping it as simple as possible please fork it and make a PR.
