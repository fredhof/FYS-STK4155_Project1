import numpy as np


""" creates 2D poly design matrix
e.g. 2D 'degree' polynomial where DIM = N x P:

P5_column(x**5, x**4 * y, x**3 * y**2, x**2 * y**3, x * y**4, y**5) + lower degree columns
e.g P2 = [ 1, x, x**2
           1, y, x*y
           1, 1, y**2 ]

"""
def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    P = int((degree+1)*(degree+2)/2)  # Number of elements in beta
    X = np.ones((N,P))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = (x**(i-j))*(y**j)

    return X

class regression:
	def __init__(self, X, z, reg_method = None, lmd = None):
		self.X = X # the design matrix
		self.z = z # the function values that will be done regression on
		self.reg_method = reg_method # String that specifices regression method
		self.lmd = lmd # Lambda variable used in Ridge regression, here "k": https://en.wikipedia.org/wiki/Ridge_regression
		self.z_pred = None # the regression predicted new values for z

		reg_set = {'ols','ridge','lasso'}

		if self.reg_method not in reg_set:
			raise Exception(f"Please set 'reg_method' to a valid keyword. Valid input keywords are {reg_set}")


	def ols(self):
		"""
		Does ordinary least squares linear regression with matrices
		"""
		self.z_pred = self.X @ (np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z)
		return self.z_pred

	def ridge(self):
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS")
		self.z_pred = self.X @ (np.linalg.pinv(self.X.T @ self.X + self.lmd*np.identity(self.P)) @ self.X.T @ self.z)
		return self.z_pred
	   
	def lasso(self):
		# manual lasso regression is optional, can implement if needed, see lectures week 36
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero")
		lasso_reg = linear_model.Lasso(fit_intercept=True, max_iter=100000, alpha=self.lmd)
		lasso_reg.fit(self.X,self.z)
		self.beta = lasso_reg.coef_
		self.beta[0] = lasso_reg.intercept_
		self.z_pred = self.X @ self.beta
		return self.z_pred

	
class resampling:
	def __init__(self, X, z, resampling_method = None):
		self.X = X # design matrix
		self.z = z # function values
		self.resampling_method = resampling_method # String that specifices resampling method

		resampling_set = {'bootstrap', 'cross-validation'}

		if self.resampling_method not in resampling_set:
			raise Exception(f"Please set 'resampling_method' to a valid keyword. Valid input keywords are {resampling_set}")


	def bootstrap(self):
		P = int(len(X[0,:]))
		t = np.zeros(P)
		n = len(self.z)

		for i in range(P):
			t[i] = np.mean(self.z[np.random.randint(0,n,n)])
		
		self.t = t

	def cross_validation(self):

		print("nothing")
