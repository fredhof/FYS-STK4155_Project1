import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold


""" creates 2D poly design matrix
e.g. 2D 'degree' polynomial where DIM = N x P:
N = number of random numbers, P = combinations for 2D 'degree' polynomial
e.g P2_row1 = [ 1, x_1, y_1, x_1**2, x_1*y_1, y_1**2] 
        

"""
def design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    P = int((degree+1)*(degree+2)/2)
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
		self.beta = None # Least squares coefficients
		self.z_pred = None # the regression predicted new values for z
		self.P = None # number of columns in design matrix, also len(beta)

		reg_set = {'ols','ridge','lasso'}

		if self.reg_method not in reg_set:
			raise Exception(f"Please set 'reg_method' to a valid keyword. Valid input keywords are {reg_set}")


	def ols(self):
		"""
		Does ordinary least squares linear regression with matrices
		"""
		self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z
		self.z_pred = self.X @ self.beta

	def ridge(self):
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		self.beta = np.linalg.pinv(self.X.T @ self.X + self.lmd*np.identity(self.P)) @ self.X.T @ self.z
		self.z_pred = self.X @ self.beta
	   
	def lasso(self):
		# manual lasso regression is optional, can implement if needed, see lectures week 36
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		lasso_reg = linear_model.Lasso(fit_intercept=False, max_iter=10000, alpha=self.lmd)
		lasso_reg.fit(self.X,self.z)
		self.beta = lasso_reg.coef_
		self.z_pred = self.X @ self.beta


	def predict(self):
		if self.reg_method == 'ols':
			self.ols()
		elif self.reg_method == 'ridge':
			self.ridge()
		elif self.reg_method == 'lasso':
			self.lasso()
		return self.z_pred


	def return_beta(self):
		return self.beta

	
class resampling:
	def __init__(self, X, z, resampling_method = None, test_ratio = 0.25):
		self.X = X # design matrix
		self.z = z # function values
		self.resampling_method = resampling_method # String that specifices resampling method
		self.test_ratio = test_ratio # ratio of train/test split 
		self.N = None # number of bootstraps
		self.k = None # number of folds

		resampling_set = {'bootstrap', 'cross-validation'}

		if self.resampling_method not in resampling_set:
			raise Exception(f"Please set 'resampling_method' to a valid keyword. Valid input keywords are {resampling_set}")


	def bootstrap(self):
		X_train, X_test, z_train, z_test = train_test_split(self.X,self.z, test_size=self.test_ratio)

		n = X_train.shape[0]


		for i in range(self.N):
			resample = np.random.randint(0,n,n)
			X_new, z_new = X_train[resample], z_train[resample]


		

	def cross_validation(self):

		print("nothing")
