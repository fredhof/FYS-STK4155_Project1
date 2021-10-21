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
		self.P = self.X.shape[1] # number of columns in design matrix, also len(beta)

		reg_set = {'ols','skl_ols','ridge','skl_ridge', 'skl_lasso'}

		if self.reg_method not in reg_set:
			raise Exception(f"Please set 'reg_method' to a valid keyword. Valid input keywords are {reg_set}")


	def ols(self):
		"""
		Does ordinary least squares linear regression with matrices
		"""
		self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z

	def skl_ols(self):
		ols_reg = linear_model.LinearRegression(fit_intercept=False).fit(self.X,self.z)
		self.beta = ols_reg.coef_.T

	def ridge(self): 
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		self.beta = np.linalg.pinv(self.X.T @ self.X + self.lmd*np.identity(self.P)) @ self.X.T @ self.z
	   
	def skl_ridge(self):
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		ridge_reg = linear_model.Ridge(fit_intercept=False, alpha = self.lmd).fit(self.X,self.z)
		self.beta = ridge_reg.coef_.T

	def skl_lasso(self):
		# manual lasso regression is optional, can implement if needed, see lectures week 36
		if self.lmd == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		lasso_reg = linear_model.Lasso(fit_intercept=False, max_iter=1000000, alpha=self.lmd).fit(self.X,self.z)
		self.beta = lasso_reg.coef_.T # returns transposed dimensions for some reason, and is horribly slow..


	def predict(self):
		if self.reg_method == 'ols':
			self.ols()
		elif self.reg_method == 'skl_ols':
			self.skl_ols()
		elif self.reg_method == 'ridge':
			self.ridge()
		elif self.reg_method == 'skl_ridge':
			self.skl_ridge()	
		elif self.reg_method == 'skl_lasso':
			self.skl_lasso()
	
		self.z_pred = X @ self.beta
		return self.z_pred


	def return_beta(self):
		return self.beta

	
class resampling:
	def __init__(self, X, z, resampling_method = None, test_ratio = 0.25):
		self.X = X # design matrix
		self.z = z # function values
		self.resampling_method = resampling_method # String that specifices resampling method
		self.test_ratio = test_ratio # ratio of train/test split 
		# self.N number of bootstraps
		# self.k number of folds

		resampling_set = {'bootstrap', 'cross-validation'}

		if self.resampling_method not in resampling_set:
			raise Exception(f"Please set 'resampling_method' to a valid keyword. Valid input keywords are {resampling_set}")


	def bootstrap(self, N = 100):
		X_train, X_test, z_train, z_test = train_test_split(self.X,self.z, test_size=self.test_ratio)

		n = X_train.shape[0]


		for i in range(N):
			resample = np.random.randint(0,n,n)
			X_train_new, z_train_new = X_train[resample], z_train[resample]

		return X_train_new, z_train_new

		

	def cross_validation(self, k = 5):
		X_train, X_test, z_train, z_test = train_test_split(self.X,self.z, test_size=self.test_ratio)

