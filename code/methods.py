import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold


def design_matrix(x, y, degree):
	"""
	Implements the Nd polynomial design matrix of a given degree based on a dataset.

	Args:
		x (np.Array[float]):        	x-data
		y (np.Array[float]):        	y-data
		degree (int):					N degree of the polynomial

	Returns:
		X (np.Array['z, n', float]):	Design matrix X

	"""

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
	"""
	A class to represent a regression instance.

	Attributes:
		X:	np.Array['x, y, n', float]
			The design matrix X of the training set
		z:	np.Array['x, y, n', float]
			Function output to apply regression on
		regression_method:	string
			Regression method. Choose between {ols, ridge, lasso}
		lambda_:	float
			Parameter lambda, here "k": https://en.wikipedia.org/wiki/Ridge_regression
		beta:	np.Array['b, n', float]
			Confidence intervals beta
		z_pred:	np.Array['x, y', float]
			Predicted regression values of the training dataset
		z_pred_test:	np.Array['x, y', float]
			Predicted regression values of the testing dataset

	Methods:
		get_prediction():
			Returns the predicted regression values.
		get_beta():
			Returns the covariance intervals beta.
	
	"""


	def __init__(self, X, z, regression_method, lambda_):
		"""
		Constructs the necessary attributes for the regression instance and applies it.

		Args:
			X (np.Array['x, y, n', float]):	The design matrix
			z (np.Array['x, y, n', float]):	Function output to apply regression on
			regression_method (string):		Regression method. Choose between {ols, skleran_ols, ridge, sklearn_ridge, sklearn_lasso}
			lambda_ (float): 				Parameter lambda, here "k": https://en.wikipedia.org/wiki/Ridge_regression
			beta (np.Array['b, n', float]):	Confidence intervals beta
			z_pred (np.Array['x, y', float]):Predicted regression values of the training dataset
			z_pred_test (np.Array['x, y', float]):Predicted regression values of the testing dataset
		
		"""

		self.X = X
		self.z = z
		self.regression_method = regression_method
		self.lambda_ = lambda_
		self.beta = None
		self.z_pred = None
		self.z_pred_test = None

		reg_set = {'ols', 'sklearn_ols', 'ridge', 'sklearn_ridge', 'sklearn_lasso'}

		if self.regression_method not in reg_set:
			raise MethodNotImplementedError(f"Please set 'reg_method' to a valid keyword. Valid input keywords are {reg_set}")

		self.__predict()


	def __ols(self):
		"""
		Applies the Ordinary Least Squares regression model on data.

		"""
		
		self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z


	def __sklearn_ols(self):
		"""
		Applies the Ordinary Least Squares regression model on data using the sklearn package.

		"""

		ols_reg = linear_model.LinearRegression(fit_intercept=False).fit(self.X,self.z)
		self.beta = ols_reg.coef_.T

	def __ridge(self):
		"""
		Applies the Ridge regression model on data.

		"""

		if self.lambda_ == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		self.beta = np.linalg.pinv(self.X.T @ self.X + self.lambda_*np.identity(self.X.shape[1])) @ self.X.T @ self.z


	def __sklearn_ridge(self):
		"""
		Applies the Ridge regression model on data using the sklearn package.

		"""

		if self.lambda_ == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		ridge_reg = linear_model.Ridge(fit_intercept=False, alpha=self.lambda_).fit(self.X,self.z)
		self.beta = ridge_reg.coef_.T

	   

	def __sklearn_lasso(self):
		"""
		Applies the Lasso regression model on data using the sklearn package.

		"""

		if self.lambda_ == 0:
			raise Exception("Lambda must be greater than zero. Otherwise use OLS.")
		print(self.lambda_)
		lasso_reg = linear_model.Lasso(fit_intercept=False, normalize=False, max_iter=1000000, alpha=self.lambda_).fit(self.X,self.z)
		self.beta = lasso_reg.coef_.T # returns transposed dimensions for some reason, and is horribly slow..


	def __predict(self):
		"""
		Applies the chosen regression method.

		"""

		if self.regression_method == 'ols':
			self.__ols()
		if self.regression_method == 'sklearn_ols':
			self.__sklearn_ols()
		elif self.regression_method == 'ridge':
			self.__ridge()
		if self.regression_method == 'sklearn_ridge':
			self.__sklearn_ridge()
		elif self.regression_method == 'lasso':
			self.__sklearn_lasso()


	def get_beta(self):
		"""
		Returns the covariance intervals beta.

		Args:
			None

		Returns:
			beta (np.Array['b, n', float]):		Confidence intervals beta

		"""

		return self.beta


	def get_prediction(self, X):
		"""
		Returns the predicted regression values against test set X.

		Args:
			X np.Array['x, y, n', float]:		The test design matrix X

		Returns:
			z_pred_test (np.Array['x, y', float]):	Predicted regression values on test set

		"""

		self.z_pred_test = X @ self.beta

		return self.z_pred_test



	
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
			X_new, z_new = X_train[resample], z_train[resample]

		return X_new, z_new

		

	def cross_validation(self, k = 5):
		X_train, X_test, z_train, z_test = train_test_split(self.X,self.z, test_size=self.test_ratio)

