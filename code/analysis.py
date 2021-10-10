import numpy as np
from sklearn import linear_model, metrics

class analysis:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

        if self.design == "Yes" or self.design == "yes":
            self.design_matrix()

    def design_matrix(self):
        if len(self.x.shape) > 1:
            self.x = np.ravel(self.x)
            self.y = np.ravel(self.y)

        self.N = len(self.x)
        self.P = int((self.degree+1)*(self.degree+2)/2)  # Number of elements in beta
        X = np.ones((self.N,self.P))

        for i in range(1,self.degree+1):
            q = int((i)*(i+1)/2)
            for j in range(i+1):
                X[:,q+j] = (self.x**(i-j))*(self.y**j)

        self.X = X      

    def MSE(self):
        """
        Calculates the Mean Squared error score.

        Inputs:
        None

        Outputs:
        self.MSE - MSE score 
        """

        self.regression(self.method)
        n = np.size(self.y_pred)
        self.MSE = 1/n * np.sum((self.y-self.y_pred)**2)
        return self.MSE

    def R2(self):
        """
        Calculates the R^2 score.

        Inputs:
        None

        Outputs:
        self.R2 - R^2 score 
        """
        self.regression(self.method)
        self.R2 = 1 - np.sum((self.y-self.y_pred)**2)/np.sum((self.y-np.mean(self.y))**2)
        return self.R2

    def conf_interval(self):
        self.regression(self.method)
        cov = np.var(self.y_pred)*np.linalg.pinv(self.X.T @ self.X)
        std = np.sqrt(np.diag(cov))
        return std
        

    def regression(self,regression_method):
        if self.method == "OLS" or self.method == "ols":
            self.y_pred = self.X @ (np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y)

        elif self.method == "Ridge" or self.method == "ridge":
            if self.lmd == 0:
                raise Exception("Lambda must be greater than zero")
            self.y_pred = self.X @ (np.linalg.pinv(self.X.T @ self.X + self.lmd*np.identity(self.P)) @ self.X.T @ self.y)
       
        elif self.method == "Lasso" or self.method == "lasso":
            # manual lasso regression is optional, can implement if needed, see lectures week 36
            if self.lmd == 0:
                raise Exception("Lambda must be greater than zero")
            lasso_reg = linear_model.Lasso(fit_intercept=True, max_iter=100000, alpha=self.lmd)
            lasso_reg.fit(self.X,self.y)
            self.beta = lasso_reg.coef_
            self.beta[0] = lasso_reg.intercept_
            self.y_pred = self.X @ self.beta

        else:
            raise Exception("Available keywords are 'OLS, ols, Ridge, ridge, Lasso and lasso'")

    
    def resampling(self,resampling_method):
        if self.resampling_method == "Bootstrap" or self.resampling_method == "bootstrap":
            t = np.zeros(len(self.P))
            n = len(self.x)

            for i in range(len(self.P)):
                t[i] = np.mean(self.x[np.random.randint(0,n,n)])
            
            self.t = t

        if self.resampling_method == "Cross-validation" or self.resampling_method == "cross-validation":
            print("nothing")


    def tests(self,run_tests):
    # not finished, but easy to expand upon
        if self.run_tests == True:
            self.MSE()
            skl_MSE = metrics.mean_squared_error(self.y, self.y_pred)
            self.R2()
            skl_R2 = metrics.r2_score(self.y, self.y_pred)

            