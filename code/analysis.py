import numpy as np
from sklearn import linear_model, metrics, model_selection

class analysis:
    def __init__(self, X, z, z_pred):
        self.X = X # the 2D design matrix  
        self.z = z # Original function output
        self.z_pred = z_pred  # Regression output, z_pred = X @ beta

    def MSE(self):
        """
        Calculates the Mean Squared error score.

        Explicit inputs:
        None

        Implicit inputs:
        z, z_pred (from regression

        Outputs:
        self.MSE - MSE score 
        """


        n = np.size(self.z_pred) # number of elements
        self.MSE_score = 1/n * np.sum((self.z-self.z_pred)**2) # definition of MSE
        return self.MSE_score

    def R2(self):
        """
        Calculates the R^2 score.

        Inputs:
        None

        Implicit inputs:
        z, z_pred (from regression)

        Outputs:
        self.R2 - R^2 score 
        """
        self.R2_score = 1 - np.sum((self.z-self.z_pred)**2)/np.sum((self.z-np.mean(self.z))**2) # possible axis=1 arg in mean
        return self.R2_score

    def conf_interval(self):
        """
        Calculates the confidence interval

        Inputs:
        None

        Implicit inputs:
        z_pred, X

        Outputs:
        std - the variation in the beta parameter
        """
      
        cov = np.var(self.z_pred)*np.linalg.pinv(self.X.T @ self.X)
        std = np.sqrt(np.diag(cov))
        return std

    def bias(self):
        self.bias_score = np.mean((self.z_test - np.mean(self.z_pred))**2)
        return self.bias_score

    def variance(self):
        self.var = np.mean(np.var(self.z_pred,axis=1))
        return self.var


    def run_tests(self, error = 1e-12):
        self.MSE()
        skl_MSE = metrics.mean_squared_error(self.z, self.z_pred)
        self.R2()
        skl_R2 = metrics.r2_score(self.z, self.z_pred)
        #lin_reg = linear_model.LinearRegression()
        #lin_reg.fit(self.z,self.z_pred)

        #R2 = model_selection.cross_val_score(lin_reg,self.z,self.z_pred)
        #print(np.mean(R2))
        print((self.MSE()-skl_MSE < error))
        print((self.R2()-skl_R2 < error))
        
            