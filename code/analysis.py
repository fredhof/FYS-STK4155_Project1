import numpy as np
from sklearn import linear_model, metrics, model_selection

class analysis:
    def __init__(self, X, z, z_pred):
        self.X = X # the 2D design matrix  
        self.z = z # TRUE function output, data without noise
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
        return std*1.96 # z = 1.96 is 95% conf interval, divide by sqrt N ?

    def bias(self):
        self.bias_score = np.mean((self.z - np.mean(self.z_pred))**2)
        return self.bias_score

    def variance(self):
        self.var = np.mean(np.var(self.z_pred,axis=1))
        return self.var


    def run_comparison_tests(self, X_compare, z_compare, z_pred_compare, error = 1e-12):
        """
        Compares analysis methods between regression methods.
        Should be used with comparable methods, e.g. ols and skl_ols, not ols and ridge.

        Inputs:
        X_compare, z_compare, z_pred_compare (, error)

        Outputs:
        Runs tests. Prints "Tests successful" if the tests succeeds.
        """

        MSE_score = self.MSE()
        R2_score = self.R2()

        compare = analysis(X_compare,z_compare,z_pred_compare)
        MSE_compare = compare.MSE()
        R2_compare = compare.R2()
        
        assert np.abs(MSE_score - MSE_compare) < error
        assert np.abs(R2_score - R2_compare) < error
        print('Tests successful')
        
            