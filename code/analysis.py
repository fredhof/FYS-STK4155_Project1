import numpy as np
from sklearn import linear_model, metrics, model_selection

class analysis:
    def __init__(self, z, z_pred, X = None):

        self.z = z # Original function output
        self.z_pred = z_pred  # Regression output, z_pred = X @ beta
        self.X = X # the 2D design matrix
        self.run_tests = False # Boole, if True, runs tests on methods    

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
        self.R2_score = 1 - np.sum((self.z-self.z_pred)**2)/np.sum((self.z-np.mean(self.z))**2)
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


    def tests(self,run_tests= True,error = 1e-12):
    # not finished, but easy to expand upon
        if self.run_tests == True:
            self.MSE()
            skl_MSE = metrics.mean_squared_error(self.z, self.z_pred)
            self.R2()
            skl_R2 = metrics.r2_score(self.z, self.z_pred)
            #lin_reg = linear_model.LinearRegression()
            #lin_reg.fit(self.z,self.z_pred)

            #R2 = model_selection.cross_val_score(lin_reg,self.z,self.z_pred)
            #print(np.mean(R2))
            print((self.MSE()-skl_MSE < error))
            print((self.R2()-skl_R2) < error)
            
            