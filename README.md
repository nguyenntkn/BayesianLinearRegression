# BayesianLinearRegression
Comparing the use of Bayesian Linear Regression vs Simple Linear Regression.

Bayesian Regressors: 
- BayesianRidge 
- ARDRegressor 

Evaluation metrics to look into:
- MSE and R2 score.
- Negative Log-Likelihood (NLL) - for probabilistic classification, not regression
- Continuous Ranked Probability Score (CRPS) - not available on sklearn
- Prediction Interval Coverage Probability (PICP)
- Expected Calibration Error (ECE) - mainly for probabilistic classification, may be applied to regression but not relevant

Resources:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
