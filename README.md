# Bayesian-Logistic-Regression-JAGS
This project implements Bayesian logistic regression using JAGS, with MCMC sampling to estimate posterior distributions. The goal is to predict stroke risk using a combination of continuous variables (e.g., age, BMI, glucose level) and categorical variables (e.g., gender, smoking status, work type).

## **Workflow**
- Data cleaning and standardizing
- Performing stratified sampling to split the data [ training (70%) and test (30%) ]
- Applying SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class
- Structuring the logistic regression model

```math
\text{logit}(p_i) = b_0 + b_{\text{age}} \cdot \text{age}_i + b_{\text{glucose}} \cdot \text{avg\_glucose\_level}_i + b_{\text{bmi}} \cdot \text{bmi}_i + \sum_j b_{\text{cat}_j} \cdot \text{cat}_j[i]
```
and assigning normal priors to coefficients and Dirichlet(1) to categorical variable distributions
```math
\text{stroke}_i \sim \text{Bernoulli}(p_i)
```
```math
b_0, b_{\text{age}}, b_{\text{glucose}}, b_{\text{bmi}}, b_{\text{cat}_j} \sim \mathcal{N}(0, 10^6)
```

- Performing MCMC sampling with three chains and extracting posterior medians
- Assessing convergence using the Gelman-Rubin diagnostic
- Applying the logistic function to predict stroke risk
- Evaluating model performance using a confusion matrix and ROC-AUC
- Testing the model with the original unbalanced data to evaluate the impact of SMOTE

## **Notes**
- This model is implemented in JAGS. Because the posterior distribution of the slope parameters has no closed-form solution, JAGS employs Metropolis-Hastings sampling.
```math
P(\beta \mid D) \propto
\left[
  \prod_{i=1}^N 
    \left(\frac{1}{1 + \exp(-x_i^T \beta)}\right)^{y_i}
    \left(1 - \frac{1}{1 + \exp(-x_i^T \beta)}\right)^{1 - y_i}
\right]
\;\times\;
\prod_{j=0}^k 
  \exp\!\Bigl(-\frac{\beta_j^2}{2 \times 10^6}\Bigr)
```

## **MY MISTAKES**
*To avoid creating dummy variables, I mistakenly modeled categorical predictors (e.g., smoking_status) as latent variables in JAGS, making the model unnecessarily complex. These variables should be treated as fixed inputs, while JAGS can handle them more efficiently. This is fixed.

*The intercept is sampled again due to a prior mistake. Always remember to check if all parameters are tracked.

*Non-informative priors for coefficients slowed down the process, and this issue needs further attention.




