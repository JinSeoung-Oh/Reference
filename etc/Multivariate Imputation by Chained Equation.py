"""
1. Overview of MICE
   Multivariate Imputation by Chained Equations (MICE) is a model-based, iterative imputation method that minimizes bias and uncertainty 
   by generating multiple imputed datasets and combining them for valid statistical inference. 
   Unlike one-time imputations, MICE iteratively refines missing values using predictive models and uncertainty estimation.

2. Mechanism
   -a. Initial Imputation: Replace all missing values with basic placeholders such as mean or median.
   -b. Posterior Modeling and Sampling: For each variable with missing data, a predictive model is trained to estimate its conditional posterior distribution, 
                                        from which imputed samples are drawn.
       -1. For regression tasks:
           -1) Normal Linear Regression – samples from Normal distribution based on predicted mean and residual variance.
           -2) Bootstrap Regression – estimates parameters from bootstrapped samples.
           -3) Predictive Mean Matching (PMM) – matches predicted means to observed “donor pool” samples with minimal distance and randomly selects one.
       -2. For classification tasks:
           -1) Bayesian Logistic Regression – draws samples from Bernoulli distribution.
           -2) Polytomous Logistic Regression – samples from Multinomial distribution.
           -3) Proportional Odds Regression – shares slope coefficients across cumulative categories for efficiency.
       -3. Chained Prediction: The imputation of one variable depends on previously imputed variables, creating a chain of predictions.
       -4. Convergence: Repeat until imputed values stabilize.
       -5. Multiple Datasets: The entire process is repeated M times to produce multiple imputed datasets with different sampled values.
       -6. Uncertainty Analysis and Pooling: Statistical analyses are performed on each dataset, and Rubin’s Rules combine results into a pooled estimate:
           -1) θ̄ = average of M parameter estimates
           -2) Total variance T = Ū (within-imputation variance) + (1 + 1/M)B (between-imputation variance)
           -3) Pooled SE = √T

3. Key Concepts
   -a. Chained Equation Structure: Each variable’s imputation leverages results from prior imputations.
   -b. MAR Assumption (Missing At Random): Missingness depends only on observed variables.
       -1. MCAR: completely random → use simple statistical or model-based methods.
       -2. MNAR: depends on unobserved values → use domain-specific or autoencoder methods.
   -c. Best Use Cases:
       -1. Multivariate datasets with correlated missing values
       -2. Mixed continuous/categorical features
       -3. High missingness (~10% or more)
       -4. Ideal for unbiased estimation in clinical and public health research
   -d. Limitations: Computationally slow, algorithmically complex, and sensitive to model assumptions such as linearity.

"""
##### Step 1. Structure Missing Values
import pandas as pd
import numpy as np

def structure_missing_values(df: pd.DataFrame, target_cols: list = []) -> pd.DataFrame:
    target_cols = target_cols if target_cols else df.columns.tolist()
    
    # list up unstructured missing val options
    unstructured_missing_vals = ['', '?', ' ', 'nan', 'N/A', None, 'na', 'None', 'none']

    # structure
    structured_nan = { item: np.nan for item in unstructured_missing_vals }
    structured_df = df.copy()
    for col in target_cols: structured_df[col].replace(structured_nan)

    return structured_df


df_structured = structure_missing_values(df=original_df)

##### Step 2. Analyzing Missing Values
from scipy import stats

def assess_missingness_diagnostics(df, target_variable='X1', missingness_type='mcar'):
    # create a missingness indicator
    df_temp = df.copy()
    df_temp['is_missing'] = df_temp[target_variable].isna().astype(int)
    observed_vars = [col for col in df_temp.columns if col not in [target_variable, 'is_missing']]
    
    # compare means of observed variables (x2, y) between 1) group_observed (x1 observed) and 2) group_missing (x1 missing)
    for var in observed_vars:
        # create the groups
        group_observed = df_temp[df_temp['is_missing'] == 0][var]
        group_missing = df_temp[df_temp['is_missing'] == 1][var]
        
        # check if enough samples exist in both groups
        if len(group_missing) < 2 or len(group_observed) < 2: continue

        # perform two-sample t-test to compute the mean difference
        _, p_value = stats.ttest_ind(group_observed.dropna(), group_missing.dropna())
        mean_obs = group_observed.mean()
        mean_miss = group_missing.mean()
        
         # rejection of h0 (equal means) suggests missingness depends on the observed variable
        if p_value < 0.05:
            print(f"  -> conclusion: mar or mnar because means are statistically different.")
        else:
            # failure to reject h0 suggests independence
            print(f"  -> conclusion: mcar because means are not statistically different.")


assess_missingness_diagnostics(
    df=df_structured,
    target_variable='X1', 
    missingness_type='mar'
)

##### Step 3. Defining the Imputer

from sklearn.impute import IterativeImputer

imputer_mice_pmm = IterativeImputer(
    estimator=PMM(),            # pmm
    max_iter=10,                # num of cycles     
    initial_strategy='mean',    # initial imputation val
    random_state=42,            # for reproducibility
)
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.base import BaseEstimator, RegressorMixin


# implement a custom pmm estimator
class PMM(BaseEstimator, RegressorMixin):
    def __init__(self, k: int = 5, base_estimator=LinearRegression()):
        self.k = k # k-neigbor
        self.base_estimator = base_estimator
    
    def fit(self, observed_X, observed_y):
        self.base_estimator.fit(observed_X, observed_y) 
        
        # store the observed data
        self.X_donors = observed_X.copy()
        self.y_donors = observed_y.copy()
        
        # predict the means for all observed donors
        self.y_pred_donors = self.base_estimator.predict(self.X_donors) 
        return self


    # core pmm logic of sampling val from the k nearest neighbors of observed data. x has missing vals
    def predict(self, X):
        # predict the mean for the missing data (recipients)
        y_pred_recipients = self.base_estimator.predict(X)
        imputed_values = np.zeros(X.shape[0])
        
        # perform pmm for each recipient (row in x)
        for i, pred_recipient in enumerate(y_pred_recipients):
            # compute the absolute difference between the recipient's predicted mean and all the donor's predicted means.
            diffs = np.abs(self.y_pred_donors - pred_recipient)
            
            # get the indices that correspond to the k smallest differences (k nearest matches)
            nearest_indices = np.argsort(diffs)[:self.k] # taking k indices plus 1 to avoid an exact match of the imputed val from the prev round
            
            # randomly sample an observed value from the k nearest neighbors (donor_pool)
            donor_pool = self.y_donors[nearest_indices]
            imputed_value = np.random.choice(donor_pool, size=1)[0]
            imputed_values[i] = imputed_value
        return imputed_values


    ## func for IterativeImputer compatibility
    def _predict_with_uncertainty(self, X, return_std=False):
        if return_std:
            return self.predict(X), np.zeros(X.shape[0])  # pmm is semi-parametric. set standard deviation = 0
        return self.predict(X)

from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer_mice_lr = IterativeImputer(
    estimator=BayesianRidge(max_iter=500, tol=1e-3, alpha_1=1e-10, alpha_2=1e-10, lambda_1=1e-10, lambda_2=1e-10),
    max_iter=10,
    initial_strategy='mean',
    random_state=42,
    sample_posterior=True # add random noise drawn from the posterior distribution to the prediction.
)

##### Step 4. Imputation
import pandas as pd
from sklearn.impute import IterativeImputer


def run_mice_imputation(df: pd.DataFrame, imputer: IterativeImputer, M: int = 5) -> list:
    # iteration
    imputed_datasets= [] # subject to analysis and pooling later
    imputed_values_x1 = {}
    missing_indices_x1 = df[df['X1'].isna()].head(3).index.tolist()

    for m in range(M): 
        # setup imputer for each interation (unique random state controls the numpy seed for pmm's sampling)
        setattr(imputer, 'random_state', m)

        # impute df and convert generated array to pandas df
        df_m = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # recording
        imputed_datasets.append(df_m)
        imputed_values_x1[m] = df_m['X1'].loc[missing_indices_x1].tolist()

    # outputs - unique imputation values
    print("Imputed values across M datasets (True PMM - expecting variability):")
    print(f"imputed dataset 1 - the first three imputed values for X1: {[f'{x:.14f}' for x in imputed_values_x1[0]]}")
    print(f"imputed dataset 2 - the first three imputed values for X1: {[f'{x:.14f}' for x in imputed_values_x1[1]]}")
    print(f"imputed dataset 3 - the first three imputed values for X1: {[f'{x:.14f}' for x in imputed_values_x1[2]]}")
    print(f"imputed dataset 4 - the first three imputed values for X1: {[f'{x:.14f}' for x in imputed_values_x1[3]]}")
    print(f"imputed dataset 5 - the first three imputed values for X1: {[f'{x:.14f}' for x in imputed_values_x1[4]]}")

    return imputed_datasets


##### Step 5. Uncertainty Analysis and Pooling

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import t

def perform_analysis_and_pooling(imputed_datasets, target_param='X1'):
    m = len(imputed_datasets)
    estimates = []  # theta_m
    variances = []  # within-imputation variance u_m

    # 1. analysis - run the model on each imputed dataset
    for i, df_m in enumerate(imputed_datasets):
        # ols model
        model = smf.ols(formula='Y ~ X1 + X2', data=df_m).fit()
        # extract the estimate (theta_m) and its variance (u_m) for the target parameter
        estimate = model.params[target_param]
        variance = model.bse[target_param]**2
        
        estimates.append(estimate)
        variances.append(variance)

    # 2. pooling w rubin's rules
    # pooled point estimate (theta_bar)
    theta_bar = np.mean(estimates)
    
    # within-imputation variance (u_bar)
    u_bar = np.mean(variances)
    
    # between-imputation variance (b)
    b = (1 / (m - 1)) * np.sum([(est - theta_bar)**2 for est in estimates])
    
    # total variance (t)
    t_total = u_bar + (1 + (1 / m)) * b
    
    # total standard error (se)
    se_pooled = np.sqrt(t_total)
    
    # relative increase in variance (riv) and degrees of freedom (v)
    riv = ((1 + (1 / m)) * b) / u_bar
    v_approx = (m - 1) * (1 + (1 / riv))**2 
   
    # confidence interval
    t_critical = t.ppf(0.975, df=v_approx)
    ci_lower = theta_bar - t_critical * se_pooled
    ci_upper = theta_bar + t_critical * se_pooled

    print("\n--- pooled results using rubin's rules ---")
    print(f"pooled estimate ({target_param}): {theta_bar:.10}")
    print(f"within variance (u_bar): {u_bar:.10}")
    print(f"between variance (b): {b:.10f}")
    print(f"total variance (t): {t_total:.10}")
    print(f"pooled standard error: {se_pooled:.10}")
    print(f"relative increase in variance (riv): {riv:.10}")
    print(f"degrees of freedom (approx): {v_approx:.2f}")
    print(f"95% ci: [{ci_lower:.10}, {ci_upper:.10}]")
    
    return theta_bar, se_pooled, t_total, v_approx

##### Step 6. Model Training and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


mse_train_mice_pmm_list = []
mse_val_mice_pmm_list = []
for i, df in enumerate(created_datasets):
    # create X, y
    y = df['Y']
    X = df[['X1', 'X2']]
 
    # create train, val, test datasets
    test_size, random_state = 1000, 42
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=test_size, shuffle=True, random_state=random_state)

    # preprocess
    num_cols = X.columns
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),], remainder='passthrough')
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # train the model
    model = SVR(kernel="rbf", degree=3, gamma='scale', coef0=0, tol=1e-5, C=1, epsilon=1e-5, max_iter=1000)
    model.fit(X_train, y_train)

    # inference
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_pred_train, y_train)
    y_pred_val = model.predict(X_val)
    mse_val = mean_squared_error(y_pred_val, y_val)

    # recording
    mse_train_mice_pmm_list.append(mse_train)
    mse_val_mice_pmm_list.append(mse_val)


# final performance metric that accounts for imputation uncertainty.
pooled_mse_train_mice_pmm = np.mean(mse_train_mice_pmm_list)
pooled_mse_val_mice_pmm = np.mean(mse_val_mice_pmm_list)
print(f'\nfinal performance - train mse: {pooled_mse_train_mice_pmm:.6f}, val mse: {pooled_mse_val_mice_pmm:.6f}')
