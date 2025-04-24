### From https://medium.com/@yujiisobe/optimizing-causal-decisions-with-gurobi-machine-learning-a-step-by-step-tutorial-d9cd357dcc3c

"""
1. Context & Motivation
   -a. From Insight to Action: In causal‐effect modeling (e.g., uplift or treatment‐response models), 
                               you learn who is likely to respond to a treatment—but you still face the question: 
                               “Who should actually receive treatment when resources are limited?”
   -b. Decision under Constraints: Uplift models partition individuals (Persuadables, Sure Things, etc.) but don’t by themselves solve
                                   which subset to treat when you can’t treat everyone.
2. Key Idea: Embed ML into an Optimization Pipeline 
    -a. Predictive Model
        -1. Train a causal prediction model (here, logistic regression) to estimate each individual’s probability of a positive outcome
            if treated.
    -b. Optimization Model
        -1. Formulate a mixed‐integer program (MIP) that chooses who to treat (and potentially how much to pay) to maximize total
            expected returns subject to a budget (or other) constraint.
    -c. Gurobi-ML Integration
        -1. The gurobi-ml library (since Gurobi 10.0) translates your trained ML model into MIP-friendly constraints 
            (e.g., piecewise-linear approximations of the logistic function), so the solver can jointly reason about prediction 
            and allocation.
3. Walk-Through Example
   -a. Dataset: Thornton (2008) study in rural Malawi, where individuals were randomly offered varying monetary incentives
                to return for HIV test results.
   -b. Objective: With a fixed total incentive budget, decide who (and how much) to pay to maximize the number of people 
                  who actually return.
   -c. Steps:
       -1. Fit a logistic regression to predict return‐probability as a function of individual features and offered incentive.
       -2. Embed that model into a Gurobi MIP:
           -1) Binary decision variables indicating who to incentivize.
           -2) Budget constraint on total payouts.
           -3) Objective summing predicted return‐probabilities for chosen individuals.
       -3. Solve to get the optimal allocation—the exact set of people (and incentive amounts) that maximizes expected returns
           under the budget.

4. Takeaways
   -a. Bridges ML & OR: This approach marries data-driven causal estimates with rigorous decision optimization.
   -b. Generalizable: Any predictive model (classification, regression) can—in principle—be embedded similarly, 
                      enabling optimal resource allocation across domains.
   -c. Actionable Insights: Rather than just ranking individuals by uplift, you get a concrete, globally optimal treatment plan
                            given your real-world constraints.
"""

# Import necessary packages
import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
from gurobi_ml import add_predictor_constr
from causaldata import thornton_hiv  # dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the Thornton (2008) HIV dataset
data = thornton_hiv.load_pandas().data

# Define feature and target columns
features = ['tinc', 'distvct', 'age']
target = 'got'

# Split data into training and test sets
train, test = train_test_split(data, test_size=0.2, random_state=0)
# For optimization, we'll treat 'tinc' (incentive) as a decision variable.
# Remove the actual 'tinc' and outcome 'got' from the test feature set (they will be decided/predicted).
test = test.drop(columns=['tinc', 'got'])

# (Optional) Wrap StandardScaler to log attribute access for demonstration
class LoggingStandardScaler(StandardScaler):
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr):
            def new_func(*args, **kwargs):
                print(f'Calling StandardScaler.{name}()')
                return attr(*args, **kwargs)
            return new_func
        else:
            print(f'Accessing StandardScaler.{name} attribute')
            return attr

# Wrap LogisticRegression similarly
class LoggingLogisticRegression(LogisticRegression):
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr):
            def new_func(*args, **kwargs):
                print(f'Calling LogisticRegression.{name}()')
                return attr(*args, **kwargs)
            return new_func
        else:
            print(f'Accessing LogisticRegression.{name} attribute')
            return attr
# Use the wrapped classes for transparency
scaler = LoggingStandardScaler()
logreg = LoggingLogisticRegression(random_state=1)

# Create a pipeline and train the model on the training set
pipe = make_pipeline(scaler, logreg)
pipe.fit(X=train[features], y=train[target])

---------------------------------------------
# Create a new Gurobi model
m = gp.Model()

# Add a decision variable y_i for each test instance to represent probability of outcome (got=1)
y = gppd.add_vars(m, test, name="probability")
# Add a decision variable x_i (incentive) for each test instance, with bounds 0 <= x_i <= 3
test = test.gppd.add_vars(m, lb=0.0, ub=3.0, name="tinc")
x = test["tinc"]
# Ensure the DataFrame `test` now has columns [tinc, distvct, age] in the correct order
test = test[["tinc", "distvct", "age"]]
budget = 0.2 * test.shape[0]

# Set objective: maximize sum of y_i probabilities
m.setObjective(y.sum(), gp.GRB.MAXIMIZE)
# Add budget constraint: total incentive sum <= budget
m.addConstr(x.sum() <= budget, name="budget")
m.update()
---------------------------------------------

# Add constraints from the trained ML model (pipeline) to link x, distvct, age to predicted y
pred_constr = add_predictor_constr(m, pipe, test, y, output_type="probability_1")
pred_constr.print_stats()

# Optimize the model
m.optimize()
max_error = np.max(pred_constr.get_error())
tinc_solution = pred_constr.input_values['tinc']  # pandas Series of optimal x_i
tinc_solution = tinc_solution.apply(lambda x: 0 if x < 0 else np.floor(x * 1e5) / 1e5)
print(tinc_solution.sum() <= budget)  # This should output True

