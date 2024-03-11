### QUESTION 2


import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load the data
df = pd.read_csv('/Users/pratiksha/Downloads/non_profits.csv')

# Create lists for alpha and beta values
alpha = df['alpha_i'].tolist()
beta = df['beta_i'].tolist()

budget = 50000000

# Create a new model
model = gp.Model('NonprofitFunding')

# Number of nonprofits
N = len(df)

# Decision variables
# Decision variables
a = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="a")
# No need for x as the power constraint can be directly enforced

# Objective function
model.setObjective(gp.quicksum(2 * a[i]**(2.0/3.0) for i in range(N)), GRB.MAXIMIZE)

# Constraints
# Budget Constraint
model.addConstr(gp.quicksum(a[i] for i in range(N)) <= budget, "Budget")


# Optimize the model
model.optimize()

# Check the optimization status
if model.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    # Print the decision variables
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")
    print(f"Objective Value: {model.ObjVal}")
elif model.Status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    # Compute and print the IIS (Irreducible Inconsistent Subsystem)
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to file 'model.ilp'")
elif model.Status == GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print(f"Optimization was stopped with status {model.Status}")


print(f"Number of constraints: {model.NumConstrs}") 
print(f"Number of decision variables: {model.NumVars}")

# Sum of decision variables in the optimal solution
sum_decision_variables = sum(v.x for v in model.getVars())
print(f"Sum of decision variables: {sum_decision_variables}")



#%%
import gurobipy as gp
from gurobipy import GRB

import pandas as pd
data = pd.read_csv('/Users/pratiksha/Downloads/non_profits.csv')

# Extract alpha_i and beta_i values
alphas = data['alpha_i'].tolist()
betas = data['beta_i'].tolist()


def utility_function(e, a, alpha, beta):
    return alpha * a - 0.5 * e**2 + 2 * (e * beta * a)**0.5

def utility_function(e, a, alpha, beta):
    return alpha * a - 0.5 * e**2 + 2 * (e * beta * a)**0.5

def optimal_effort(a, beta):
    return (beta * a)**(1/3)


def optimal_output(a, beta):
    return 2 * (optimal_effort(a, beta) * beta * a)**0.5


import gurobipy as gb

# Create the optimization model
model = gb.Model("Nonprofit Allocation")

# Decision variables
N = len(alphas)
allocations = model.addVars(N, lb=0, vtype=gb.GRB.CONTINUOUS, name="Allocation")

# Objective function: Maximize total output
model.setObjective(gb.quicksum(optimal_output(allocations[i], betas[i]) for i in range(N)), gb.GRB.MAXIMIZE)

# Budget constraint: Total allocation cannot exceed $50 million
model.addConstr(gb.quicksum(allocations[i] for i in range(N)) <= 50000000, "Budget_Constraint")

# Step 7: Solve the optimization model
model.optimize()



# (f) Output value in the optimal allocation strategy
print("Output value in the optimal allocation strategy:", model.objVal)

# (g) Percentage of budget allocated in the optimal solution
print("Percentage of budget allocated in the optimal solution:", model.objVal / 50000000 * 100, "%")

# (h) Number of nonprofits receiving no funding
num_nonprofits_no_funding = sum(allocations[i].x == 0 for i in range(N))
print("Number of nonprofits receiving no funding:", num_nonprofits_no_funding)