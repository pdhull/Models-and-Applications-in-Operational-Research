# -*- coding: utf-8 -*-
"""Assignment2_Question1 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fHC0S0LaismWHjrmKD9qnpGP4YCxEK9u
"""

from gurobipy import GRB
import gurobipy as gb
import pandas as pd
import numpy as np
import autograd as ag

# Create the optimization model
model = gb.Model("Question 1a): TechEssentials Certain Product Line")

# Read costs from CSV files
price_response_df = pd.read_csv(r"C:\Users\gabri\Downloads\price_response.csv")

price_response_df.head(10)

# Extract the "Intercept", "Capacity", and "Sensitivity" column
intercept_values = price_response_df['Intercept'].values.reshape(3, -1)
product_capacity = price_response_df['Capacity'].values.reshape(3, -1)
slope = price_response_df['Sensitivity'].values.reshape(3, -1)

intercept_values

product_capacity

slope

"""# Part a)"""

from scipy.optimize import minimize

# Objective function (negated for maximization)
def objective(p):
    p1, p2 = p
    return -1 * (p1 * (intercept_values[0,0] + slope[0,0] * p1) + p2 * (intercept_values[0,1] + slope[0,1] * p2))

# Constraint functions
def constraint1(p):
    return p[1] - p[0]

def constraint2(p):
    return intercept_values[0,0] + slope[0,0] * p[0]

def constraint3(p):
    return intercept_values[0,1] + slope[0,1] * p[1]

# KKT conditions using only constraint1
def kkt_conditions(p, lagrange_multipliers):
    grad_objective = [-(intercept_values[0,0] + 2 * slope[0,0] * p[0]), -(intercept_values[0,1] + 2 * slope[0,1] * p[1])]
    grad_constraint1 = [-1, 1]

    # Stationarity condition
    stationarity = [grad_objective[i] + lagrange_multipliers[0] * grad_constraint1[i] for i in range(len(p))]

    # Complementary slackness conditions
    complementary_slackness = lagrange_multipliers[0] * constraint1(p)

    # Feasibility conditions
    feasibility = constraint1(p)

    return stationarity + [complementary_slackness] + [feasibility]

# Initial guess for Lagrange multipliers
initial_lagrange_multipliers = [1.0]

# Bounds for Lagrange multipliers
bounds_lagrange_multipliers = [(0, None)] * len(initial_lagrange_multipliers)

# Initial guess
initial_guess = [0, 0]

# Bounds for variables
bounds = [(0, None), (0, None)]  # P1 and P2 are non-negative

# Solve using minimize with original objective and constraints
result = minimize(objective, initial_guess, bounds=bounds, constraints=[{'type': 'ineq', 'fun': constraint1},
                                                                       {'type': 'ineq', 'fun': constraint2},
                                                                       {'type': 'ineq', 'fun': constraint3}])

# Extract solution
p_solution = result.x
maximized_profit = -1 * result.fun  # Convert back to positive for interpretation

# Now that p_solution is defined, you can use it in the minimize function for Lagrange multipliers

result = minimize(lambda lagrange_multipliers: sum([val**2 for val in kkt_conditions(p_solution, lagrange_multipliers)]),
                  initial_lagrange_multipliers,
                  bounds=bounds_lagrange_multipliers)

# Extract Lagrange multipliers solution
lagrange_multipliers_solution = result.x

print("Optimal values:")
print("P1:", p_solution[0])
print("P2:", p_solution[1])
print("Maximized Revenue:", maximized_profit)
print("Lagrange Multipliers:", lagrange_multipliers_solution)

"""# Part b)"""

# Objective function
def objective_function(p1, p2):
    return p1 * (intercept_values[0,0] + slope[0,0] * p1) + p2 * (intercept_values[0,1] + slope[0,1] * p2)

# Projected Gradient Descent with Gurobi
def projected_gradient_descent_with_gurobi(learning_rate, threshold):
    # Initialize Gurobi model
    model = gb.Model("projected_gradient_descent")

    # Decision variables
    p1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p1")
    p2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p2")

    # Objective
    obj = p1 * (intercept_values[0,0] + slope[0,0] * p1) + p2 * (intercept_values[0,1] + slope[0,1] * p2)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Optimization loop
    prev_obj = float('inf')
    i = 0

    while True:
        # Price Constraint
        model.addConstr(p1 <= p2, f"Price Constraint {i}")

        model.addConstr(intercept_values[0,0] + slope[0,0]*p1 >= 0, "Demand Definition Product Line 1 Basic")
        model.addConstr(intercept_values[0,1] + slope[0,1]*p2 >= 0, "Demand Definition Product Line 1 Advance")

        # Optimize model
        model.optimize()

        # Get current solution
        current_p1 = p1.X
        current_p2 = p2.X

        # Compute objective function
        current_obj = objective_function(current_p1, current_p2)

        # Check convergence
        if (current_obj - prev_obj) < threshold:
            break

        prev_obj = current_obj

        # Compute gradient
        df_dx = intercept_values[0,0] + 2 * slope[0,0] * current_p1
        df_dy = intercept_values[0,1] + 2 * slope[0,1] * current_p2

        # Update parameters
        current_p1 -= learning_rate * df_dx
        current_p2 -= learning_rate * df_dy

        # Set new starting point
        p1.Start = current_p1
        p2.Start = current_p2

        # Increment iteration counter
        i += 1

    return current_p1, current_p2

# Hyperparameters
learning_rate = 0.001
threshold = 1e-6

# Run projected gradient descent with Gurobi
final_x, final_y = projected_gradient_descent_with_gurobi(learning_rate, threshold)
print(f"Final solution: x = {final_x}, y = {final_y}, Objective = {round(objective_function(final_x, final_y),10)}")

"""# Part c)"""

# Create the optimization model
part_c_model = gb.Model("Question 1c): TechEssentials Certain Product Line")

p = part_c_model.addVars(3,3, lb=0, vtype=GRB.CONTINUOUS, name="Price")

#Objective Function
part_c_model.setObjective(gb.quicksum(p[i,n]*(intercept_values[i,n] + slope[i,n]*p[i,n]) for i in range(3) for n in range(3)), GRB.MAXIMIZE)

for i in range(3):
    for n in range(3):
        part_c_model.addConstr((intercept_values[i,n] + slope[i,n]*p[i,n]) >= 0, "Demand Lower Bound")

# Price Constraint
for n in range(2):
    part_c_model.addConstr(p[0, n] <= p[0, n + 1], f"Price Constraint {i}")
    part_c_model.addConstr(p[1, n] <= p[1, n + 1], f"Price Constraint {i}")
    part_c_model.addConstr(p[2, n] <= p[2, n + 1], f"Price Constraint {i}")

for i in range(3):
    for n in range(3):
        part_c_model.addConstr((intercept_values[i,n] + slope[i,n]*p[i,n]) <= product_capacity[i,n], "Max Demand")

# Solve our model
part_c_model.optimize()

# Value of the objective function
print("Revenue: ", round(part_c_model.objVal,2))

# Print the decision variables
print(part_c_model.printAttr('X'))

"""# Part d)"""

# Create the optimization model
part_d_model = gb.Model("Question 1d): TechEssentials Certain Product Line")

p = part_d_model.addVars(3,3, lb=0, vtype=GRB.CONTINUOUS, name="Price")

#Objective Function
part_d_model.setObjective(gb.quicksum(p[i,n]*(intercept_values[i][n] + slope[i][n]*p[i,n]) for i in range(3) for n in range(3)), GRB.MAXIMIZE)

for i in range(3):
    for n in range(3):
        part_d_model.addConstr((intercept_values[i][n] + slope[i][n]*p[i,n]) >= 0, "Demand Lower Bound")

# Price Constraint
for n in range(2):
    part_d_model.addConstr(p[0, n] <= p[0, n + 1], f"Price Constraint {i}")
    part_d_model.addConstr(p[1, n] <= p[1, n + 1], f"Price Constraint {i}")
    part_d_model.addConstr(p[2, n] <= p[2, n + 1], f"Price Constraint {i}")

for n in range(2):
    part_d_model.addConstr(p[n, 0] <= p[n + 1, 0], f"Price Constraint {i}")
    part_d_model.addConstr(p[n, 1] <= p[n + 1, 1], f"Price Constraint {i}")
    part_d_model.addConstr(p[n, 2] <= p[n + 1, 2], f"Price Constraint {i}")

#part_d_model.addConstr(p[0, 1] <= p[1, 0], f"Price Constraint Additional 1")
#part_d_model.addConstr(p[0, 2] <= p[1, 0], f"Price Constraint Additional 2")
#part_d_model.addConstr(p[1, 1] <= p[2, 0], f"Price Constraint Additional 3")
#part_d_model.addConstr(p[1, 2] <= p[2, 0], f"Price Constraint Additional 4")

for i in range(3):
    for n in range(3):
        part_d_model.addConstr((intercept_values[i,n] + slope[i,n]*p[i,n]) <= product_capacity[i,n], "Max Demand")

# Solve our model
part_d_model.optimize()

# Value of the objective function
print("Revenue: ", round(part_d_model.objVal,2))

# Print the decision variables
print(part_d_model.printAttr('X'))

# Extract values from the tupledict
values = [v for v in part_d_model.getAttr('x', p).values()]

# Reshape the values
price_values = np.array(values).reshape(3, -1)

price_values

demand = [[intercept_values[i][n] + slope[i][n] * price_values[i][n] for n in range(3)] for i in range(3)]
demand