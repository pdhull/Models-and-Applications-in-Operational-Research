from gurobipy import GRB
import gurobipy as gb
import pandas as pd
import numpy as np
# Create the optimization model
model = gb.Model("Question 1: Canola Oil")
# Read costs from CSV files
direct_shipping_costs_df = pd.read_csv(r"C:\Users\gabri\Downloads\Cost_Production_to_Refinement.csv")
shipping_to_transsipment_costs_df = pd.read_csv(r"C:\Users\gabri\Downloads\Cost_Production_to_Transshipment.csv")
transsipment_to_refinement_costs_df = pd.read_csv(r"C:\Users\gabri\Downloads\Cost_Transshipment_to_Refinement.csv")
# Extract the "Cost" column
direct_shipping_costs = direct_shipping_costs_df['Cost'].values.reshape(25, -1)
shipping_to_transsipment_costs = shipping_to_transsipment_costs_df['Cost'].values.reshape(15, -1)
transsipment_to_refinement_costs = transsipment_to_refinement_costs_df['Cost'].values.reshape(2, -1)
direct_shipping_costs
shipping_to_transsipment_costs
transsipment_to_refinement_costs
# Create the a single class of decision variables where
# From = {Ca,US,M,C,F} and To = {R1,R2,R3,R4,R5}.
x = model.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="Direct Shipping")
# From = {I,U,G} and To = {Italy,Greece}.
y = model.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="Ship to Transshipment")
# From = {Italy,Greece} and To = {R1,R2,R3,R4,R5}.
z = model.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Transshipment to Refinement")
# The objective function
direct_objective = gb.quicksum(direct_shipping_costs[i][j]*x[i,j] for i in range(25) for j in range(5))
trans_objective = gb.quicksum(shipping_to_transsipment_costs[i][j]*y[i,j] for i in range(15) for j in range(2))
trans_to_refinement_objective = gb.quicksum(transsipment_to_refinement_costs[i][j]*z[i,j] for i in range(2) for j in range(5))
model.setObjective(direct_objective + trans_objective + trans_to_refinement_objective, GRB.MINIMIZE)
# Read capacity and demand from CSV files
direct_shipping_supply_capacity_df = pd.read_csv(r"C:\Users\gabri\Downloads\Capacity_for_Direct_Production_Facilities.csv")
transshipment_supply_capacity_df = pd.read_csv(r"C:\Users\gabri\Downloads\Capacity_for_Transship_Distribution_Centers.csv")
shipping_to_transshipment_supply_capacity_df = pd.read_csv(r"C:\Users\gabri\Downloads\Capacity_for_Transship_Production_Facilities.csv")
refinement_demand_df = pd.read_csv(r"C:\Users\gabri\Downloads\Refinement_Demand.csv")
# Extract the "Capacity" and "Demand" column
direct_shipping_supply_capacity = direct_shipping_supply_capacity_df['Capacity'].values.reshape(25, -1)
transshipment_supply_capacity = transshipment_supply_capacity_df['Capacity'].values.reshape(2, -1)
shipping_to_transshipment_supply_capacity = shipping_to_transshipment_supply_capacity_df['Capacity'].values.reshape(15, -1)
refinement_demand = refinement_demand_df['Demand'].values.reshape(5, -1)
direct_shipping_supply_capacity
transshipment_supply_capacity
shipping_to_transshipment_supply_capacity
refinement_demand
# Add the supply constraints from source nodes for direct shipping
for i in range(len(direct_shipping_supply_capacity)):
    model.addConstr(gb.quicksum(x[i, j] for j in range(5)) <= direct_shipping_supply_capacity[i], name=f"Direct Supply Constraint {i + 1}")
# Add the supply constraints from source nodes for transshipment shipping
for i in range(len(shipping_to_transshipment_supply_capacity)):
    model.addConstr(gb.quicksum(y[i, j] for j in range(2)) <= shipping_to_transshipment_supply_capacity[i], name=f"Transshipment Supply Constraint {i + 1}")
    # Add the supply constraints from transshipment nodes
model.addConstr(gb.quicksum(y[i,0] for i in range(15)) <= transshipment_supply_capacity[0], name="Transship Capacity 1")
model.addConstr(gb.quicksum(y[i,1] for i in range(15)) <= transshipment_supply_capacity[1], name="Transship Capacity 2")
# Add the flow balance constrainits
model.addConstr(gb.quicksum(y[i,0] for i in range(15)) == gb.quicksum(z[0,k] for k in range(5)), name="Flow Balance 1")
model.addConstr(gb.quicksum(y[i,1] for i in range(15)) == gb.quicksum(z[1,k] for k in range(5)), name="Flow Balance 2")
# Add the demand constraints
for k in range(len(refinement_demand)):  # Iterate over refinement nodes R1 to R5
    model.addConstr(gb.quicksum(x[i, k] for i in range(25)) + gb.quicksum(z[j, k] for j in range(2)) == refinement_demand[k], name=f"Refinement Demand Constraint {k + 1}")
    # Optimally solve the problem
model.optimize()
# The status of the model (Optimization Status Codes)
print("Model Status: ", model.status)
# Number of variables in the model
print("Number of Decision Variables: ", model.numVars)
# Value of the objective function
print("Total Transportation cost: ", round(model.objVal, 2))
# Print the decision variables
print(model.printAttr('X'))
# Get the optimal values of decision variables
to_transshipment_values = model.getAttr('x', y)
to_transshipment_values
values_array = np.array(list(to_transshipment_values.values()))
# Calculate the mean
mean_value = np.mean(values_array)
mean_value
# Get the optimal values of decision variables
transshipment_values = model.getAttr('x', z)
transshipment_values
# Calculate the total amount of canola oil transshipped
total_transshipped = sum(transshipment_values[i, j] for i in range(2) for j in range(5))
total_transshipped
# Get the optimal values of decision variables
direct_shipping_values = model.getAttr('x', x)
direct_shipping_values
# Calculate the total amount of canola oil directly shipped
direct_shipping = sum(direct_shipping_values[i, j] for i in range(25) for j in range(5))
direct_shipping
# Calculate the total amount of canola oil directly shipped for North America
direct_shipping_for_f = sum(direct_shipping_values[i, j] for i in range(15) for j in range(5))
direct_shipping_for_f
# Calculate the proportion
total_canola_oil = direct_shipping + total_transshipped
proportion_transshipped = total_transshipped / total_canola_oil
proportion_transshipped
# Create the optimization model
model_modified = gb.Model("Question 1: Canola Oil Objective Function Change")
# Create the a single class of decision variables where
# From = {Ca,US,M,C,F} and To = {R1,R2,R3,R4,R5}.
x = model_modified.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="Direct Shipping")
# From = {I,U,G} and To = {Italy,Greece}.
y = model_modified.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="Ship to Transshipment")
# From = {Italy,Greece} and To = {R1,R2,R3,R4,R5}.
z = model_modified.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Transshipment to Refinement")
direct_shipping_costs
# The objective function
direct_objective = gb.quicksum(direct_shipping_costs[i][j]*x[i,j] for i in range(25) for j in range(5))
trans_objective = gb.quicksum(shipping_to_transsipment_costs[i][j]*y[i,j] for i in range(15) for j in range(2))
trans_to_refinement_objective = gb.quicksum(transsipment_to_refinement_costs[i][j]*z[i,j] for i in range(2) for j in range(5))

# Adding the modified objective to the model
model_modified.setObjective((0.95*direct_objective) + (1.15*trans_objective) + trans_to_refinement_objective, GRB.MINIMIZE)
# Add the supply constraints from source nodes for direct shipping
for i in range(len(direct_shipping_supply_capacity)):
    model_modified.addConstr(gb.quicksum(x[i, j] for j in range(5)) <= direct_shipping_supply_capacity[i], name=f"Direct Supply Constraint {i + 1}")
# Add the supply constraints from source nodes for transshipment shipping
for i in range(len(shipping_to_transshipment_supply_capacity)):
    model_modified.addConstr(gb.quicksum(y[i, j] for j in range(2)) <= shipping_to_transshipment_supply_capacity[i], name=f"Transshipment Supply Constraint {i + 1}")
# Add the supply constraints from transshipment nodes
model_modified.addConstr(gb.quicksum(y[i,0] for i in range(15)) <= transshipment_supply_capacity[0], name="Transship Capacity 1")
model_modified.addConstr(gb.quicksum(y[i,1] for i in range(15)) <= transshipment_supply_capacity[1], name="Transship Capacity 2")
# Add the flow balance constrainits
model_modified.addConstr(gb.quicksum(y[i,0] for i in range(15)) == gb.quicksum(z[0,k] for k in range(5)), name="Flow Balance 1")
model_modified.addConstr(gb.quicksum(y[i,1] for i in range(15)) == gb.quicksum(z[1,k] for k in range(5)), name="Flow Balance 2")
# Add the demand constraints
for k in range(len(refinement_demand)):  # Iterate over refinement nodes R1 to R5
    model_modified.addConstr(gb.quicksum(x[i, k] for i in range(25)) + gb.quicksum(z[j, k] for j in range(2)) == refinement_demand[k], name=f"Refinement Demand Constraint {k + 1}")
# Optimally solve the problem
model_modified.optimize()
# Value of the objective function
print("Total Transportation cost: ", round(model_modified.objVal, 2))
# Get the optimal values of decision variables
transshipment_values_modified = model_modified.getAttr('x', y)
transshipment_values_modified
# Calculate the total amount of canola oil transshipped
total_transshipped_modified = sum(transshipment_values_modified[i, j] for i in range(15) for j in range(2))
total_transshipped_modified
# Create the optimization model
model_constraint = gb.Model("Question 1: Canola Oil Constraint Change")
# Create the a single class of decision variables where
# From = {Ca,US,M,C,F} and To = {R1,R2,R3,R4,R5}.
x = model_constraint.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="Direct Shipping")
# From = {I,U,G} and To = {Italy,Greece}.
y = model_constraint.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="Ship to Transshipment")
# From = {Italy,Greece} and To = {R1,R2,R3,R4,R5}.
z = model_constraint.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Transshipment to Refinement")
# The objective function
direct_objective = gb.quicksum(direct_shipping_costs[i][j]*x[i,j] for i in range(25) for j in range(5))
trans_objective = gb.quicksum(shipping_to_transsipment_costs[i][j]*y[i,j] for i in range(15) for j in range(2))
trans_to_refinement_objective = gb.quicksum(transsipment_to_refinement_costs[i][j]*z[i,j] for i in range(2) for j in range(5))
model_constraint.setObjective(direct_objective + trans_objective + trans_to_refinement_objective, GRB.MINIMIZE)
# Add the supply constraints from source nodes for direct shipping
for i in range(len(direct_shipping_supply_capacity)):
    model_constraint.addConstr(gb.quicksum(x[i, j] for j in range(5)) <= direct_shipping_supply_capacity[i], name=f"Direct Supply Constraint {i + 1}")

# Add the supply constraints from source nodes for transshipment shipping
for i in range(len(shipping_to_transshipment_supply_capacity)):
    model_constraint.addConstr(gb.quicksum(y[i, j] for j in range(2)) <= shipping_to_transshipment_supply_capacity[i], name=f"Transshipment Supply Constraint {i + 1}")

# Add the supply constraints from transshipment nodes
model_constraint.addConstr(gb.quicksum(y[i,0] for i in range(15)) <= transshipment_supply_capacity[0], name="Transship Capacity 1")
model_constraint.addConstr(gb.quicksum(y[i,1] for i in range(15)) <= transshipment_supply_capacity[1], name="Transship Capacity 2")
# Add the flow balance constrainits
model_constraint.addConstr(gb.quicksum(y[i,0] for i in range(15)) == gb.quicksum(z[0,k] for k in range(5)), name="Flow Balance 1")
model_constraint.addConstr(gb.quicksum(y[i,1] for i in range(15)) == gb.quicksum(z[1,k] for k in range(5)), name="Flow Balance 2")
# Add the demand constraints
for k in range(len(refinement_demand)):  # Iterate over refinement nodes R1 to R5
    model_constraint.addConstr(gb.quicksum(x[i, k] for i in range(25)) + gb.quicksum(z[j, k] for j in range(2)) == refinement_demand[k], name=f"Refinement Demand Constraint {k + 1}")
# Ratio constraint
model_constraint.addConstr((0.25*(gb.quicksum(x[i,j] for i in range(25) for j in range(5)) + gb.quicksum(z[i,j] for i in range(2) for j in range(5)))) >= gb.quicksum(y[i,j] for i in range(15) for j in range(2)), name="Ratio constraint")
# Optimally solve the problem
model_constraint.optimize()
# Value of the objective function
print("Total Transportation cost: ", round(model_constraint.objVal, 2))
# Get the optimal values of decision variables
transshipment_values = model_constraint.getAttr('x', z)
transshipment_values
# Calculate the total amount of canola oil transshipped
total_transshipped = sum(transshipment_values[i, j] for i in range(2) for j in range(5))
total_transshipped
# Get the optimal values of decision variables
direct_shipping_values = model_constraint.getAttr('x', x)
direct_shipping_values
# Calculate the total amount of canola oil directly shipped
direct_shipping = sum(direct_shipping_values[i, j] for i in range(25) for j in range(5))
direct_shipping
# Calculate the proportion
total_canola_oil = direct_shipping + total_transshipped
proportion_transshipped = total_transshipped / total_canola_oil
proportion_transshipped
# Create the optimization model
model_NA = gb.Model("Question 1: Canola Oil North America Change")
# Create the a single class of decision variables where
# From = {Ca,US,M,C,F} and To = {R1,R2,R3,R4,R5}.
x = model_NA.addVars(25, 5, lb=0, vtype=GRB.CONTINUOUS, name="Direct Shipping")
# From = {I,U,G} and To = {Italy,Greece}.
y = model_NA.addVars(15, 2, lb=0, vtype=GRB.CONTINUOUS, name="Ship to Transshipment")
# From = {Italy,Greece} and To = {R1,R2,R3,R4,R5}.
z = model_NA.addVars(2, 5, lb=0, vtype=GRB.CONTINUOUS, name="Transshipment to Refinement")
direct_shipping_costs
for i in range(15):
    for j in range(5):
        direct_shipping_costs[i, j] *= 0.82
direct_shipping_costs
# The objective function
direct_objective = gb.quicksum(direct_shipping_costs[i][j]*x[i,j] for i in range(25) for j in range(5))
trans_objective = gb.quicksum(shipping_to_transsipment_costs[i][j]*y[i,j] for i in range(15) for j in range(2))
trans_to_refinement_objective = gb.quicksum(transsipment_to_refinement_costs[i][j]*z[i,j] for i in range(2) for j in range(5))
model_NA.setObjective(direct_objective + trans_objective + trans_to_refinement_objective, GRB.MINIMIZE)
# Add the supply constraints from source nodes for direct shipping
for i in range(len(direct_shipping_supply_capacity)):
    model_NA.addConstr(gb.quicksum(x[i, j] for j in range(5)) <= direct_shipping_supply_capacity[i], name=f"Direct Supply Constraint {i + 1}")
# Add the supply constraints from source nodes for transshipment shipping
for i in range(len(shipping_to_transshipment_supply_capacity)):
    model_NA.addConstr(gb.quicksum(y[i, j] for j in range(2)) <= shipping_to_transshipment_supply_capacity[i], name=f"Transshipment Supply Constraint {i + 1}")
# Add the supply constraints from transshipment nodes
model_NA.addConstr(gb.quicksum(y[i,0] for i in range(15)) <= transshipment_supply_capacity[0], name="Transship Capacity 1")
model_NA.addConstr(gb.quicksum(y[i,1] for i in range(15)) <= transshipment_supply_capacity[1], name="Transship Capacity 2")
# Add the flow balance constrainits
model_NA.addConstr(gb.quicksum(y[i,0] for i in range(15)) == gb.quicksum(z[0,k] for k in range(5)), name="Flow Balance 1")
model_NA.addConstr(gb.quicksum(y[i,1] for i in range(15)) == gb.quicksum(z[1,k] for k in range(5)), name="Flow Balance 2")
# Add the demand constraints
for k in range(len(refinement_demand)):  # Iterate over refinement nodes R1 to R5
    model_NA.addConstr(gb.quicksum(x[i, k] for i in range(25)) + gb.quicksum(z[j, k] for j in range(2)) == refinement_demand[k], name=f"Refinement Demand Constraint {k + 1}")

# Optimally solve the problem
model_NA.optimize()
# Value of the objective function
print("Total Transportation cost: ", round(model_NA.objVal, 2))
# Get the optimal values of decision variables
direct_shipping_values = model_NA.getAttr('x', x)
direct_shipping_values
# Calculate the total amount of canola oil directly shipped
direct_shipping = sum(direct_shipping_values[i, j] for i in range(25) for j in range(5))
direct_shipping
# Calculate the total amount of canola oil directly shipped
direct_shipping_for_f = sum(direct_shipping_values[i, j] for i in range(15) for j in range(5))
direct_shipping_for_f
# Create the optimization model
question_2_model = gb.Model("Question 2: Sunnyshore Bay")
# Create the three classes of decision variables where each Python
# variable represents a different number of Gurobi decision variables
B = question_2_model.addVars(6, lb=0, vtype=GRB.CONTINUOUS, name="Borrow")
w = question_2_model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Wealth")
# The objective function
question_2_model.setObjective(w[3], GRB.MAXIMIZE)
# Add the balance constraints
question_2_model.addConstr(w[0] == 140000 + 180000 - 300000 + B[0] + B[1] + B[2], "May Balance Constraint")
question_2_model.addConstr(w[1] == w[0] + 260000 - 400000 + B[3] + B[4] - 1.0175*B[2], "June Balance Constraint")
question_2_model.addConstr(w[2] == w[1] + 420000 - 350000 + B[5] - 1.0225*B[1] - 1.0175*B[4], "July Balance Constraint")
question_2_model.addConstr(w[3] == w[2] + 580000 - 200000 - 1.0275*B[0] - 1.0225*B[3] - 1.0175*B[5], "August Balance Constraint")
# Add the cash flow constraints
May_Cash_Flow_Constraint = question_2_model.addConstr(w[0] >= 25000, "May Cash Flow Constraint")
June_Cash_Flow_Constraint = question_2_model.addConstr(w[1] >= 20000, "June Cash Flow Constraint")
July_Cash_Flow_Constraint = question_2_model.addConstr(w[2] >= 35000, "July Cash Flow Constraint")
August_Cash_Flow_Constraint = question_2_model.addConstr(w[3] >= 18000, "August Cash Flow Constraint")
# Add the borrowing constraints
question_2_model.addConstr((B[0] + B[1] + B[2]) <= 250000, "May Borrowing Constraint")
question_2_model.addConstr((B[3] + B[4]) <= 150000, "June Borrowing Constraint")
question_2_model.addConstr(B[5] <= 350000, "July Borrowing Constraint")
# Ratio constraint
question_2_model.addConstr((0.65*(w[0] + w[1])) <= w[2], name="Ratio constraint")
# Optimally solve the problem
question_2_model.optimize()
# The status of the model (Optimization Status Codes)
print("Model Status: ", question_2_model.status)
# Number of variables in the model
print("Number of Decision Variables: ", question_2_model.numVars)
# Value of the objective function
print("Total Amount of Money: ", round(question_2_model.objVal, 2))
# Print the decision variables
print(question_2_model.printAttr('X'))
# Get the optimal values of decision variables
borrowing_money = question_2_model.getAttr('x', B)

# Display the optimal values
print("Optimal Borrowing Values:")
for i in range(len(borrowing_money)):
    print(f"Borrow[{i}] = {borrowing_money[i]}")

# Calculate the total amount borrowed
total_borrowing_amount = borrowing_money[0] + borrowing_money[1] + borrowing_money[2] + borrowing_money[3] + borrowing_money[4] + borrowing_money[5]
print("\nTotal Borrowing Amount:", total_borrowing_amount)
total_repay_amount = 1.0275*borrowing_money[0] + 1.0225*borrowing_money[1] + 1.0175*borrowing_money[2] + 1.0225*borrowing_money[3] + 1.0175*borrowing_money[4] + 1.0175*borrowing_money[5]
print("\nTotal Repayment Amount:", total_repay_amount)
# Print sensitivity information
print("")
print(f"Sensitivity Information for June Cash Flow Constraint {June_Cash_Flow_Constraint.pi:.2f}:")
print("(LHS, RHS, Slack): ", (question_2_model.getRow(June_Cash_Flow_Constraint).getValue(), June_Cash_Flow_Constraint.RHS, June_Cash_Flow_Constraint.slack))
print("Shadow Price: ", June_Cash_Flow_Constraint.pi)
print("Range of Feasibility: ", (June_Cash_Flow_Constraint.SARHSUp, June_Cash_Flow_Constraint.SARHSLow))
# Check if the optimization was successful
if question_2_model.status == gb.GRB.OPTIMAL:
    # Print the sensitivity analysis for the amount sold
    print("Optimal Amount Sold:")
    print(f"{'1'} = {B[0].x, B[0].RC, B[0].SAObjUp, B[0].SAObjLow}")
    print(f"{'2'} = {B[1].x, B[1].RC, B[1].SAObjUp, B[1].SAObjLow}")
    print(f"{'3'} = {B[2].x, B[2].RC, B[2].SAObjUp, B[2].SAObjLow}")
    print(f"{'4'} = {B[3].x, B[3].RC, B[3].SAObjUp, B[3].SAObjLow}")
    print(f"{'5'} = {B[4].x, B[4].RC, B[4].SAObjUp, B[4].SAObjLow}")
    print(f"{'6'} = {B[5].x, B[5].RC, B[5].SAObjUp, B[5].SAObjLow}")
else:
    print("Optimization was not successful.")
# Change in Objective Function Value
change_in_ofv = June_Cash_Flow_Constraint.pi * (27500 - June_Cash_Flow_Constraint.RHS)

# Additional money needed to be borrowed
additional_borrowing = -change_in_ofv  # Note: Multiply by -1 to make it positive

print("Change in Objective Function Value:", change_in_ofv)
print("Additional Money Needed to be Borrowed:", additional_borrowing)
# Create the optimization model
question_2_model2 = gb.Model("Question 2: Sunnyshore Bay")

# Create the three classes of decision variables where each Python
# variable represents a different number of Gurobi decision variables
B = question_2_model2.addVars(6, lb=0, vtype=GRB.CONTINUOUS, name="Borrow")
w = question_2_model2.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Wealth")

# The objective function
question_2_model2.setObjective(w[3], GRB.MAXIMIZE)

# Add the balance constraints
question_2_model2.addConstr(w[0] == 140000 + 180000 - 300000 + B[0] + B[1] + B[2], "May Balance Constraint")
question_2_model2.addConstr(w[1] == w[0] + 260000 - 400000 + B[3] + B[4] - 1.0175*B[2], "June Balance Constraint")
question_2_model2.addConstr(w[2] == w[1] + 420000 - 350000 + B[5] - 1.0225*B[1] - 1.0175*B[4], "July Balance Constraint")
question_2_model2.addConstr(w[3] == w[2] + 580000 - 200000 - 1.0275*B[0] - 1.0225*B[3] - 1.0175*B[5], "August Balance Constraint")

# Add the cash flow constraints
May_Cash_Flow_Constraint = question_2_model2.addConstr(w[0] >= 25000, "May Cash Flow Constraint")
June_Cash_Flow_Constraint = question_2_model2.addConstr(w[1] >= 27500, "June Cash Flow Constraint")
July_Cash_Flow_Constraint = question_2_model2.addConstr(w[2] >= 35000, "July Cash Flow Constraint")
August_Cash_Flow_Constraint = question_2_model2.addConstr(w[3] >= 18000, "August Cash Flow Constraint")

# Add the borrowing constraints
question_2_model2.addConstr((B[0] + B[1] + B[2]) <= 250000, "May Borrowing Constraint")
question_2_model2.addConstr((B[3] + B[4]) <= 150000, "June Borrowing Constraint")
question_2_model2.addConstr(B[5] <= 350000, "July Borrowing Constraint")

# Ratio constraint
question_2_model2.addConstr((0.65*(w[0] + w[1])) <= w[2], name="Ratio constraint")

# Optimally solve the problem
question_2_model2.optimize()

# Value of the objective function
print("Total Amount of Money: ", round(question_2_model2.objVal, 2))
# Get the optimal values of decision variables
borrowing_money = question_2_model2.getAttr('x', B)

# Display the optimal values
print("Optimal Borrowing Values:")
for i in range(len(borrowing_money)):
    print(f"Borrow[{i}] = {borrowing_money[i]}")

# Calculate the total amount borrowed
total_borrowing_amount = borrowing_money[0] + borrowing_money[1] + borrowing_money[2] + borrowing_money[3] + borrowing_money[4] + borrowing_money[5]
print("\nTotal Borrowing Amount:", total_borrowing_amount)
total_repay_amount = 1.0275*borrowing_money[0] + 1.0225*borrowing_money[1] + 1.0175*borrowing_money[2] + 1.0225*borrowing_money[3] + 1.0175*borrowing_money[4] + 1.0175*borrowing_money[5]
print("\nTotal Repayment Amount:", total_repay_amount)
# Create the optimization model for the dual problem
dual_model = gb.Model("Dual of Question 2: Sunnyshore Bay")

# Create dual variables for each constraint in the primal model
y_may_balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Balance_Dual")
y_june_balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Balance_Dual")
y_july_balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Balance_Dual")
y_august_balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="August_Balance_Dual")
y_may_cash_flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Cash_Flow_Dual")
y_june_cash_flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Cash_Flow_Dual")
y_july_cash_flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Cash_Flow_Dual")
y_august_cash_flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="August_Cash_Flow_Dual")
y_may_borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Borrowing_Dual")
y_june_borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Borrowing_Dual")
y_july_borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Borrowing_Dual")
y_ratio_constraint = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Ratio_Constraint_Dual")

# Set the objective function for the dual problem (minimize)
dual_model.setObjective(25000*y_may_cash_flow + 20000*y_june_cash_flow + 35000*y_july_cash_flow + 18000*y_august_cash_flow
                        + 250000*y_may_borrowing + 150000*y_june_borrowing + 350000*y_july_borrowing
                        + (0.65 * (y_may_balance + y_june_balance)), GRB.MINIMIZE)

# Add dual constraints for each primal variable
dual_model.addConstr(y_may_balance >= 1, "May_Balance_Dual_Constraint")
dual_model.addConstr(y_june_balance >= 1.0175, "June_Balance_Dual_Constraint")
dual_model.addConstr(y_july_balance >= 1.0225, "July_Balance_Dual_Constraint")
dual_model.addConstr(y_august_balance >= 1.0275, "August_Balance_Dual_Constraint")

dual_model.addConstr(y_may_cash_flow >= 0, "May_Cash_Flow_Dual_Constraint")
dual_model.addConstr(y_june_cash_flow >= 0, "June_Cash_Flow_Dual_Constraint")
dual_model.addConstr(y_july_cash_flow >= 0, "July_Cash_Flow_Dual_Constraint")
dual_model.addConstr(y_august_cash_flow >= 0, "August_Cash_Flow_Dual_Constraint")

dual_model.addConstr(y_may_borrowing >= 0, "May_Borrowing_Dual_Constraint")
dual_model.addConstr(y_june_borrowing >= 0, "June_Borrowing_Dual_Constraint")
dual_model.addConstr(y_july_borrowing >= 0, "July_Borrowing_Dual_Constraint")

dual_model.addConstr(y_ratio_constraint >= 0.65, "Ratio_Constraint_Dual_Constraint")

# Optimally solve the dual problem
dual_model.optimize()

# The status of the model (Optimization Status Codes)
print("Dual Model Status: ", dual_model.status)

# Number of variables in the model
print("Number of Dual Variables: ", dual_model.numVars)

# Value of the objective function
print("Dual Objective Value: ", round(dual_model.objVal, 2))

# Print the dual variables
print(dual_model.printAttr('X'))
# Create the optimization model for the dual problem
dual_model = gb.Model("Dual of Question 2: Sunnyshore Bay")

# Create dual variables for each constraint in the primal model
y_May_Balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Balance_Dual")
y_June_Balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Balance_Dual")
y_July_Balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Balance_Dual")
y_August_Balance = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="August_Balance_Dual")
y_May_Cash_Flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Cash_Flow_Dual")
y_June_Cash_Flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Cash_Flow_Dual")
y_July_Cash_Flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Cash_Flow_Dual")
y_August_Cash_Flow = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="August_Cash_Flow_Dual")
y_May_Borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="May_Borrowing_Dual")
y_June_Borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="June_Borrowing_Dual")
y_July_Borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="July_Borrowing_Dual")
y_August_Borrowing = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="August_Borrowing_Dual")
y_Ratio_Constraint = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Ratio_Constraint_Dual")

# Set the correct objective function for the dual problem (minimize)
dual_model.setObjective(
    25000*y_May_Cash_Flow + 20000*y_June_Cash_Flow + 35000*y_July_Cash_Flow + 18000*y_August_Cash_Flow
    + 250000*y_May_Borrowing + 150000*y_June_Borrowing + 350000*y_July_Borrowing
    + 0.65*y_Ratio_Constraint, GRB.MINIMIZE
)

# Add dual constraints for each primal variable
dual_model.addConstr(-y_May_Balance + y_June_Balance + y_July_Balance + y_August_Balance - y_May_Cash_Flow - y_May_Borrowing == -1, "May_Dual_Constraint")
dual_model.addConstr(-1.0175*y_May_Balance + y_June_Balance - y_July_Balance - 1.0225*y_August_Balance + y_June_Cash_Flow + y_June_Borrowing == 1.0225, "June_Dual_Constraint")
dual_model.addConstr(-1.0175*y_May_Balance + 1.0175*y_June_Balance + y_July_Balance - 1.0175*y_August_Balance + y_July_Cash_Flow + y_July_Borrowing == 1.0175, "July_Dual_Constraint")
dual_model.addConstr(-1.0225*y_May_Balance + 1.0175*y_June_Balance + 1.0225*y_July_Balance - y_August_Balance + y_August_Cash_Flow + y_August_Borrowing == 1.0225, "August_Dual_Constraint")
dual_model.addConstr(-0.65*y_Ratio_Constraint == 0, "Ratio_Constraint_Dual")

# Optimally solve the dual problem
dual_model.optimize()

# The status of the model (Optimization Status Codes)
print("Dual Model Status: ", dual_model.status)

# Number of variables in the model
print("Number of Dual Variables: ", dual_model.numVars)

# Value of the objective function for the primal model
print("Total Amount of Money (Primal): ", round(question_2_model.objVal, 2))

# Value of the objective function for the dual model
print("Dual Objective Value: ", round(dual_model.objVal, 2))

# Print the dual variables
print(dual_model.printAttr('X'))





 