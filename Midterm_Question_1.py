## - QUESTION 1


##SOLVING THE LINEAR OPTIMISATION PROBLEM 

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load data from CSV files
food_categories = pd.read_csv("/Users/pratiksha/Downloads/food_categories.csv")
food_preferences = pd.read_csv("/Users/pratiksha/Downloads/food_preferences.csv")
nutrient_content = pd.read_csv("/Users/pratiksha/Downloads/nutrient_content.csv", index_col=0)
nutrient_requirements = pd.read_csv("/Users/pratiksha/Downloads/nutrient_requirements.csv", index_col=0)

# Create a new model
model = gp.Model("Diet Optimization")

# Decision variables
foods = range(len(food_categories))
weeks = range(1)  # Assuming optimization for one week only

x = model.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model.setObjective(sum(x[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Dietary preferences constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Halal_grams"])

# Variety constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= 0.03 * sum(food_preferences.iloc[0][["Veggie_grams", "Vegan_grams", "Kosher_grams", "Halal_grams", "All_grams"]]))

# Solve the model
model.optimize()

# Print the optimal diet plan
for i in foods:
    for j in weeks:
        if x[i, j].x > 0:
            print(f"Food item {food_categories.iloc[i]['Food_Item']} for week {j}: {x[i, j].x} grams")

# Print the total cost
print(f"Total cost: {model.objVal}")


##(B) 

# After adding all constraints to the model
total_constraints = model.NumConstrs
print(f"Total number of constraints (excluding non-negativity): {total_constraints}")

## (E)

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

#Load the dataset
food_categories = pd.read_csv("/Users/pratiksha/Downloads/food_categories.csv")
food_preferences = pd.read_csv("/Users/pratiksha/Downloads/food_preferences.csv")
nutrient_content = pd.read_csv("/Users/pratiksha/Downloads/nutrient_content.csv", index_col=0)
nutrient_requirements = pd.read_csv("/Users/pratiksha/Downloads/nutrient_requirements.csv", index_col=0)

# Create a new model
model = gp.Model("Diet Optimization")

# Decision variables
foods = range(len(food_categories))
weeks = range(1)  # Assuming optimization for one week only

x = model.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model.setObjective(sum(x[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Dietary preferences constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Halal_grams"])

# Variety constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= 0.03 * sum(food_preferences.iloc[0][["Veggie_grams", "Vegan_grams", "Kosher_grams", "Halal_grams", "All_grams"]]))

# Solve the model
model.optimize()

# Print the optimal food production cost
print(f"Optimal food production cost: ${model.objVal:.2f}")


## (F)

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


#Load the dataset
food_categories = pd.read_csv("/Users/pratiksha/Downloads/food_categories.csv")
food_preferences = pd.read_csv("/Users/pratiksha/Downloads/food_preferences.csv")
nutrient_content = pd.read_csv("/Users/pratiksha/Downloads/nutrient_content.csv", index_col=0)
nutrient_requirements = pd.read_csv("/Users/pratiksha/Downloads/nutrient_requirements.csv", index_col=0)

# Create a new model
model = gp.Model("Diet Optimization")

# Decision variables
foods = range(len(food_categories))
weeks = range(1)  # Assuming optimization for one week only

x = model.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model.setObjective(sum(x[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Dietary preferences constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= food_preferences.iloc[0]["Halal_grams"])

# Variety constraints
for i in foods:
    model.addConstr(gp.quicksum(x[i, j] for j in weeks) <= 0.03 * sum(food_preferences.iloc[0][["Veggie_grams", "Vegan_grams", "Kosher_grams", "Halal_grams", "All_grams"]]))

# Solve the model
model.optimize()

# Print the optimal food production cost
print(f"Optimal food production cost: ${model.objVal:.2f}")

# Extract solution
solution = {food_categories.iloc[i]["Food_Item"]: x[i, 0].x for i in foods}

# Calculate total grams from Halal and Kosher foods
halal_grams = sum([solution.get("Food_15", 0), solution.get("Food_42", 0), solution.get("Food_48", 0), solution.get("Food_50", 0), solution.get("Food_62", 0)])
kosher_grams = sum([solution.get("Food_16", 0), solution.get("Food_44", 0), solution.get("Food_47", 0), solution.get("Food_49", 0), solution.get("Food_55", 0), solution.get("Food_58", 0), solution.get("Food_61", 0), solution.get("Food_70", 0), solution.get("Food_80", 0)])


# Calculate proportions
total_grams = sum(solution.values())
halal_proportion = halal_grams / total_grams
kosher_proportion = kosher_grams / total_grams

# Print proportions
print(f"Proportion of grams from Halal foods: {halal_proportion:.4f}")
print(f"Proportion of grams from Kosher foods: {kosher_proportion:.4f}")


## (G)

# Create a new model without the Variety constraint(s)
model_without_variety = gp.Model("Diet Optimization Without Variety Constraint")

# Decision variables
x_without_variety = model_without_variety.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model_without_variety.setObjective(sum(x_without_variety[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model_without_variety.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_without_variety[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model_without_variety.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_without_variety[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Dietary preferences constraints
for i in foods:
    model_without_variety.addConstr(gp.quicksum(x_without_variety[i, j] for j in weeks) <= food_preferences.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model_without_variety.addConstr(gp.quicksum(x_without_variety[i, j] for j in weeks) <= food_preferences.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model_without_variety.addConstr(gp.quicksum(x_without_variety[i, j] for j in weeks) <= food_preferences.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model_without_variety.addConstr(gp.quicksum(x_without_variety[i, j] for j in weeks) <= food_preferences.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model_without_variety.addConstr(gp.quicksum(x_without_variety[i, j] for j in weeks) <= food_preferences.iloc[0]["Halal_grams"])

# Solve the model without the Variety constraint(s)
model_without_variety.optimize()

# Extract solution
solution_without_variety = {food_categories.iloc[i]["Food_Item"]: x_without_variety[i, 0].x for i in foods}

# Count the number of food items produced
num_food_items_original = sum(1 for val in solution.values() if val > 0)
num_food_items_without_variety = sum(1 for val in solution_without_variety.values() if val > 0)
num_food_items_difference = num_food_items_original - num_food_items_without_variety

# Calculate the production cost for both models
production_cost_original = model.objVal
production_cost_without_variety = model_without_variety.objVal
production_cost_difference = production_cost_original - production_cost_without_variety

# Print results
print(f"Number of food items produced without the Variety constraint(s): {num_food_items_without_variety}")
print(f"Number of fewer food items produced: {num_food_items_difference}")
print(f"Production cost without the Variety constraint(s): ${production_cost_without_variety:.2f}")
print(f"Difference in production cost: ${production_cost_difference:.2f}")


## (I)

# Update the right-hand-side values of the dietary preference constraints
food_preferences_corrected = food_preferences.copy()
food_preferences_corrected.iloc[0] += 10000

# Create a new model with the corrected dietary preference constraints
model_corrected = gp.Model("Diet Optimization Corrected")

# Decision variables
x_corrected = model_corrected.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model_corrected.setObjective(sum(x_corrected[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model_corrected.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_corrected[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model_corrected.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_corrected[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Corrected dietary preferences constraints
for i in foods:
    model_corrected.addConstr(gp.quicksum(x_corrected[i, j] for j in weeks) <= food_preferences_corrected.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model_corrected.addConstr(gp.quicksum(x_corrected[i, j] for j in weeks) <= food_preferences_corrected.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model_corrected.addConstr(gp.quicksum(x_corrected[i, j] for j in weeks) <= food_preferences_corrected.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model_corrected.addConstr(gp.quicksum(x_corrected[i, j] for j in weeks) <= food_preferences_corrected.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model_corrected.addConstr(gp.quicksum(x_corrected[i, j] for j in weeks) <= food_preferences_corrected.iloc[0]["Halal_grams"])

# Solve the corrected model
model_corrected.optimize()

# Compare the objective function values
original_objective = model.objVal
corrected_objective = model_corrected.objVal
difference = corrected_objective - original_objective

# Print the results
print(f"Original Objective Function Value: {original_objective}")
print(f"Corrected Objective Function Value: {corrected_objective}")
print(f"Difference in Objective Function Value: {difference}")


##(J)

# Create a new model for sensitivity analysis
model_sensitivity = gp.Model("Diet Optimization Sensitivity")

# Decision variables
x_sensitivity = model_sensitivity.addVars(foods, weeks, vtype=GRB.CONTINUOUS, name="food")

# Objective function
model_sensitivity.setObjective(sum(x_sensitivity[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)

# Nutritional balance constraints
for j in weeks:
    for k in nutrient_requirements.index:
        model_sensitivity.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_sensitivity[i, j] for i in foods) >= nutrient_requirements.loc[k, "Min_Requirement"])
        model_sensitivity.addConstr(gp.quicksum(nutrient_content.loc[food_categories.iloc[i]["Food_Item"], k] * x_sensitivity[i, j] for i in foods) <= nutrient_requirements.loc[k, "Max_Requirement"])

# Dietary preferences constraints
for i in foods:
    model_sensitivity.addConstr(gp.quicksum(x_sensitivity[i, j] for j in weeks) <= food_preferences.iloc[0]["All_grams"])
    if food_categories.iloc[i]["Is_Vegetarian"]:
        model_sensitivity.addConstr(gp.quicksum(x_sensitivity[i, j] for j in weeks) <= food_preferences.iloc[0]["Veggie_grams"])
    if food_categories.iloc[i]["Is_Vegan"]:
        model_sensitivity.addConstr(gp.quicksum(x_sensitivity[i, j] for j in weeks) <= food_preferences.iloc[0]["Vegan_grams"])
    if food_categories.iloc[i]["Is_Kosher"]:
        model_sensitivity.addConstr(gp.quicksum(x_sensitivity[i, j] for j in weeks) <= food_preferences.iloc[0]["Kosher_grams"])
    if food_categories.iloc[i]["Is_Halal"]:
        model_sensitivity.addConstr(gp.quicksum(x_sensitivity[i, j] for j in weeks) <= food_preferences.iloc[0]["Halal_grams"])

# Add constraint to include the first food item
model_sensitivity.addConstr(gp.quicksum(x_sensitivity[0, j] for j in weeks) >= 1)

# Solve the sensitivity analysis model for different costs of the first food item
cost_reduction = 0
while True:
    # Set new cost for the first food item
    food_categories.at[0, "Cost_per_gram"] -= 0.001
    # Update the objective function with the new cost
    model_sensitivity.setObjective(sum(x_sensitivity[i, j] * food_categories.iloc[i]["Cost_per_gram"] for i in foods for j in weeks), GRB.MINIMIZE)
    # Solve the model
    model_sensitivity.optimize()
    # Check if the first food item is included in the optimal solution
    if x_sensitivity[0, 0].x > 0.5:
        break
    cost_reduction += 0.001

# Print the results
print(f"Amount the first food item needs to be less costly: {cost_reduction}")


### END OF QUESTION 1