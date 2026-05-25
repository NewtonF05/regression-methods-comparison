#!/usr/bin/env python
# coding: utf-8

"""
Comparison of Regression Methods on a Falling-Sphere Dataset
============================================================

Author: Newton Fernihough

A comparison of several linear-regression techniques applied to an
experimental dataset of falling spheres of different materials, radii
and masses. The target variable is the time of fall for a given drop
height; the underlying physical relationship includes nonlinear
quadratic-drag behaviour, against which the various regressors are
benchmarked.

Methods compared:
    - Ordinary least-squares linear regression
    - Ridge regression
    - Lasso regression
    - Huber regression (robust to outliers)
    - Stochastic gradient descent regression

Each method is evaluated by mean squared error on a held-out test set,
with learning curves and a theoretical drag-model overlay used to
contextualise the fit quality.

Data:
    exercise3data.csv -- expected in a ``data/`` folder alongside the script.
"""

# **Imports**

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from tabulate import tabulate
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler


# **Functions**

# In[7]:


# Function to check if a value is a float
def is_float(value):
    """
    Check if a value can be converted to a float.

    Parameters:
    value (any): The value to check.

    Returns:
    bool: True if the value can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Function to format regression coefficients into a readable table.
def format_coefficients_table(coef, intercept, feature_names):
    """
    Format regression coefficients into a readable table.

    Parameters:
    coef (array-like): The coefficients from the regression model.
    intercept (float or array-like): The intercept from the regression model.
    feature_names (list): List of feature names corresponding to the coefficients.

    Returns:
    str: A formatted table of feature names and their coefficients.
    """
    # Create a dictionary to hold feature names and their coefficients
    coef_dict = {name: value for name, value in zip(feature_names, coef)}
    # Add the intercept to the dictionary, ensuring it's a scalar value
    coef_dict["Intercept"] = intercept[0] if isinstance(intercept, (list, np.ndarray)) else intercept
    # Format the dictionary into a table using tabulate
    return tabulate(coef_dict.items(), headers=["Feature", "Coefficient"], tablefmt="pretty")

# Function to calculate the Mean Squared Error (MSE) between true and predicted values.
def calculate_mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The Mean Squared Error.
    """
    return mean_squared_error(y_true, y_pred)

# Function to evaluate and print model results
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """
    Evaluate a machine learning model and print the results.

    Parameters:
    model: The machine learning model to evaluate.
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Test data features.
    y_test (array-like): Test data labels.
    name (str): Name of the model for identification in the output.

    Returns:
    float: The Mean Squared Error of the model on the test data.
    """
    model.fit(X_train, y_train) # Train the model
    predictions = model.predict(X_test) # Get predictions
    mse = mean_squared_error(y_test, predictions) # Compute MSE

    print(f"{name} MSE: {mse}")
    # Debugging: Print the MSE
    return mse # Ensure a valid numerical return

# overlay a theoretical physics-based model on the data.
def add_theoretical_model(data, color, **kwargs):
    """
    Overlay a theoretical physics-based model on the data.

    Parameters:
    data (DataFrame): The data containing 'radius' and 'mass' columns.
    color (str): The color to use for the theoretical model plot.
    **kwargs: Additional keyword arguments for plotting.

    Returns:
    None
    """
    ax = plt.gca() # Get current axis

    # Constants
    g_acc = 9.81 # Gravity (m/s²)
    Cd = 0.47 # Drag coefficient for a sphere
    rho0 = 1.2 # Air density (kg/m³)

    # Extract the radius and mass for the current facet
    radius = data['radius'].iloc[0] # Same for the facet
    mass = data['mass'].iloc[0] # Same for the facet

    # Compute the cross-sectional area
    A = np.pi * radius**2

    # Compute k
    k = (Cd * rho0 * A) / 2

    # Generate height values
    h_values = np.linspace(0, 1500, 500) # Forces range to 0-1500 for all facets

    # Theoretical time calculation
    exp_values = np.exp((h_values * k) / mass)
    t_values = np.sqrt(mass / (k * g_acc)) * np.arccosh(np.maximum(exp_values, 1.0001))

    # Plot the theoretical model
    ax.plot(h_values, t_values, color='red', linestyle='dashed', linewidth=2)


# **Part 1**

# In[9]:


# Load the dataset
columns = ["material", "density", "radius", "mass", "temperature", "pressure", "height", "time"] 
raw_data = pd.read_csv("data/exercise3data.csv", on_bad_lines='skip', names=columns)

# Drop rows with missing values
cleaned_data = raw_data.dropna()

# List of valid materials to keep
valid_materials = ["iron", "titanium", "magnesium", "silicon_carbide", "zinc_oxide", "silica", "polycarbonate"]

# Filter the DataFrame to keep only rows with valid materials
cleaned_data = cleaned_data[cleaned_data['material'].isin(valid_materials)]

# Remove rows where columns expected to contain strings have non-string data
string_columns = ["material"] # List of columns expected to be strings for col in string_columns: df_clean = df_clean[df_clean[col].apply(lambda x: isinstance(x, str))]

# Ensure numeric columns contain only numeric data
numeric_columns = ["density", "radius", "mass", "temperature", "pressure", "height", "time"] 
for col in numeric_columns: 
    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')

# Drop rows with NaN in numeric columns after conversion
cleaned_data = cleaned_data.dropna(subset=numeric_columns)

# Print dataset statistics
original_rows = raw_data.shape[0] 
cleaned_rows = cleaned_data.shape[0] 
rows_removed = original_rows - cleaned_rows

print(f"Original number of rows: {original_rows}") 
print(f"Number of rows after cleaning: {cleaned_rows}") 
print(f"Number of rows removed: {rows_removed}")

# Print min and max values for each column
for col in cleaned_data.columns: 
    min_val = cleaned_data[col].min() 
    max_val = cleaned_data[col].max() 
    print(f"Column: {col}, Min: {min_val}, Max: {max_val}")

# Scatter plots of Time vs Height for different materials and radii
g = sns.FacetGrid(cleaned_data, col="material", row="radius", height=4, sharex=True, sharey=True)
g.map(sns.scatterplot, "height", "time", alpha=0.5)  # Scatter plot without regression line  

# Add titles and labels  
g.set_axis_labels("Height", "Time")  
g.set_titles("Material: {col_name} | Radius: {row_name}")  
plt.subplots_adjust(top=0.9)  
g.fig.suptitle("Scatter Plot of Time vs Height for Different Materials and Radii", fontsize=16)  
plt.show()


# **Part 2**

# In[11]:


# Correlation matrix
corr_matrix = cleaned_data.iloc[:, 1:].corr()  # Skips the first column (material)

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.title('Correlation Matrix')
plt.show()


# **Part 3**

# In[13]:


# Define features (X) and target (y)
X = cleaned_data[["density", "radius", "mass", "temperature", "pressure", "height"]]
y = cleaned_data["time"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and evaluate Linear Regression
linear_model = LinearRegression()
mse_linear = evaluate_model(linear_model, X_train, y_train, X_test, y_test, "Linear Regression")
print("Linear Regression Coefficients:")
print(format_coefficients_table(linear_model.coef_, linear_model.intercept_, X.columns))
print("-" * 50)


# Fit and evaluate Ridge Regression (L2 penalty)
ridge_model = Ridge(alpha=1.0)
mse_ridge = evaluate_model(ridge_model, X_train, y_train, X_test, y_test, "Ridge Regression")
print("Ridge Regression Coefficients:")
print(format_coefficients_table(ridge_model.coef_, ridge_model.intercept_, X.columns))
print("-" * 50)

# Fit and evaluate Lasso Regression (L1 penalty)
lasso_model = Lasso(alpha=1.0)
mse_lasso = evaluate_model(lasso_model, X_train, y_train, X_test, y_test, "Lasso Regression")
print("Lasso Regression Coefficients:")
print(format_coefficients_table(lasso_model.coef_, lasso_model.intercept_, X.columns))
print("-" * 50)

# Fit and evaluate Huber Regression (robust to outliers)
huber_model = HuberRegressor(epsilon=1.35)
mse_huber = evaluate_model(huber_model, X_train, y_train, X_test, y_test, "Huber Regression")
print("Huber Regression Coefficients:")
print(format_coefficients_table(huber_model.coef_, huber_model.intercept_, X.columns))
print("-" * 50)

# Debugging: Check MSE values
# print("MSE Values:", mse_ridge, mse_lasso, mse_huber)

# Ensure no None values before plotting
if None in [mse_ridge, mse_lasso, mse_huber]:
    raise ValueError("One of the MSE values is None. Check evaluate_model function!")

# Scale the features to ensure SGDRegressor performs well
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit and evaluate SGD Regression with different loss functions
sgd_model_l2 = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.1, max_iter=1000, tol=1e-3, random_state=42)
mse_sgd_l2 = evaluate_model(sgd_model_l2, X_train_scaled, y_train, X_test_scaled, y_test, "SGD Regression (L2 Penalty)")
print("SGD Regression (L2 Penalty) Coefficients:")
print(format_coefficients_table(sgd_model_l2.coef_, sgd_model_l2.intercept_, X.columns))
print("-" * 50)

sgd_model_l1 = SGDRegressor(loss='squared_error', penalty='l1', alpha=0.1, max_iter=1000, tol=1e-3, random_state=42)
mse_sgd_l1 = evaluate_model(sgd_model_l1, X_train_scaled, y_train, X_test_scaled, y_test, "SGD Regression (L1 Penalty)")
print("SGD Regression (L1 Penalty) Coefficients:")
print(format_coefficients_table(sgd_model_l1.coef_, sgd_model_l1.intercept_, X.columns))
print("-" * 50)

sgd_model_huber = SGDRegressor(loss='huber', penalty='l2', alpha=0.1, max_iter=1000, tol=1e-3, random_state=42)
mse_sgd_huber = evaluate_model(sgd_model_huber, X_train_scaled, y_train, X_test_scaled, y_test, "SGD Regression (Huber Loss)")
print("SGD Regression (Huber Loss) Coefficients:")
print(format_coefficients_table(sgd_model_huber.coef_, sgd_model_huber.intercept_, X.columns))
print("-" * 50)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 grid of subplots
axes = axes.ravel()  # Flatten the axes array for easy iteration

# Residual Plots
predicted_values = linear_model.predict(X_test).ravel()
residuals = (y_test - predicted_values).to_numpy()
axes[0].scatter(predicted_values, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residual Plot")

# Actual vs Expected Values
axes[1].scatter(y_test, predicted_values, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")
axes[1].set_title("Actual vs. Predicted Values")

# Learning Curves
train_sizes, train_scores, test_scores = learning_curve(
    sgd_model_l2, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
axes[2].plot(train_sizes, train_scores_mean, label="Training Error")
axes[2].plot(train_sizes, test_scores_mean, label="Validation Error")
axes[2].set_xlabel("Training Set Size")
axes[2].set_ylabel("MSE")
axes[2].set_title("Learning Curve")
axes[2].legend()

# Coefficient Magnitudes
models = [sgd_model_l2, sgd_model_l1, sgd_model_huber]
model_names = ["SGD (L2)", "SGD (L1)", "SGD (Huber)"]
coefficients = [model.coef_ for model in models]
for i, coef in enumerate(coefficients):
    axes[3].bar(np.arange(len(coef)) + i * 0.2, coef, width=0.2, label=model_names[i])
axes[3].set_xticks(np.arange(len(X.columns)) + 0.2)
axes[3].set_xticklabels(X.columns, rotation=45)
axes[3].set_xlabel("Features")
axes[3].set_ylabel("Coefficient Magnitude")
axes[3].set_title("Comparison of Coefficients Across Models")
axes[3].legend()

# Distribution of Errors
sns.histplot(residuals, kde=True, ax=axes[4])
axes[4].set_xlabel("Residuals")
axes[4].set_ylabel("Frequency")
axes[4].set_title("Distribution of Residuals")

# Feature Importance (for Regularized Models)
feature_importance = np.abs(sgd_model_l1.coef_)
axes[5].barh(X.columns, feature_importance)
axes[5].set_xlabel("Coefficient Magnitude")
axes[5].set_ylabel("Features")
axes[5].set_title("Feature Importance (L1 Regularization)")

# Comparison of MSE Across Models
mse_values = [mse_ridge or 0, mse_lasso or 0, mse_huber or 0]  # Replace None with 0
model_names = ["Ridge (L2)", "Lasso (L1)", "Huber"]
axes[6].bar(model_names, mse_values, color=['blue', 'orange', 'green'])
axes[6].set_xlabel("Models")
axes[6].set_ylabel("MSE")
axes[6].set_title("Comparison of MSE Across Models")

# Convergence Plot for SGD
sgd_model_l2 = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.1, max_iter=1000, tol=1e-3, random_state=42, warm_start=True)
losses = []
for i in range(1000):
    sgd_model_l2.partial_fit(X_train, y_train)
    losses.append(mean_squared_error(y_train, sgd_model_l2.predict(X_train)))
axes[7].plot(losses)
axes[7].set_xlabel("Iterations")
axes[7].set_ylabel("Training Loss (MSE)")
axes[7].set_title("Convergence Plot for SGD (L2 Penalty)")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# **Part 4**

# In[15]:


# Split data while maintaining sequence (90% train, 10% test)
train_size = int(0.9 * len(cleaned_data))
df_train = cleaned_data.iloc[:train_size]
df_test = cleaned_data.iloc[train_size:]

# Define features and target
X_train, y_train = df_train.drop(columns=["time", "material"]), df_train["time"]
X_test, y_test = df_test.drop(columns=["time", "material"]), df_test["time"]

# Train linear model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
residuals = y_test - y_pred

print(f"Mean Squared Error: {mse:.5f}")

# Analyze model for h < 500m and h > 1000m
df_low = cleaned_data[cleaned_data['height'] < 500]
df_high = cleaned_data[cleaned_data['height'] > 1000]

mse_low = mean_squared_error(df_low['time'], model.predict(df_low.drop(columns=['time', 'material'])))
mse_high = mean_squared_error(df_high['time'], model.predict(df_high.drop(columns=['time', 'material'])))

print(f"MSE for h < 500m: {mse_low:.5f}")
print(f"MSE for h > 1000m: {mse_high:.5f}")

all_residuals = []
unique_materials = cleaned_data['material'].unique()
unique_radii = cleaned_data['radius'].unique()

# Create a plot for each combination of material and radius
for material in unique_materials:
    for radius in unique_radii:
        # Filter data for the current combination of material and radius
        df_combination = cleaned_data[(cleaned_data['material'] == material) & (cleaned_data['radius'] == radius)]
        
        if df_combination.shape[0] > 1:  # Ensure there are enough data points to make predictions
            # Define the features (X) and target (y) for the current combination
            X_combination = df_combination[["density", "radius", "mass", "temperature", "pressure", "height"]]
            y_combination = df_combination["time"]
            
            # Split the data into training and testing sets
            X_train_combination, X_test_combination, y_train_combination, y_test_combination = train_test_split(
                X_combination, y_combination, test_size=0.2, random_state=42
            )

            # Fit the linear regression model
            model_combination = LinearRegression()
            model_combination.fit(X_train_combination, y_train_combination)

            # Make predictions
            y_pred_combination = model_combination.predict(X_test_combination)

            # Calculate residuals
            residuals_combination = y_test_combination - y_pred_combination

            # Append residuals to the list
            all_residuals.extend(residuals_combination)

            # Plot residuals for the current combination
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred_combination, residuals_combination, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title(f"Residual Plot for Material: {material}, Radius: {radius}")
            plt.show()

# Plot a histogram of all residuals
plt.figure(figsize=(10, 6))
sns.histplot(all_residuals, kde=True, bins=30, color='blue', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency Density")  # Renaming the y-axis label
plt.title("Histogram of Residuals for All Materials and Radii")
plt.show()


# In[16]:


# Create a FacetGrid for Time vs. Height by Material and Radius
g = sns.FacetGrid(cleaned_data, col="material", row="radius", height=4, sharex=True, sharey=True)

# Map the scatter plot to the grid
g.map(sns.scatterplot, "height", "time", alpha=0.6, color='blue')

# Apply the theoretical model function to each facet
g.map_dataframe(add_theoretical_model)

# Set labels and titles
g.set_axis_labels("Height", "Time")
g.set_titles(col_template="{col_name}", row_template="Radius: {row_name}")

# Adjust layout
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Time vs Height for Different Material and Radius Combinations", fontsize=16)

plt.show()


# In[17]:


# Unique materials and radii
unique_materials = cleaned_data['material'].unique()
unique_radii = cleaned_data['radius'].unique()

# Prepare subplots
num_plots = len(unique_materials) * len(unique_radii)
cols = 5  # Set number of columns in grid
rows = int(np.ceil(num_plots / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
axes = axes.flatten()  # Flatten in case of single row

all_residuals = []
plot_index = 0

for material in unique_materials:
    for radius in unique_radii:
        df_combination = cleaned_data[(cleaned_data['material'] == material) & (cleaned_data['radius'] == radius)]
        
        if df_combination.shape[0] > 1:  # Ensure enough data points
            X_combination = df_combination[["density", "radius", "mass", "temperature", "pressure", "height"]]
            y_combination = df_combination["time"]
            
            # Train-test split
            X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
                X_combination, y_combination, test_size=0.2, random_state=42
            )
            
            # Train model
            model_comb = LinearRegression()
            model_comb.fit(X_train_comb, y_train_comb)
            y_pred_comb = model_comb.predict(X_test_comb)
            
            # Residuals
            residuals_comb = y_test_comb - y_pred_comb
            all_residuals.extend(residuals_comb)
            
            # Plot in grid
            ax = axes[plot_index]
            ax.scatter(y_pred_comb, residuals_comb, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title(f"Material: {material}, Radius: {radius}")
            plot_index += 1

# Hide any unused subplots
for i in range(plot_index, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Histogram of all residuals
plt.figure(figsize=(10, 6))
sns.histplot(all_residuals, kde=True, bins=30, color='blue', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency Density")
plt.title("Histogram of Residuals for All Materials and Radii")
plt.show()


# Having the graphs in a grid like this makes them easier to view as a whole but makes the data they are representing harder to interpret. Note that this is just one of the 2 sets of residuals plots.
