import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


def clean_data(fp):

    df = pd.read_csv(fp)


    # check repeat line and delete it
    print(f"original line number: {len(df)}")
    df = df.drop_duplicates()
    print(f"the line number after deleting repeat value: {len(df)}")

    # Find the number of missing values per column
    missing_values = df.isnull().sum()
    print("Missing value statistics:")
    print(missing_values)

    # Fill in missing values # For numerical features, use mean to fill in missing values # For category features, use mode to fill in missing values # Here, adjustments need to be made based on your data characteristics
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    return df


# Use data cleaning function and design file path
fp = (r'/Users/christencampbell/Desktop/QMUL/EMS702/CW/FIles23_12/combined_features.csv')
cleaned_data = clean_data(fp)

# rwy_sum->angle   #QArrArr->distance_else   #NArrArr->distance
# Select features for PCA
features_for_pca = \
    ['other_moving_ac', 'distance', 'runway hour','distance_else', 'NDepDep',
     'gate (block) hour', 'angle',]

# 正态分布检验
for feature in features_for_pca:  # Use features_for_pca instead of X.columns if X is not defined yet
    p_value = kstest(cleaned_data[feature], 'norm')[1]  # Use cleaned_data as X is not defined yet

    if p_value < 0.05:
        print(feature + " is not normal")  # H1
        # The line below is incorrect as 'new_feature' and 'X_scaled_cleaned' are not defined in the provided code

    else:
        print(feature + " is normal")  # H0


# why?

# Define the outlier detection and removal function, IQR method
def remove_outliers_iqr(df, feature_names):
    for feature in feature_names:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

def remove_outliers_zscore(df, feature_names, threshold=3.0):

    for feature in feature_names:
        z_scores = np.abs(stats.zscore(df[feature]))
        df = df[(z_scores < threshold)]
    return df


cleaned_data_no_outliers = remove_outliers_iqr(cleaned_data, features_for_pca)
cleaned_data_no_outliers_zscore = remove_outliers_zscore(cleaned_data_no_outliers, features_for_pca)

scaler = StandardScaler()
X = cleaned_data_no_outliers_zscore[features_for_pca]
X_scaled = scaler.fit_transform(X)
# Create PCA objects and perform PCA fitting on the data
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Get explained variance ratio
explained_variance_ratio_full = pca_full.explained_variance_ratio_

# Compute cumulative explained variance
cumulative_explained_variance_full = np.cumsum(explained_variance_ratio_full)

# Find the number of principal components that explain more than 80% variance
n_components_80 = np.argmax(cumulative_explained_variance_full >= 0.8) + 1

# Print the number of components to explain at least 80% variance
print(f"Number of components to explain 80% variance: {n_components_80}")

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance_full) + 1)
         , cumulative_explained_variance_full, marker='o')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% explained variance')
plt.axvline(x=n_components_80, color='r', linestyle='--', label=f'{n_components_80} components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# After computing the PCA
n_components_to_plot = min(n_components_80, 4)  # Limit to 4 components for a 2x2 subplot grid

# Plot feature loadings for the first two principal components
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust the figure size as needed

# Flatten the grid into a 1D array for easy iteration
axes = axes.flatten()

# Plot only the first two components (PCA1 and PCA2)
for i in range(3):
    component_loadings = pca_full.components_[i]
    axes[i].barh(features_for_pca, component_loadings, align='center')
    axes[i].set_title(f'PCA Component {i + 1} Loadings')
    axes[i].invert_yaxis()  # To place the largest bars at the top
    for index, value in enumerate(component_loadings):
        axes[i].text(value, index, f"{value:.2f}")  # Add text with the loading values

# Plot scatter plots for each pair of the first three principal components
# Create a scatter plot for components 1 and 2
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Principal Components 1 and 2')
plt.grid(True)
plt.show()

# Create a scatter plot for components 1 and 3
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_full[:, 0], X_pca_full[:, 2], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.title('Scatter Plot of Principal Components 1 and 3')
plt.grid(True)
plt.show()

# Create a scatter plot for components 2 and 3
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_full[:, 1], X_pca_full[:, 2], alpha=0.7)
plt.xlabel('Principal Component 2')
plt.ylabel('Principal Component 3')
plt.title('Scatter Plot of Principal Components 2 and 3')
plt.grid(True)
plt.show()

# Print PCA results
print("Explained variance ratio:", explained_variance_ratio_full)
for i, ratio in enumerate(explained_variance_ratio_full[:n_components_80]):
    print(f"Principal Component {i + 1}: {ratio}")

# After fitting PCA, print the feature loadings for each principal component
for i in range(n_components_80):
    print(f"Principal Component {i + 1}:")
    loadings = pca_full.components_[i]
    for j, feature in enumerate(features_for_pca):
        print(f"{feature}: {loadings[j]:.4f}")
    print()  # Blank line for readability between components

cleaned_data = clean_data(fp)


pca_data = pd.DataFrame({
    'PCA1' : X_pca_full[:, 0],
    'PCA2' : X_pca_full[:, 1],
    'PCA3' : X_pca_full[:, 2]
})

X = pca_data[['PCA1', 'PCA2', 'PCA3']]
df = cleaned_data['taxi_time']
rowz = len(X)
ss = X[:rowz]
k = len(ss)
y = df.head(k)

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Evaluate the performance of the model
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

# Print the evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error (MAE): {mae}")
# Optionally, you can also print the model coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Calculate MAE and MSE for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

# Calculate MAE and MSE for validation set
mae_val = mean_absolute_error(y_val, y_val_pred)

# Plot the training and validation curves
epochs_train = np.arange(1, len(y_train) + 1)
epochs_val = np.arange(1, len(y_val) + 1)

# Ensure that the lengths of training and validation are the same
min_len = min(len(epochs_train), len(epochs_val))
epochs_train = epochs_train[:min_len]
epochs_val = epochs_val[:min_len]

# Create a smooth curve for MAE
smooth_train_mae = np.convolve(np.abs(y_train[:min_len] - y_train_pred[:min_len]), np.ones(10)/10, mode='valid')
smooth_val_mae = np.convolve(np.abs(y_val[:min_len] - y_val_pred[:min_len]), np.ones(10)/10, mode='valid')
#smooth_train_mae = cleaned_data_no_outliers_zscore(smoot_train_mae)
#smooth_val_mae = cleaned_data_no_outliers_zscore(smoot_val_mae)
plt.figure(figsize=(12, 6))

# Plot MAE
plt.plot(epochs_train, [mae_train] * len(epochs_train), label='Training MAE (Constant)', linestyle='--', color='blue')
plt.plot(epochs_val, [mae_val] * len(epochs_val), label='Validation MAE (Constant)', linestyle='--', color='orange')
plt.plot(epochs_train[:len(smooth_train_mae)], smooth_train_mae, label='MAE (Training)', color='blue')
plt.plot(epochs_val[:len(smooth_val_mae)], smooth_val_mae, label='MAE (Validation)', color='orange')
plt.title('MAE Training and Validation Curves')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()