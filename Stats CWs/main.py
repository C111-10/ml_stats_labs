import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest
import scipy.stats as stats


def clean_data(file_path):
    # read CSV file
    df = pd.read_csv(file_path)

    # check repeat line and delete it
    print(f"original line number: {len(df)}")
    df = df.drop_duplicates()
    print(f"the line number after deleting repeat value: {len(df)}")

    # Find the number of missing values per column
    missing_values = df.isnull().sum()
    print("Missing value statistics:")
    print(missing_values)

    #Fill in missing values # For numerical features, use mean to fill in missing values # For category features, use mode to fill in missing values # Here, adjustments need to be made based on your data characteristics
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    return df


# Use data cleaning function and design file path
cleaned_data = clean_data(r'C:\Users\xyf12\Desktop\EMS702\data collection\features.csv')

# Select features for PCA
features_for_pca = \
    ['other_moving_ac', 'runway hour',
     'gate (block) hour', 'rwy_num',
     'QArrArr', 'NDepDep', 'NArrArr']


for feature in features_for_pca:  # Use features_for_pca instead of X.columns if X is not defined yet
    p_value = kstest(cleaned_data[feature], 'norm')[1]  # Use cleaned_data as X is not defined yet

    if p_value < 0.05:
        print(feature + " is not normal") # H1
        # The line below is incorrect as 'new_feature' and 'X_scaled_cleaned' are not defined in the provided code
        # x[new_feature]=X_scaled_cleaned
    else:
        print(feature + " is normal") # H0


# Define the outlier detection and removal function, IQR method
def remove_outliers_iqr(df, feature_names):
    for feature in feature_names:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

cleaned_data_no_outliers = remove_outliers_iqr(cleaned_data, features_for_pca)



scaler = StandardScaler()
X = cleaned_data_no_outliers[features_for_pca]
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

# Create a 2x2 subplot grid for the feature loadings bar charts
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the figure size as needed
axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array for easy iteration

for i in range(n_components_to_plot):
    component_loadings = pca_full.components_[i]
    axes[i].barh(features_for_pca, component_loadings, align='center')
    axes[i].set_title(f'PCA Component {i + 1} Loadings')
    axes[i].invert_yaxis()  # To place the largest bars at the top
    for index, value in enumerate(component_loadings):
        axes[i].text(value, index, f"{value:.2f}")  # Optional: add text with the loading values

# Adjust layout for better fit and display the plot
plt.tight_layout()
plt.show()

# Plot scatter plots for each pair of principal components
for i in range(0, n_components_80, 2):
    if i + 1 < n_components_80:  # Check if there is a pair to plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca_full[:, i], X_pca_full[:, i + 1], alpha=0.7)
        plt.xlabel(f'Principal Component {i + 1}')
        plt.ylabel(f'Principal Component {i + 2}')
        plt.title(f'Scatter Plot of Principal Components {i + 1} and {i + 2}')
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
