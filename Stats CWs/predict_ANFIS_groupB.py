import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Data cleaning and preprocessing functions
def clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df
def remove_outliers_iqr(df, feature_names):
    for feature in feature_names:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df
# Load and clean the data
cleaned_data = clean_data(r'C:/Users/johnsmith/Desktop/combined_features.csv')
features_for_pca = ['other_moving_ac', 'runway hour', 'distance_else', 'NDepDep', 'distance','gate (block) hour', 'angle']
cleaned_data_no_outliers = remove_outliers_iqr(cleaned_data, features_for_pca)
# Feature Scaling and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cleaned_data_no_outliers[features_for_pca])
pca = PCA(n_components=0.8)  # Keep 80% of variance
X_pca = pca.fit_transform(X_scaled)
y = cleaned_data_no_outliers['taxi_time']
n_mfs = 25
n_rules = 25
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
# K-Means clustering for initial fuzzy set centers
cluster_model = KMeans(n_clusters=n_mfs)
cluster_model.fit(X_train)
centers = cluster_model.cluster_centers_
# ANFIS Model Definition
class FuzzificationLayer(nn.Module):
    def __init__(self, n_input, n_mfs, initial_centers):
        super().__init__()
        self.mf_centers = nn.Parameter(torch.tensor(initial_centers, dtype=torch.float32))
        self.mf_sigmas = nn.Parameter(torch.ones(n_mfs, n_input))

    def forward(self, x):
        out = torch.exp(-torch.pow(x.unsqueeze(1) - self.mf_centers, 2) / (2 * torch.pow(self.mf_sigmas, 2)))
        return out
class RuleLayer(nn.Module):
    def __init__(self, n_rules):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_rules))
        self.n_rules = n_rules

    def forward(self, x):
        batch_size = x.size(0)
        firing_strengths = torch.prod(x, dim=2)
        sum_strengths = torch.sum(firing_strengths, dim=1, keepdim=True)
        normalized_strengths = firing_strengths / sum_strengths
        return normalized_strengths.view(batch_size, self.n_rules)
# Number of PCA components
n_input = X_train_t.shape[1]
# ANFIS Model instantiation
class ANFIS(nn.Module):
    def __init__(self, n_input, n_rules, n_mfs, initial_centers):
        super().__init__()
        self.fuzz_layer = FuzzificationLayer(n_input, n_mfs, initial_centers)
        self.rule_layer = RuleLayer(n_rules)
        self.fc1 = nn.Linear(n_rules, 1)  # Output layer

    def forward(self, x):
        out = self.fuzz_layer(x)
        out = self.rule_layer(out)
        out = self.fc1(out)
        return out
model = ANFIS(n_input=X_train_t.shape[1], n_rules=n_rules, n_mfs=n_mfs, initial_centers=centers)
optimizer = optim.Adam(model.parameters(), lr=0.005) 
criterion = nn.MSELoss()
# Training loop
training_mae_list = []
validation_mae_list = []
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_t)
    train_loss = criterion(y_pred_train.squeeze(), y_train_t)
    train_loss.backward()
    optimizer.step()
    train_mae = mean_absolute_error(y_train_t.numpy(), y_pred_train.detach().numpy())
    training_mae_list.append(train_mae)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_test_t)
        val_loss = criterion(y_pred_val.squeeze(), y_test_t)
        val_mae = mean_absolute_error(y_test_t.numpy(), y_pred_val.numpy())
        validation_mae_list.append(val_mae)
    y_pred_val_np = y_pred_val.squeeze().detach().numpy()
    y_test_np = y_test_t.detach().numpy()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training MAE: {train_mae}, Validation MAE: {val_mae}')
plt.figure(figsize=(10, 5))
plt.plot(training_mae_list, label='Training MAE')
plt.plot(validation_mae_list, label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.show()
