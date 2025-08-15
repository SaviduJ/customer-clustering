import pandas as pd
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("customers.csv")

# Select features
X = df[["AnnualIncome", "SpendingScore"]]

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)  # Train model

# Example new customer data to predict cluster for
new_customer = [[15000, 50]]  # [AnnualIncome, SpendingScore]

# Predict cluster
predicted_cluster = kmeans.predict(new_customer)

print(f"The new customer belongs to cluster: {predicted_cluster[0]}")
