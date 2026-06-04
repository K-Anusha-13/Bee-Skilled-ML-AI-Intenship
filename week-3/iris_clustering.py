import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ==========================
# Step 1: Load Dataset
# ==========================

df = pd.read_csv("Iris.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# Step 2: Prepare Features
# ==========================

# Remove Id column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Store true labels
true_labels = df['Species']

# Features only
X = df.drop('Species', axis=1)

# ==========================
# Step 3: Apply K-Means
# ==========================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add cluster column
df['Cluster'] = clusters

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# ==========================
# Step 4: PCA (2D Reduction)
# ==========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# ==========================
# Step 5: Visualize Clusters
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

plt.title("K-Means Clusters on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()

# ==========================
# Step 6: Compare with Actual Labels
# ==========================

species_mapping = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2,
    'setosa':0,
    'versicolor':1,
    'virginica':2
}

actual_numeric = true_labels.map(species_mapping)

print("\nConfusion Matrix")
print(confusion_matrix(actual_numeric, clusters))

# ==========================
# Step 7: Sample Results
# ==========================

comparison = pd.DataFrame({
    'Actual Species': true_labels,
    'Predicted Cluster': clusters
})

print("\nActual Species vs Predicted Cluster")
print(comparison.head(15))