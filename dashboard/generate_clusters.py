import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load your raw dataset
df = pd.read_csv("spotify_dataset.csv")

# 2. Pick numeric features for clustering
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Save with clusters to the same folder as your app
df.to_csv("spotify_with_clusters.csv", index=False)
print("✅ spotify_with_clusters.csv created!")
