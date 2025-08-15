# streamlit_app.py (replace or insert this section in your existing app)
import streamlit as st
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ---- Config ----
st.set_page_config(page_title="Spotify Clusters (Interactive PCA)", layout="wide")
DATA_PATH = r"C:\Users\surya\SPOTIFY-GENRE-CLUSTERING\src\dashboard\spotify_clusters.csv"
PCA_CACHE = r"C:\Users\surya\SPOTIFY-GENRE-CLUSTERING\src\dashboard\spotify_pca_cache.csv"

# ---- Load data ----
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)
st.sidebar.markdown("**Dataset loaded**")
st.sidebar.write(f"Tracks: {len(df):,}")

# ---- auto-detect column names ----
track_col = None
artist_col = None
for c in ['track_name','track','name','title']:
    if c in df.columns:
        track_col = c
        break
for c in ['artist_name','artists','artist']:
    if c in df.columns:
        artist_col = c
        break

# ---- select numeric features used for PCA (fallback list) ----
possible_features = ['danceability','energy','loudness','speechiness',
                     'acousticness','instrumentalness','liveness','valence','tempo','duration_ms','popularity']
features = [f for f in possible_features if f in df.columns]
if not features:
    st.error("No numeric audio features found in dataset. Check column names.")
    st.stop()

# ---- Filters in sidebar ----
st.sidebar.header("Filters")
clusters_available = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Clusters", clusters_available, default=clusters_available)

tempo_min = float(df['tempo'].min()) if 'tempo' in df.columns else 0.0
tempo_max = float(df['tempo'].max()) if 'tempo' in df.columns else 250.0
tempo_range = st.sidebar.slider("Tempo (BPM)", tempo_min, tempo_max, (tempo_min, tempo_max))

energy_min = float(df['energy'].min()) if 'energy' in df.columns else 0.0
energy_max = float(df['energy'].max()) if 'energy' in df.columns else 1.0
energy_range = st.sidebar.slider("Energy", energy_min, energy_max, (energy_min, energy_max))

dance_min = float(df['danceability'].min()) if 'danceability' in df.columns else 0.0
dance_max = float(df['danceability'].max()) if 'danceability' in df.columns else 1.0
dance_range = st.sidebar.slider("Danceability", dance_min, dance_max, (dance_min, dance_max))

# ---- apply filters ----
filtered = df[
    df['Cluster'].isin(selected_clusters) &
    (df['tempo'].between(tempo_range[0], tempo_range[1])) &
    (df['energy'].between(energy_range[0], energy_range[1])) &
    (df['danceability'].between(dance_range[0], dance_range[1]))
].reset_index(drop=True)

st.markdown(f"### Showing {len(filtered):,} tracks")

# ---- PCA (cache results to speed up) ----
def compute_and_cache_pca(df_in, features, cache_path):
    # If cache exists and matches length, use it
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path)
            if len(cache_df) == len(df_in):
                return cache_df[['PC1','PC2']].values
        except Exception:
            pass

    X = df_in[features].fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    cache_df = pd.DataFrame({'PC1': pcs[:,0], 'PC2': pcs[:,1]})
    # store index to ensure later alignment if needed
    cache_df.to_csv(cache_path, index=False)
    return pcs

pcs = compute_and_cache_pca(filtered, features, PCA_CACHE)
filtered['PC1'] = pcs[:,0]
filtered['PC2'] = pcs[:,1]

# ---- Plotly interactive scatter ----
st.subheader("Cluster PCA Scatter (interactive)")
hover_cols = [track_col] if track_col else []
if artist_col:
    hover_cols.append(artist_col)
# Add some numeric features to hover
hover_cols += [c for c in ['tempo','energy','danceability','valence','popularity'] if c in filtered.columns]

fig = px.scatter(
    filtered, x='PC1', y='PC2',
    color='Cluster',
    hover_data=hover_cols,
    title="PCA projection of tracks (click legend to toggle clusters)",
    width=1200, height=700,
    color_continuous_scale=px.colors.qualitative.Pastel
)

fig.update_traces(marker=dict(size=6, opacity=0.7), selector=dict(mode='markers'))
st.plotly_chart(fig, use_container_width=True)

# ---- show sample table for selected point(s) ----
st.subheader("Sample Tracks in Filtered Selection")
sample_n = st.slider("How many sample tracks to show", 3, 30, 8)
st.dataframe(filtered[[track_col, artist_col, 'Cluster'] + features].head(sample_n))

