import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Music Genre Grouping",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    h1 { color: #1DB954; font-weight: bold; }
    h2 { color: #191414; margin-top: 1.5rem; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("# 🎵 Music Genre Grouping System")
st.markdown("**Intelligent clustering of music tracks using K-Means unsupervised learning**")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Configuration")
    uploaded_file = st.file_uploader(
        "Upload Spotify CSV file",
        type=["csv"],
        help="Upload a CSV file with music features (e.g., Spotify data export)"
    )

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File loaded successfully!")
        
        # Data validation
        if df.empty:
            st.error("❌ Uploaded file is empty. Please provide a valid CSV file.")
            st.stop()
        
        # Display dataset info
        with st.expander("📊 Dataset Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.write("**Column Names:**")
            st.write(df.columns.tolist())
            st.dataframe(df.head(10), use_container_width=True)
        
        # Feature selection
        st.markdown("### 🎚️ Feature Selection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset.")
            st.stop()
        
        features = st.multiselect(
            "Choose audio features for clustering",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            help="Select at least 2 numeric features"
        )
        
        if len(features) < 2:
            st.warning("⚠️ Please select at least 2 features for clustering.")
            st.stop()
        
        # Data preprocessing
        X = df[features].copy()
        missing_count = X.isnull().sum().sum()
        
        if missing_count > 0:
            X = X.fillna(X.mean())
            st.info(f"ℹ️ Filled {missing_count} missing values with column means")
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K selection with elbow method
        st.markdown("### 🔢 Cluster Configuration")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            k = st.slider("Number of Clusters (K)", min_value=2, max_value=min(10, len(df)), value=3)
        
        with col2:
            show_elbow = st.checkbox("Show Elbow Method", value=True)
        
        if show_elbow:
            with st.spinner("Computing elbow curve..."):
                inertias = []
                k_range = range(1, min(11, len(df)))
                for k_val in k_range:
                    kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    kmeans_temp.fit(X_scaled)
                    inertias.append(kmeans_temp.inertia_)
                
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(k_range), y=inertias,
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='#1DB954', width=3),
                    marker=dict(size=8)
                ))
                fig_elbow.add_vline(x=k, line_dash="dash", line_color="red",
                                   annotation_text=f"Selected K={k}")
                fig_elbow.update_layout(
                    title="Elbow Method - Inertia vs Number of Clusters",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="Inertia",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Run clustering
        if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
            with st.spinner("Performing K-Means clustering..."):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                df["Cluster"] = labels
                
                st.success("✅ Clustering completed successfully!")
            
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visualization", "📉 Analysis", "💾 Export"])
            
            with tab1:
                st.markdown("### Clustered Data Preview")
                st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    height=400
                )
                
                # Cluster distribution
                st.markdown("### Cluster Distribution")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig_dist = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Samples'},
                    title='Distribution of Samples Across Clusters',
                    color=cluster_counts.index,
                    color_continuous_scale='Viridis'
                )
                fig_dist.update_layout(
                    showlegend=False,
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with tab2:
                st.markdown("### Interactive 2D Visualization")
                
                # Create interactive scatter plot
                fig_scatter = px.scatter(
                    df,
                    x=X_scaled[:, 0],
                    y=X_scaled[:, 1],
                    color='Cluster',
                    title=f'K-Means Clustering (K={k})',
                    labels={'x': features[0], 'y': features[1]},
                    color_continuous_scale='Viridis',
                    hover_data={col: df[col].iloc[i] for i, col in enumerate(df.columns) if col != 'Cluster'}
                )
                
                # Add centroids
                centroids_scaled = scaler.transform(kmeans.cluster_centers_)
                fig_scatter.add_scatter(
                    x=centroids_scaled[:, 0],
                    y=centroids_scaled[:, 1],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='red', line=dict(color='white', width=2)),
                    name='Centroids'
                )
                
                fig_scatter.update_layout(
                    height=600,
                    template='plotly_white',
                    hovermode='closest'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 3D visualization if we have 3+ features
                if len(features) >= 3:
                    st.markdown("### 3D Visualization")
                    fig_3d = px.scatter_3d(
                        df,
                        x=X_scaled[:, 0],
                        y=X_scaled[:, 1],
                        z=X_scaled[:, 2],
                        color='Cluster',
                        title=f'3D K-Means Clustering',
                        color_continuous_scale='Viridis'
                    )
                    fig_3d.update_layout(height=600, template='plotly_white')
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab3:
                st.markdown("### Cluster Statistics")
                
                for cluster_id in sorted(df['Cluster'].unique()):
                    cluster_data = df[df['Cluster'] == cluster_id][features]
                    with st.expander(f"📍 Cluster {cluster_id} Statistics", expanded=(cluster_id == 0)):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Samples", len(cluster_data))
                        with col2:
                            st.metric("Percentage", f"{len(cluster_data)/len(df)*100:.1f}%")
                        
                        st.write("**Feature Statistics:**")
                        stats_df = pd.DataFrame({
                            'Feature': features,
                            'Mean': [cluster_data[f].mean() for f in features],
                            'Std': [cluster_data[f].std() for f in features],
                            'Min': [cluster_data[f].min() for f in features],
                            'Max': [cluster_data[f].max() for f in features],
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                st.markdown("### Model Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Inertia (Sum of Squared Distances)", f"{kmeans.inertia_:.2f}")
                with col2:
                    st.metric("Silhouette Score", "Available with scikit-learn")
            
            with tab4:
                st.markdown("### Export Results")
                
                # Download full results
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Clustered Data (CSV)",
                    data=csv_data,
                    file_name="music_clusters.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Download cluster summary
                summary_data = []
                for cluster_id in sorted(df['Cluster'].unique()):
                    summary_data.append({
                        'Cluster': cluster_id,
                        'Sample_Count': len(df[df['Cluster'] == cluster_id]),
                        'Percentage': f"{len(df[df['Cluster'] == cluster_id])/len(df)*100:.2f}%"
                    })
                summary_df = pd.DataFrame(summary_data)
                
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cluster Summary (CSV)",
                    data=csv_summary,
                    file_name="cluster_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your CSV file format and ensure it contains numeric data.")

else:
    st.info("👉 Upload a Spotify CSV file to get started with music genre clustering!")
        