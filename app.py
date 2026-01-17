import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Music Genre Grouping System")
st.write("Unsupervised Learning using K-Means Clustering")

uploaded_file = st.file_uploader("Upload Spotify CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.write("### Dataset Preview")
    st.dataframe(df.head())


    st.write("### Select Audio Features")
    features = st.multiselect(
        "Choose features for clustering",
        df.columns,
        default=["tempo", "loudness", "energy"]
    )

    if len(features) >= 2:
        X = df[features]

       
        X = X.fillna(X.mean())

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

       
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        if st.button("Run Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            df["Cluster"] = labels

            st.success("Clustering completed!")

            st.write("### Clustered Data")
            st.dataframe(df.head())

          
            st.write("### Cluster Visualization")
            fig, ax = plt.subplots()
            ax.scatter(
                X_scaled[:, 0],
                X_scaled[:, 1],
                c=labels
            )
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)

    else:
        st.warning("Please select at least 2 features.")
        