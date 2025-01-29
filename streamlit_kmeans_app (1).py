import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit app
st.title('K-Means Clustering App')

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(data.head())
    
    # Interactive slider for selecting number of clusters
    num_clusters = st.slider('Select the number of clusters:', min_value=2, max_value=10, value=3)
    
    # Preprocess the data
    st.write("Preprocessing the data...")
    data_scaled = StandardScaler().fit_transform(data.select_dtypes(include=[np.number]))
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # K-Means clustering
    st.write(f"Running K-Means clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)  # Use the number of clusters from the slider
    kmeans_labels = kmeans.fit_predict(data_pca)
    
    # Add cluster labels to the original data
    data['Cluster'] = kmeans_labels
    
    # Silhouette score
    silhouette_avg = silhouette_score(data_pca, kmeans_labels)
    st.write(f'Silhouette Score: {silhouette_avg:.4f}')
    
    # Display data with cluster labels
    st.write("Clustered Data Preview:")
    st.write(data.head())  # Show the first few rows of the data with the new cluster column
    
    # Plotting
    st.write("Cluster Plot:")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=kmeans_labels, palette='viridis')
    plt.title(f'K-Means Clustering with {num_clusters} Clusters')
    st.pyplot(plt)
    
    # Download button for clustered data
    st.write("Download the clustered dataset:")
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv"
    )
    
    # Filter and show Tourism Inbound and Tourism Outbound for each cluster
    st.write("Tourism Inbound and Outbound for Each Cluster:")
    
    # Group by cluster and calculate the mean for Tourism Inbound and Outbound
    tourism_stats = data.groupby('Cluster')[['Tourism Inbound', 'Tourism Outbound']].mean()
    
    # Display the stats
    st.write(tourism_stats)
    
    # Optional: Show individual countries' tourism data for each cluster
    selected_cluster = st.selectbox('Select Cluster to View Tourism Data', list(data['Cluster'].unique()))
    
    st.write(f"Tourism data for Cluster {selected_cluster}:")
    cluster_data = data[data['Cluster'] == selected_cluster][['Country', 'Tourism Inbound', 'Tourism Outbound']]
    st.write(cluster_data)
    
    # Graphical Representation of Tourism Inbound and Outbound for Each Cluster
    st.write("Graphical Representation of Tourism Inbound and Outbound for Each Cluster:")
    
    # Create a bar plot for Tourism Inbound and Tourism Outbound by Cluster
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for Tourism Inbound by Cluster
    sns.barplot(x=tourism_stats.index, y='Tourism Inbound', data=tourism_stats, ax=ax[0], palette='viridis')
    ax[0].set_title('Tourism Inbound by Cluster')
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Tourism Inbound')
    
    # Bar plot for Tourism Outbound by Cluster
    sns.barplot(x=tourism_stats.index, y='Tourism Outbound', data=tourism_stats, ax=ax[1], palette='viridis')
    ax[1].set_title('Tourism Outbound by Cluster')
    ax[1].set_xlabel('Cluster')
    ax[1].set_ylabel('Tourism Outbound')
    
    st.pyplot(fig)
    
    # Display GDP and population data for each cluster
    st.write("GDP and Population Metrics for Each Cluster:")
    
    # Group by cluster and calculate the mean for GDP and population metrics
    development_stats = data.groupby('Cluster')[['GDP', 'Population 0-14', 'Population 15-64', 'Population 65+', 'Population Total']].mean()
    
    # Display the stats
    st.write(development_stats)
    
    # Graphical representation of GDP and population metrics by cluster
    st.write("Graphical Representation of GDP and Population Metrics for Each Cluster:")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Bar plots for each metric
    sns.barplot(x=development_stats.index, y='GDP', data=development_stats, ax=axes[0, 0], palette='cool')
    axes[0, 0].set_title('GDP by Cluster')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('GDP')
    
    sns.barplot(x=development_stats.index, y='Population 0-14', data=development_stats, ax=axes[0, 1], palette='cool')
    axes[0, 1].set_title('Population 0-14 by Cluster')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Population 0-14')
    
    sns.barplot(x=development_stats.index, y='Population 15-64', data=development_stats, ax=axes[0, 2], palette='cool')
    axes[0, 2].set_title('Population 15-64 by Cluster')
    axes[0, 2].set_xlabel('Cluster')
    axes[0, 2].set_ylabel('Population 15-64')
    
    sns.barplot(x=development_stats.index, y='Population 65+', data=development_stats, ax=axes[1, 0], palette='cool')
    axes[1, 0].set_title('Population 65+ by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Population 65+')
    
    sns.barplot(x=development_stats.index, y='Population Total', data=development_stats, ax=axes[1, 1], palette='cool')
    axes[1, 1].set_title('Population Total by Cluster')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Population Total')

    # Hide the last subplot (empty)
    axes[1, 2].axis('off')
    
    st.pyplot(fig)

    # Decision-making insights
    st.write("### Strategic Recommendations Based on Cluster Analysis")

    # Example decision logic based on GDP and population metrics
    for cluster_id in development_stats.index:
        gdp = development_stats.loc[cluster_id, 'GDP']
        pop_total = development_stats.loc[cluster_id, 'Population Total']
        
        st.write(f"**Cluster {cluster_id} Recommendations:**")
        
        # Example logic: Provide strategies based on GDP and population size
        if gdp > 50000 and pop_total > 50000000:
            st.write("- This cluster has a high GDP and large population. Consider focusing on high-value investments and expanding infrastructure to support economic growth.")
        elif gdp < 20000 and pop_total < 10000000:
            st.write("- This cluster may benefit from targeted economic support and development programs. Enhancing educational and healthcare facilities could be a priority.")
        else:
            st.write("- This cluster shows balanced development. Continuing steady investments in technology and sustainable growth practices could maintain progress.")
