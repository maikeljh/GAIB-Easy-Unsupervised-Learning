import random as rand
import numpy as np
import plotly.graph_objects as go

class KMeans:
    def __init__(self, k = 3):
        self.k = k

    # Place K centroids at random locations
    def random_centroids(self):
        # Define centroids
        self.centroids = []

        # Place K centroids at random locations
        for _ in range(self.k):
            centroid = self.X.iloc[rand.randint(0, len(self.X) - 1)]
            self.centroids.append(centroid)
    
    # Create clusters
    def create_clusters(self):
        # Define clusters
        self.clusters = []
        
        # Assign each data to closest centroids
        for _, row in self.X.iterrows():
            # Define distances for each row to all centroids
            distances = []

            # Calculate distance to each centroid
            for centroid in self.centroids:
                distance = np.linalg.norm(np.array(row) - np.array(centroid))
                distances.append(distance)

            # Choose closest distance
            cluster = np.argmin(distances)
            self.clusters.append(cluster)

    # Create new centroids
    def create_new_centroids(self):
        # Define new centroids
        new_centroids = []

        # Get all clusters
        for i in range(self.k):
            # Define cluster
            cluster = []

            # Get all data for each cluster
            for j in range(len(self.X)):
                if self.clusters[j] == i:
                    cluster.append(self.X.iloc[j])
            
            # Calculate mean of the data within cluster
            mean_centroid = np.mean(cluster, axis = 0)
            new_centroids.append(mean_centroid)

        self.centroids = new_centroids
    
    # Calculate Sum of Squared Errors
    def calculate_sse(self):
        # Define errors
        errors = []

        for idx, row in self.X.iterrows():
            # Get assigned centroid for each row
            centroid = self.centroids[self.clusters[idx]]

            # Compute error
            error = (np.linalg.norm(np.array(row) - np.array(centroid))) ** 2

            # Save error
            errors.append(error)
        
        # Sum all errors
        SSE = sum(errors)

        return SSE
    
    # Main Algorithm
    def fit(self, X, max_iterations = 100, tolerance = pow(10, -3)):
        # Save data
        self.X = X

        # Define variables
        iteration = -1
        SSES = []

        # Step 1 : Place K centroids at random location
        self.random_centroids()

        # Until algorithm converges
        while len(SSES) <= 1 or (iteration < max_iterations and 
                                 np.absolute(SSES[iteration] - SSES[iteration - 1]) / 
                                 SSES[iteration - 1] >= tolerance):
            iteration += 1

            # Step 2 : Assign all data to closest centroids
            self.create_clusters()

            # Step 3 : Calculate new centroids
            self.create_new_centroids()

            # Step 4 : Calculate SSE
            SSE = self.calculate_sse()
            SSES.append(SSE)
        
        # Save results
        self.inertia_ = np.min(SSES)
        self.cluster_centers_ = self.centroids
        self.n_iter_ = iteration
        self.labels_ = self.clusters

    # Function to plot the data points with different colors for each cluster label
    def plot_clusters(self):
        fig = go.Figure()

        for label in range(self.k):
            cluster_data = self.X.iloc[np.where(np.array(self.clusters) == label)]
            fig.add_trace(go.Scatter(
                x=cluster_data.iloc[:, 0],
                y=cluster_data.iloc[:, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(size=8, opacity=0.7)
            ))

        fig.add_trace(go.Scatter(
            x=np.array(self.cluster_centers_)[:, 0],
            y=np.array(self.cluster_centers_)[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(size=12, color='black', symbol='star')
        ))

        fig.update_layout(
            title='K-Means Clustering',
            xaxis=dict(title='Feature 1'),
            yaxis=dict(title='Feature 2'),
            showlegend=True,
            legend=dict(x=1.02, y=0.98)
        )

        fig.show()

    def predict(self, X):
        # Ensure the model has been fitted before making predictions
        if not hasattr(self, 'centroids'):
            raise RuntimeError("KMeans model has not been fitted. Call the 'fit' method first.")

        # Create an empty list to store the predicted cluster labels
        predictions = []

        # Assign each new data point to the closest centroid
        for _, row in X.iterrows():
            # Define distances for each row to all centroids
            distances = []

            # Calculate distance to each centroid
            for centroid in self.centroids:
                distance = np.linalg.norm(np.array(row) - np.array(centroid))
                distances.append(distance)

            # Choose closest distance
            cluster = np.argmin(distances)
            predictions.append(cluster)

        return predictions