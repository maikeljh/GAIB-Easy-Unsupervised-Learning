import random as rand
import numpy as np
import plotly.graph_objects as go

class KMedoids:
    def __init__(self, k=3):
        self.k = k

    # Place K medoids at random locations
    def random_medoids(self):
        # Define medoids
        self.medoids = []

        # Place K medoids at random locations
        for _ in range(self.k):
            medoid_idx = rand.randint(0, len(self.X) - 1)
            self.medoids.append(medoid_idx)

    # Create clusters
    def create_clusters(self):
        # Define clusters
        self.clusters = []

        # Assign each data to closest medoids
        for _, row in self.X.iterrows():
            # Define distances for each row to all medoids
            distances = []

            # Calculate distance to each medoid
            for medoid_idx in self.medoids:
                medoid = self.X.iloc[medoid_idx]
                distance = np.linalg.norm(np.array(row) - np.array(medoid))
                distances.append(distance)

            # Choose closest distance
            cluster = np.argmin(distances)
            self.clusters.append(cluster)

    # Create new medoids
    def create_new_medoids(self):
        # Define new medoids
        new_medoids = []

        # Get all clusters
        for i in range(self.k):
            # Define cluster
            cluster = []

            # Get all data for each cluster
            for j in range(len(self.X)):
                if self.clusters[j] == i:
                    cluster.append(j)

            # Calculate total dissimilarity within cluster for each data point
            total_dissimilarity = []
            for data_point in cluster:
                dissimilarity = 0
                for other_point in cluster:
                    dissimilarity += np.linalg.norm(np.array(self.X.iloc[data_point]) - np.array(self.X.iloc[other_point]))
                total_dissimilarity.append(dissimilarity)

            # Choose the data point with the minimum dissimilarity as the new medoid
            new_medoid_idx = cluster[np.argmin(total_dissimilarity)]
            new_medoids.append(new_medoid_idx)

        self.medoids = new_medoids

    # Calculate Sum of Dissimilarity
    def calculate_ssd(self):
        # Define dissimilarities
        dissimilarities = []

        for idx, row in self.X.iterrows():
            # Get assigned medoid for each row
            medoid_idx = self.medoids[self.clusters[idx]]

            # Compute dissimilarity
            dissimilarity = np.linalg.norm(np.array(row) - np.array(self.X.iloc[medoid_idx])) ** 2

            # Save dissimilarity
            dissimilarities.append(dissimilarity)

        # Sum all dissimilarities
        SSD = sum(dissimilarities)

        return SSD

    # Main Algorithm
    def fit(self, X, max_iterations=100, tolerance=pow(10, -3)):
        # Save data
        self.X = X

        # Define variables
        iteration = -1
        SSDs = []

        # Step 1: Place K medoids at random locations
        self.random_medoids()

        # Until algorithm converges
        while len(SSDs) <= 1 or (iteration < max_iterations and
                                 np.absolute(SSDs[iteration] - SSDs[iteration - 1]) /
                                 SSDs[iteration - 1] >= tolerance):
            iteration += 1

            # Step 2: Assign all data to closest medoids
            self.create_clusters()

            # Step 3: Calculate new medoids
            self.create_new_medoids()

            # Step 4: Calculate SSD
            SSD = self.calculate_ssd()
            SSDs.append(SSD)

        # Save results
        self.inertia_ = np.min(SSDs)
        self.cluster_centers_ = self.medoids
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
            x=np.array(self.X.iloc[self.medoids])[:, 0],
            y=np.array(self.X.iloc[self.medoids])[:, 1],
            mode='markers',
            name='Medoids',
            marker=dict(size=12, color='black', symbol='star')
        ))

        fig.update_layout(
            title='K-Medoids Clustering',
            xaxis=dict(title='Feature 1'),
            yaxis=dict(title='Feature 2'),
            showlegend=True,
            legend=dict(x=1.02, y=0.98)
        )

        fig.show()

    def predict(self, X):
        # Ensure the model has been fitted before making predictions
        if not hasattr(self, 'medoids'):
            raise RuntimeError("KMedoids model has not been fitted. Call the 'fit' method first.")

        # Create an empty list to store the predicted cluster labels
        predictions = []

        # Assign each new data point to the closest medoid
        for _, row in X.iterrows():
            # Define distances for each row to all medoids
            distances = []

            # Calculate distance to each medoid
            for medoid_idx in self.medoids:
                medoid = self.X.iloc[medoid_idx]
                distance = np.linalg.norm(np.array(row) - np.array(medoid))
                distances.append(distance)

            # Choose closest distance
            cluster = np.argmin(distances)
            predictions.append(cluster)

        return predictions