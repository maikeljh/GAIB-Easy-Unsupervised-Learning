import numpy as np
import plotly.graph_objects as go

class DBSCAN:
    def __init__(self, eps = 0.5, min_pts = 3):
        # Constructor
        self.eps = eps
        self.min_pts = min_pts

    def region_query(self, P):
        # Initialize neighbours
        neighbours = []
        
        # Find neighbouring points within epsilon distance of point P
        for Pn in range(len(self.X)):
            if np.linalg.norm(self.X.iloc[P] - self.X.iloc[Pn]) < self.eps:
                neighbours.append(Pn)

        return neighbours

    def grow_cluster(self, labels, P, neighbour_pts, cluster_id):
        # Get label of P
        labels[P] = cluster_id

        # Expand the cluster starting from point P
        i = 0
        while i < len(neighbour_pts):
            # Get point of neighbour
            Pn = neighbour_pts[i]

            # Check if noise
            if labels[Pn] == -1:
                labels[Pn] = cluster_id

            # If point hasn't been assigned
            elif labels[Pn] == 0:
                # Add point to cluster
                labels[Pn] = cluster_id

                # Find neighbours of current point
                Pn_neighbour_pts = self.region_query(Pn)

                # If the number of neighbours is valid, add
                if len(Pn_neighbour_pts) >= self.min_pts:
                    neighbour_pts = neighbour_pts + Pn_neighbour_pts

            # Increment i
            i += 1

    def fit(self, X):
        # Save X
        self.X = X

        # Initialize variables
        labels = [0] * len(self.X)
        cluster_id = 0

        # Iterate all points
        for P in range(len(self.X)):
            # If point has already been assigned
            if not (labels[P] == 0):
                continue
            
            # Find neighbours
            neighbour_pts = self.region_query(P)

            # Check if noise
            if len(neighbour_pts) < self.min_pts:
                labels[P] = -1
            else:
                # Create cluster
                cluster_id += 1
                self.grow_cluster(labels, P, neighbour_pts, cluster_id)

        # Save results
        self.labels_ = labels
        self.n_clusters_ = len(np.unique(labels)) - 1
        self.n_noises_ = labels.count(-1)

    def plot_clusters(self):
        # Plot clusters by first and second feature
        fig = go.Figure()

        for label in np.unique(self.labels_):
            if label == -1:
                cluster_data = self.X.iloc[np.where(np.array(self.labels_) == label)]
                fig.add_trace(go.Scatter(
                    x=cluster_data.iloc[:, 0],
                    y=cluster_data.iloc[:, 1],
                    mode='markers',
                    name='Noise',
                    marker=dict(size=8, opacity=0.7)
                ))
            else:
                cluster_data = self.X.iloc[np.where(np.array(self.labels_) == label)]
                fig.add_trace(go.Scatter(
                    x=cluster_data.iloc[:, 0],
                    y=cluster_data.iloc[:, 1],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(size=8, opacity=0.7)
                ))

        fig.update_layout(
            title='DBSCAN Clustering',
            xaxis=dict(title='Feature 1'),
            yaxis=dict(title='Feature 2'),
            showlegend=True,
            legend=dict(x=1.02, y=0.98)
        )

        fig.show()

    def predict(self, X_test):
        # Convert X_test to numpy array for compatibility with self.X
        X_test = X_test.to_numpy()

        # Initialize labels
        predicted_labels = np.zeros(len(X_test), dtype=int)

        # Iterate all new data
        for i, new_P in enumerate(X_test):
            # Initialize new neighbours
            new_neighbours = []

            # Find neighbours of each new data
            for Pn in range(len(self.X)):
                if np.linalg.norm(new_P - self.X.iloc[Pn].to_numpy()) < self.eps:
                    new_neighbours.append(Pn)

            # Check if noise
            if len(new_neighbours) < self.min_pts:
                predicted_labels[i] = -1
            else:
                # Check if any of the new point neighbours belong to existing clusters
                cluster_labels = [self.labels_[Pn] for Pn in new_neighbours if self.labels_[Pn] != -1]

                # Check if noise
                if len(cluster_labels) > 0:
                    # Assign the new point to the first cluster found
                    predicted_labels[i] = cluster_labels[0]
                else:
                    predicted_labels[i] = -1

        return predicted_labels
