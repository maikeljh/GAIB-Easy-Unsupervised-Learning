from src.kmeans import KMeans
from src.kmedoids import KMedoids
from src.dbscan import DBSCAN
import pandas as pd

def main():
    # Pilih algoritma
    print("Masukkan algoritma yang ingin digunakan:\n   1. kmeans\n   2. kmedoids\n   3. DBScan\n")
    algo = int(input("Pilihan: "))

    # Pilih dataset
    data = input("Masukkan dataset yang ingin digunakan: ")
    df = pd.read_csv(data)

    if(algo == 1):
        # Pilih K
        k = int(input("Masukkan nilai k: "))

        # Print data
        print("\nDataset yang digunakan:")
        print(df)
        print()

        # Define KMeans
        kmeans = KMeans(k)

        # Train model
        kmeans.fit(df.iloc[:-1])

        # Print training result
        print("Berikut adalah hasil dari pembelajaran:")
        print("Lowest SSE Value:", kmeans.inertia_)
        print("Final locations of the centorids:")
        print(kmeans.cluster_centers_)
        print("Number of iterations required to converge:", kmeans.n_iter_)
        print("Labels")
        print(kmeans.labels_)

        # Plot example result for first and second feature only
        kmeans.plot_clusters()

        # Predict
        print("Contoh prediksi")
        print(df.iloc[-1:])
        print("Label hasil prediksi:", kmeans.predict(df.iloc[-1:]))
    elif(algo == 2):
        # Pilih K
        k = int(input("Masukkan nilai k: "))

        # Print data
        print("\nDataset yang digunakan:")
        print(df)
        print()

        # Define KMeans
        kmedoids = KMedoids(k)

        # Train model
        kmedoids.fit(df.iloc[:-1])

        # Print training result
        print("Berikut adalah hasil dari pembelajaran:")
        print("Lowest SSD Value:", kmedoids.inertia_)
        print("Final locations of the medoids (index):")
        print(kmedoids.cluster_centers_)
        print("Number of iterations required to converge:", kmedoids.n_iter_)
        print("Labels")
        print(kmedoids.labels_)

        # Plot example result for first and second feature only
        kmedoids.plot_clusters()

        # Predict
        print("Contoh prediksi")
        print(df.iloc[-1:])
        print("Label hasil prediksi:", kmedoids.predict(df.iloc[-1:]))
    elif(algo == 3):
        # Pilih e dan mp
        e = float(input("Masukkan nilai epsilon: "))
        mp = int(input("Masukkan nilai minimal points: "))

        # Print data
        print("\nDataset yang digunakan:")
        print(df)
        print()

        # Define KMeans
        dbscan = DBSCAN(eps=e, min_pts=mp)

        # Train model
        dbscan.fit(df.iloc[:-1])

        # Print training result
        print("Berikut adalah hasil dari pembelajaran:")
        print("Jumlah cluster yang ditemukan:", dbscan.n_clusters_)
        print("Jumlah noise points:", dbscan.n_noises_)
        print("Labels")
        print(dbscan.labels_)

        # Plot example result for first and second feature only
        dbscan.plot_clusters()

        # Predict
        print("Contoh prediksi")
        print(df.iloc[-1:])
        print("Label hasil prediksi:", dbscan.predict(df.iloc[-1:]))
        

if __name__ == "__main__":
    main()