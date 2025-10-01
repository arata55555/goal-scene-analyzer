import joblib
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

INPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/processed/dtw_distances_concession.pkl"
OUTPUT_DIR = "/home/arata/rcss/goal-scene-analyzer/data/results/klusters/"
K_RANGE = range(2, 20)

def main():
    print("シルエットスコアの計算を開始します")
    try:
        dtw_distances = joblib.load(INPUT_PKL_PATH)
        print(f"DTW距離行列を読み込みました.形状: {dtw_distances.shape}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {INPUT_PKL_PATH}")
        return

    for k in K_RANGE:
        print(f"k = {k}の場合のシルエットスコアを計算中...")
        model = KMedoids(n_clusters=k, metric="precomputed", init='k-medoids++', random_state=42)
        model.fit(dtw_distances)
        labels = model.labels_
        silhouette_vals = silhouette_samples(dtw_distances, labels, metric="precomputed")

        plt.figure(figsize=(10, 8))
        y_ax_lower, y_ax_upper = 0, 0
        y_ticks = []
        cluster_labels = np.unique(labels)
        n_clusters = len(cluster_labels)
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[labels == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
            y_ticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--")
        plt.yticks(y_ticks, cluster_labels)
        plt.title(f"Silhouette plot (k={k}, avg={silhouette_avg:.4f})")
        plt.xlabel("Silhouette Score")
        plt.ylabel("Cluster")
        print(f"k = {k}のシルエットスコアの平均: {silhouette_avg:.4f}")

        output_path = os.path.join(OUTPUT_DIR, f"silhouette_plot_k{k}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"グラフを保存しました: {output_path}")

if __name__ == "__main__":
    main()
