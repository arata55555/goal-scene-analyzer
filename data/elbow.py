import numpy as np
import matplotlib.pyplot as plt
import joblib
# from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
import os

INPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/results/scoring_scenes.pkl"
OUTPUT_DIR = "/home/arata/rcss/goal-scene-analyzer/data/processed/"

K_RANGE = range(3, 21)

def main ():
    print("---エルボー法開始---")
    try:
        all_scenes = joblib.load(INPUT_PKL_PATH)
        print (f"データを読み込みました．形状: {len(all_scenes)}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {INPUT_PKL_PATH}")
        return

    dtw_distances = cdist_dtw(all_scenes, n_jobs=-1, verbose=True)
    print("DTW距離行列を計算しました")
    
    rist = []
    for k in K_RANGE:
        print(f"k = {k} の場合でクラスタリングを実行中...")
        model = KMedoids(n_clusters=k, metric="precomputed", init='k-medoids++', random_state=42)
        model.fit(dtw_distances)

        rist.append(model.inertia_)
        print(f"k = {k} の場合のinertia: {model.inertia_}")

    print("グラフを生成中...")
    plt.figure(figsize=(12, 7))
    plt.plot(K_RANGE, rist, marker='o', linestyle='-')

    plt.title("Elbow Method for Optimal k")
    plt.xlabel("klusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(K_RANGE)
    plt.grid(True)

    output_dir = os.path.dirname(OUTPUT_DIR)
    output_graph_path = os.path.join(output_dir, "elbow_method_scoring.png")
    plt.savefig(output_graph_path)
    print(f"グラフを保存しました: {output_graph_path}")

if __name__ == "__main__":
    main()