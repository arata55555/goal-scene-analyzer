import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import os

INPUT_NPY_PATH = "/home/arata/rcss/goal-scene-analyzer/data/finish/09-11/concession_scenes.npy"
OUTPUT_DIR = "/home/arata/rcss/goal-scene-analyzer/data/processed/"

K_RANGE = range(3, 21)

def main ():
    print("---エルボー法開始---")
    try:
        all_scenes = np.load(INPUT_NPY_PATH)
        print (f"データを読み込みました．形状: {all_scenes.shape}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {INPUT_NPY_PATH}")
        return
    
    rist = []
    for k in K_RANGE:
        print(f"k = {k} の場合でクラスタリングを実行中...")
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", verbose=True, random_state=42, n_jobs=-1, max_iter=5)

        model.fit(all_scenes)

        rist.append(model.inertia_)
        print(f"k = {k} の場合のinertia: {model.inertia_}")

    print("グラフを生成中...")
    plt.figure(figsize=(12, 7))
    plt.plot(K_RANGE, rist, marker='o', linestyle='-')

    plt.title("エルボー法による最適クラスタ数の推定")
    plt.xlabel("クラスタ数 (k)")
    plt.ylabel("Inertia")
    plt.xticks(K_RANGE)
    plt.grid(True)

    output_dir = os.path.dirname(OUTPUT_DIR)
    output_graph_path = os.path.join(output_dir, "elbow_method.png")
    plt.savefig(output_graph_path)
    print(f"グラフを保存しました: {output_graph_path}")

if __name__ == "__main__":
    main()