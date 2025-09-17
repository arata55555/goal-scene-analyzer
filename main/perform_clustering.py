import numpy as np
import joblib
from tslearn.clustering import TimeSeriesKMeans

def main():
    print("クラスタリング開始")

    npy_path ="/home/arata/rcss/goal-scene-analyzer/data/finish/09-11/concession_scenes.npy"
    try:
        loaded_npy = np.load(npy_path)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {npy_path}")
        return

    all_scenes = loaded_npy  

    for i, scene in enumerate(all_scenes):
        print(f"scene {i} shape: {scene.shape}")

    scene_numpy = np.stack(all_scenes, axis=0)
    print(f"データ形状: {scene_numpy.shape}")

    ##もしscene_numpyの2つ以上シーンがない場合はデータの数が足りない
    if scene_numpy.shape[0] < 2:
        print("データの数が足りません")
        return

    ##クラスタ数を定義する．
    n_clusters = min(6, scene_numpy.shape[0])

    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=42, n_jobs=-1)
    model.fit(scene_numpy)

    labels = model.labels_
    cluster_ids, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)

    for i in sorted_indices:
        cluster_id = cluster_ids[i]
        count = counts[i]
        percentage = (count / scene_numpy.shape[0]) * 100
        print(f"クラスタ {cluster_id}: {count} 個のシーン, 全体の {percentage:.2f}%")

    model_output_path = '/home/arata/rcss/goal-scene-analyzer/data/processed/scoring_model.pkl'
    joblib.dump(model, model_output_path)
    print(f"学習済みモデルを {model_output_path} に保存")

if __name__ == "__main__":
    main()