import numpy as np
import joblib
# from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids

def main():
    print("クラスタリング開始")

    pkl_path ="/home/arata/rcss/goal-scene-analyzer/data/results/concession_scenes.pkl"
    try:
        loaded_pkl = joblib.load(pkl_path)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path}")
        return

    all_scenes = loaded_pkl 

    for i, scene in enumerate(all_scenes):
        print(f"scene {i} shape: {scene.shape}")

    # scene_numpy = np.stack(all_scenes, axis=0)
    print(f"データ形状: {len(all_scenes)}個のシーン")

    ##もしscene_numpyの2つ以上シーンがない場合はデータの数が足りない
    if len(all_scenes) < 2:
        print("データの数が足りません")
        return
    
    # DTWを行う
    print("DTWを実行中...")
    dtw_distances = cdist_dtw(all_scenes, n_jobs=-1, verbose=True)
    print("DTW距離行列を計算しました")

    ##クラスタ数を定義する．
    n_clusters = min(4, len(all_scenes))

    model = KMedoids(n_clusters=n_clusters, metric="precomputed", init='k-medoids++', random_state=42)
    print("クラスタリングを実行中...")
    model.fit(dtw_distances)
    print("クラスタリングが完了しました")

    labels = model.labels_
    cluster_ids, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)

    for i in sorted_indices:
        cluster_id = cluster_ids[i]
        count = counts[i]
        percentage = (count / len(all_scenes)) * 100
        print(f"クラスタ {cluster_id}: {count} 個のシーン, 全体の {percentage:.2f}%")

    model_output_path = '/home/arata/rcss/goal-scene-analyzer/data/processed/scoring_model.pkl'
    joblib.dump(model, model_output_path)
    print(f"学習済みモデルを {model_output_path} に保存")

if __name__ == "__main__":
    main()