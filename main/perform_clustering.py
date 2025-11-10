from pathlib import Path

import numpy as np
import joblib
# from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    print("クラスタリング開始")

    pkl_path ="/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-31/dtw_concession_l/dtw_distances_concession_scenes_l.pkl"
    scenes_path = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2023-cyrus2023/concession_scenes_l.pkl"
    scenes_path_2024 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2024-cyrus2023/concession_scenes_r.pkl"
    try:
        loaded_pkl = joblib.load(pkl_path)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path}")
        return
    
    try:
        loaded_scenes = joblib.load(scenes_path)
        print(f"シーンデータを読み込みました.形状: {len(loaded_scenes)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {scenes_path}")
        return
    
    try:
        loaded_scenes_2024 = joblib.load(scenes_path_2024)
        print(f"2024年シーンデータを読み込みました.形状: {len(loaded_scenes_2024)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {scenes_path_2024}")
        return
    all_scenes = loaded_scenes + loaded_scenes_2024
    # all_scenes = loaded_scenes
    print(f"全シーンデータを結合しました.形状: {len(all_scenes)} ")

    # all_scenes = loaded_pkl

    # for i, scene in enumerate(all_scenes):
    #     print(f"scene {i} shape: {scene.shape}")

    # # scene_numpy = np.stack(all_scenes, axis=0)
    # print(f"データ形状: {len(all_scenes)}個のシーン")

    # ##もしscene_numpyの2つ以上シーンがない場合はデータの数が足りない
    # if len(all_scenes) < 2:
    #     print("データの数が足りません")
    #     return
    
    # # DTWを行う
    # print("DTWを実行中...")
    # dtw_distances = cdist_dtw(all_scenes, n_jobs=-1, verbose=True)
    # print("DTW距離行列を計算しました")

    ##クラスタ数を定義する．
    n_clusters = min(2, len(loaded_pkl))  # クラスタ数をデータ数以下に制限
    print(f"クラスタ数: {n_clusters}")

    model = KMedoids(n_clusters=n_clusters, metric="precomputed", init='k-medoids++', random_state=42)
    print("クラスタリングを実行中...")
    model.fit(loaded_pkl)
    print("クラスタリングが完了しました")

    labels = model.labels_
    cluster_ids, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)

    medoid_indices = model.medoid_indices_.tolist()
    templates = [all_scenes[idx] for idx in medoid_indices]

    K_RANGE = range(2, 15)
    OUTPUT_DIR = "/home/arata/rcss/goal-scene-analyzer/data/results/klusters/"

    for i in sorted_indices:
        cluster_id = cluster_ids[i]
        count = counts[i]
        percentage = (count / len(loaded_pkl)) * 100
        print(f"クラスタ {cluster_id}: {count} 個のシーン, 全体の {percentage:.2f}%")

        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"クラスタを構成する元のシーンインデックス（{len(cluster_indices)}件）:")
        print(cluster_indices)

        medoids = getattr(model, "medoid_indices_", [])
        medoids_in_cluster = [m for m in medoids if m in cluster_indices]
        if medoids_in_cluster:
            sub_medoid_relative_indices = [int(np.where(cluster_indices == m)[0][0]) for m in medoids_in_cluster]
        else:
            sub_medoid_relative_indices = []

        print(f"\nサブクラスタの代表は、上記リストの {sub_medoid_relative_indices} 番目の要素です。")

        print("これを元のシーンインデックスに直すと...")
        for rel_idx in sub_medoid_relative_indices:
            original_scene_index = cluster_indices[rel_idx]
            print(f"  -> 代表シーンの元のインデックス: {original_scene_index}")

    processed_dir = Path('/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-02/klusters_model_concession_l')
    model_output_path = processed_dir / 'concession_model_l.pkl'
    joblib.dump(
        {"model": model, "templates": templates, "medoid_indices": medoid_indices},
        model_output_path,
    )
    print(f"学習済みモデルを {model_output_path} に保存")

    # それぞれのクラスタに対してさらにクラスタリングを実行
    # for cluster_id in cluster_ids:
    #     cluster_indices = np.where(labels == cluster_id)[0]
    #     subcluster_size = len(cluster_indices)
    #     print(f"クラスタ {cluster_id} の再クラスタリングを開始 ({subcluster_size} 件)")

    #     if subcluster_size == 0:
    #         print(f"クラスタ {cluster_id} にデータが存在しないためスキップします")
    #         continue

    #     sub_n_clusters = min(3, subcluster_size)
    #     if sub_n_clusters < 2:
    #         print(f"クラスタ {cluster_id} のデータ数が {subcluster_size} 件のため再クラスタリングをスキップします")
    #         continue

    #     sub_distance_matrix = loaded_pkl[np.ix_(cluster_indices, cluster_indices)]

    #     if cluster_id == 0:
    #         sub_n_clusters = 2
    #     elif cluster_id == 1:
    #         sub_n_clusters = 2
    #     else:
    #         sub_n_clusters = 3

        # for k in K_RANGE:
        #     print(f"k = {k}の場合のシルエットスコアを計算中...")
        #     model = KMedoids(n_clusters=k, metric="precomputed", init='k-medoids++', random_state=42)
        #     model.fit(sub_distance_matrix)
        #     label = model.labels_
        #     silhouette_vals = silhouette_samples(sub_distance_matrix, label, metric="precomputed")

        #     plt.figure(figsize=(10, 8))
        #     y_ax_lower, y_ax_upper = 0, 0
        #     y_ticks = []
        #     cluster_labels = np.unique(label)
        #     n_clusters = len(cluster_labels)
        #     for i, c in enumerate(cluster_labels):
        #         c_silhouette_vals = silhouette_vals[label == c]
        #         c_silhouette_vals.sort()
        #         y_ax_upper += len(c_silhouette_vals)
        #         color = cm.jet(float(i) / n_clusters)
        #         plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        #         y_ticks.append((y_ax_lower + y_ax_upper) / 2)
        #         y_ax_lower += len(c_silhouette_vals)

        #     silhouette_avg = np.mean(silhouette_vals)
        #     plt.axvline(silhouette_avg, color="red", linestyle="--")
        #     plt.yticks(y_ticks, cluster_labels)
        #     plt.title(f"Silhouette plot (k={k}, avg={silhouette_avg:.4f})")
        #     plt.xlabel("Silhouette Score")
        #     plt.ylabel("Cluster")
        #     print(f"k = {k}のシルエットスコアの平均: {silhouette_avg:.4f}")

        #     output_path = os.path.join(OUTPUT_DIR, f"silhouette_plot_{cluster_id}_k{k}.png")
        #     plt.savefig(output_path)
        #     plt.close()
        #     print(f"グラフを保存しました: {output_path}")

        # sub_model = KMedoids(n_clusters=sub_n_clusters, metric="precomputed", init='k-medoids++', random_state=42)
        # print(f"  -> サブクラスタリング (クラスタ数 {sub_n_clusters}) を実行中...")
        # sub_model.fit(sub_distance_matrix)

        # if cluster_id == 0:
        #     print("\n--- ▼▼▼ クラスタ0のインデックス対応関係の確認 ▼▼▼ ---")
            
        #     print(f"クラスタ0を構成する元のシーンインデックス（{len(cluster_indices)}件）:")
        #     print(cluster_indices)

        #     sub_medoid_relative_indices = sub_model.medoid_indices_
        #     print(f"\nサブクラスタの代表は、上記リストの {sub_medoid_relative_indices} 番目の要素です。")

        #     print("これを元のシーンインデックスに直すと...")
        #     for rel_idx in sub_medoid_relative_indices:
        #         original_scene_index = cluster_indices[rel_idx]
        #         print(f"  -> 代表シーンの元のインデックス: {original_scene_index}")
            
        #     print("--- ▲▲▲ 確認終了 ▲▲▲ ---\n")

        # if cluster_id == 1:
        #     print("\n--- ▼▼▼ クラスタ1のインデックス対応関係の確認 ▼▼▼ ---")
            
        #     print(f"クラスタ1を構成する元のシーンインデックス（{len(cluster_indices)}件）:")
        #     print(cluster_indices)

        #     sub_medoid_relative_indices = sub_model.medoid_indices_
        #     print(f"\nサブクラスタの代表は、上記リストの {sub_medoid_relative_indices} 番目の要素です。")

        #     print("これを元のシーンインデックスに直すと...")
        #     for rel_idx in sub_medoid_relative_indices:
        #         original_scene_index = cluster_indices[rel_idx]
        #         print(f"  -> 代表シーンの元のインデックス: {original_scene_index}")
            
        #     print("--- ▲▲▲ 確認終了 ▲▲▲ ---\n")

        # if cluster_id == 2:
        #     print("\n--- ▼▼▼ クラスタ2のインデックス対応関係の確認 ▼▼▼ ---")
            
        #     print(f"クラスタ2を構成する元のシーンインデックス（{len(cluster_indices)}件）:")
        #     print(cluster_indices)

        #     sub_medoid_relative_indices = sub_model.medoid_indices_
        #     print(f"\nサブクラスタの代表は、上記リストの {sub_medoid_relative_indices} 番目の要素です。")

        #     print("これを元のシーンインデックスに直すと...")
        #     for rel_idx in sub_medoid_relative_indices:
        #         original_scene_index = cluster_indices[rel_idx]
        #         print(f"  -> 代表シーンの元のインデックス: {original_scene_index}")
            
        #     print("--- ▲▲▲ 確認終了 ▲▲▲ ---\n")

        # sub_labels = sub_model.labels_
        # sub_cluster_ids, sub_counts = np.unique(sub_labels, return_counts=True)
        # sub_sorted_indices = np.argsort(-sub_counts)
        # for idx in sub_sorted_indices:
        #     sub_cluster_id = int(sub_cluster_ids[idx])
        #     sub_count = int(sub_counts[idx])
        #     sub_percentage = (sub_count / subcluster_size) * 100
        #     print(f"    サブクラスタ {sub_cluster_id}: {sub_count} 個, クラスタ {cluster_id} 内の {sub_percentage:.2f}%")

        # processed_dir = Path('/home/arata/rcss/goal-scene-analyzer/data/processed')
        # sub_output_path = processed_dir / f'concession_model_cluster{cluster_id}_ball_x15_falloff_x0p1.pkl'
        # joblib.dump(sub_model, sub_output_path)
        # print(f"  -> サブクラスタリング結果を {sub_output_path} に保存")

if __name__ == "__main__":
    main()
