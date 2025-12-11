import numpy as np
import joblib
from tslearn.metrics import cdist_dtw
import os


INPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/finish/09-26/concession_scenes.pkl"
INPUT_PKL_PATH_2024 = "/home/arata/rcss/goal-scene-analyzer/data/finish/2024-06/concession_scenes_2024.pkl"
OUTPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/processed/"

DIM = 46

def build_scale_vector(scenes, ball_weights, falloff_rates):
    weighted_scenes = []
    base_player_weight = 0
    for s in scenes:
        s_w = np.zeros_like(s, dtype=float)
        for i, frame in enumerate(s):
            w = np.ones(DIM)
            ball_pos = frame[0:2]
            w[0:2] = ball_weights

            player_pos = frame[2:46].reshape(22, 2)
            dist = np.linalg.norm(player_pos - ball_pos, axis=1)
            sorted_dist = np.argsort(dist)
            ranked_weights = base_player_weight + (falloff_rates * np.arange(22))
            player_weights = np.empty_like(ranked_weights)
            player_weights[sorted_dist] = ranked_weights

            for j, player_weight in enumerate(player_weights):
                player_start_idx = 2 + j * 2
                player_end_idx = player_start_idx + 2
                w[player_start_idx:player_end_idx] = player_weight
            s_w[i] = frame * w
        weighted_scenes.append(s_w)
    return weighted_scenes



def main():
    print("DTWの計算を開始します")
    list_string = ["concession", "scoring"]
    list_string2 = ["scenes_l", "scenes_r", "scenes_c"]
    for ls in list_string:
        for ls2 in list_string2:
            INPUT_PKL_PATH = f"/home/arata/rcss/work/goal-scene-analyzer/data/finish/12-10/data_analyze_heliosbase_itandroids_1/{ls}_{ls2}.pkl"
            # INPUT_PKL_PATH_2024 = f"/home/arata/rcss/goal-scene-analyzer/data/finish/2024-06/{ls}_{ls2}_2024.pkl"
            print(f"現在処理中のデータ: {ls}_{ls2}")
            try:
                sub_scenes = joblib.load(INPUT_PKL_PATH)
                print(f"データを読み込みました.形状: {len(sub_scenes)} ")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {INPUT_PKL_PATH}")
                return
            # try:
            #     sub_scenes_2024 = joblib.load(INPUT_PKL_PATH_2024)
            #     print(f"2024データを読み込みました.形状: {len(sub_scenes_2024)} ")
            # except FileNotFoundError:
            #     print(f"ファイルが見つかりません: {INPUT_PKL_PATH_2024}")
            #     return
            
            all_scenes = sub_scenes # + sub_scenes_2024

            if not all_scenes:
                print("シーンデータが空です。処理を終了します。")
                return
            
            print(f"総シーン数: {len(all_scenes)}")
            
            X = [np.nan_to_num(np.asarray(s, float), nan=0.0) for s in all_scenes]

            ball_weights = [1.0]
            falloff_rates = [0]

            for bl in ball_weights:
                for fr in falloff_rates:
                    print(f"現在のパラメータ: ボール重み倍率 = {bl}, プレイヤーの重み = {fr}")
                    Xw = build_scale_vector(X, bl, fr)
                    med_len = int(np.median([s.shape[0] for s in Xw]))
                    radius = max(1, int(0.10 * med_len))

                    dtw_distances = cdist_dtw(Xw, n_jobs=-1, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius, verbose=True)
                    print("DTW距離行列を計算しました")
                    step_tag = f"{fr:.1f}".replace(".", "p")
                    output_filename = f"dtw_distances_concession_ball_x{int(bl)}_falloff_x{step_tag}.pkl"
                    output_path = os.path.join(OUTPUT_PKL_PATH, output_filename)

                    joblib.dump(dtw_distances, output_path)
                    print(f"DTW距離行列を保存しました: {output_path}")

if __name__ == "__main__":
    main()
