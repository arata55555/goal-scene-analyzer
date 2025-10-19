import joblib
import numpy as np
import os
import pandas as pd
from tslearn.metrics import cdist_dtw

#pathはあとで変更
INPUT_MODEL_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/models/dtw_model.pkl"
INPUT_PKL_PATH_2023 = "/home/arata/rcss/goal-scene-analyzer/data/processed/concession_scenes.pkl"
INPUT_PKL_PATH_2024 = "/home/arata/rcss/goal-scene-analyzer/data/processed/concession_scenes_2024.pkl"

OUTPUT_CSV_PATH = "/home/arata/rcss/goal-scene-analyzer/data/processed/dtw_results/"

def assign_scenes_to_templates(scenes_list, templates):
    if not scenes_list:
        print("シーンリストが空です。割り当て処理をスキップします。")
        return []
    
    print(f"シーン数: {len(scenes_list)}, テンプレート数: {len(templates)}")

    distance_matrix = cdist_dtw(scenes_list, templates, n_jobs=-1, verbose=True)
    print("距離行列を計算しました")

    assigned_indices = np.argmin(distance_matrix, axis=1)
    print("シーンをテンプレートに割り当てました")

    counts = np.bincount(assigned_indices, minlength=len(templates))

    return counts

def main():
    print("割り当てと集計処理を開始します")

    try:
        model = joblib.load(INPUT_MODEL_PKL_PATH)
        print("モデルを読み込みました")

        scenes_2023 = joblib.load(INPUT_PKL_PATH_2023)

        scenes_2024 = joblib.load(INPUT_PKL_PATH_2024)

    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
        return
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return
    
    templates = model.cluster_centers_
    num_templates = len(templates)
    print(f"テンプレート数: {num_templates}, シーン数: 2023年データ {len(scenes_2023)}, 2024年データ {len(scenes_2024)}")

    counts_2023 = assign_scenes_to_templates(scenes_2023, templates)
    counts_2024 = assign_scenes_to_templates(scenes_2024, templates)

    results_df = pd.DataFrame({
        "20223_count": counts_2023,
        "2024_count": counts_2024
    })
    results_df.index.name = "Template_ID"

    print("結果を表示します:")
    print(results_df)

    try:
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results_df.to_csv(OUTPUT_CSV_PATH)
        print(f"結果を{OUTPUT_CSV_PATH}に保存しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
if __name__ == "__main__":
    main()