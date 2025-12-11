import joblib
import numpy as np
import os
import pandas as pd
from tslearn.metrics import cdist_dtw

#pathはあとで変更
INPUT_MODEL_PKL_PATH = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-02/klusters_model_scoring_r/scoring_model_r.pkl"
INPUT_PKL_PATH_2023 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2023-cyrus2023/scoring_scenes_r.pkl"
INPUT_PKL_PATH_2024 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2024-cyrus2023/scoring_scenes_r.pkl"

OUTPUT_CSV_PATH = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-02/classify_scoring_r/classification_results.csv"

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

    bundle = joblib.load(INPUT_MODEL_PKL_PATH)
    model = bundle["model"]
    templates = bundle["templates"]
    medoid_indices = bundle.get("medoid_indices", list(range(len(templates))))
    print(f"テンプレート数: {len(templates)}, シーン数: 2023年データ {len(scenes_2023)}, 2024年データ {len(scenes_2024)}")

    counts_2023 = assign_scenes_to_templates(scenes_2023, templates)
    counts_2024 = assign_scenes_to_templates(scenes_2024, templates)

    if len(medoid_indices) != len(templates):
        raise ValueError("テンプレート数と medoid_indices の数が一致しません")

    results_df = pd.DataFrame({
        "OriginalSceneIndex": medoid_indices,
        "2023_count": counts_2023,
        "2024_count": counts_2024
    })
    results_df.index.name = "Template_ID"

    results_df["2023_percentage"] = (results_df["2023_count"] / len(scenes_2023)) * 100 if len(scenes_2023) > 0 else 0
    results_df["2024_percentage"] = (results_df["2024_count"] / len(scenes_2024)) * 100 if len(scenes_2024) > 0 else 0

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