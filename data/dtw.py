import numpy as np
import joblib
from tslearn.metrics import cdist_dtw

INPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/finish/09-26/scoring_scenes.pkl"
OUTPUT_PKL_PATH = "/home/arata/rcss/goal-scene-analyzer/data/processed/dtw_distances_scoring.pkl"
def main():
    print("DTWの計算を開始します")
    try:
        all_scenes = joblib.load(INPUT_PKL_PATH)
        print(f"データを読み込みました．形状: {len(all_scenes)}")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {INPUT_PKL_PATH}")
        return
    
    dtw_distances = cdist_dtw(all_scenes, n_jobs=-1, verbose=True)
    print("DTW距離行列を計算しました")

    joblib.dump(dtw_distances, OUTPUT_PKL_PATH)
    print(f"DTW距離行列を保存しました: {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    main()
