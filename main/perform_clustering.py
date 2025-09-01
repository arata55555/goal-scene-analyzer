import numpy as np
import joblib
from tslearn.clustering import TimeSeriesKMeans

def main():
    print("クラスタリング開始")

    npz_path = "/home/arata/rcss/goal-scene-analyzer/data/results/final_goal_scenes_data_l.npz"
    try:
        loaded_npz = np.load(npz_path)
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {npz_path}")
        return

    all_scenes = []
    for scene_name in loaded_npz.keys():
        all_scenes.append(loaded_npz[scene_name])

    padded_scenes = []
    for scene in all_scenes:
        if scene.shape[0] < 50:
            padded_scene = np.pad(scene, ((0, 50 - scene.shape[0]), (0, 0)), mode='constant')
            padded_scenes.append(padded_scene)
        else:
            padded_scenes.append(scene)

    for i, scene in enumerate(padded_scenes):
        print(f"scene {i} shape: {scene.shape}")

    scene_numpy = np.stack(padded_scenes, axis=0)
    print(f"データ形状: {scene_numpy.shape}")

if __name__ == "__main__":
    main()