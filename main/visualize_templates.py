import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

def main():
    model = joblib.load('/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-02/klusters_model_scoring_c/scoring_model_c.pkl')
    pkl_path_2023 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2023-cyrus2023/scoring_scenes_c.pkl"
    pkl_path_2024 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2024-cyrus2023/scoring_scenes_c.pkl"

    try:
        sub_scenes_2023 = joblib.load(pkl_path_2023)
        print(f"2023データを読み込みました.形状: {len(sub_scenes_2023)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path_2023}")
        return
    try:
        sub_scenes_2024 = joblib.load(pkl_path_2024)
        print(f"2024データを読み込みました.形状: {len(sub_scenes_2024)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path_2024}")
        return
    try:
        all_scenes = sub_scenes_2023 + sub_scenes_2024
        # all_scenes = sub_scenes_2023
        print(f"全データを結合しました.形状: {len(all_scenes)} ")
    except Exception as e:
        print(f"サブシーンの結合に失敗しました: {e}")
        return
    try:
        templates = model["templates"]
        medoid_indices = model["medoid_indices"]
        print(f"テンプレート数: {len(templates)}")
    except (AttributeError, FileNotFoundError):
        print("モデルファイルが見つかりません")
        return

    output_dir = '/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-02/templates_scoring_c/'

    for i, template in enumerate(templates):
        original_index = medoid_indices[i]
        output_path = os.path.join(output_dir, f'template_{i}_{original_index}.png')
        visualize_template(template, output_path)
    print("すべてのテンプレートの可視化が完了しました")

def visualize_template(template_data, output_path):
    n_timesteps = template_data.shape[0]

    reshaped_data = template_data.reshape(n_timesteps, 23, 2)

    fig, ax = plt.subplots(figsize=(12, 8))

    x_half = 52.5
    y_half = 34.0
    ax.plot(
        [-x_half, x_half, x_half, -x_half, -x_half],
        [-y_half, -y_half, y_half, y_half, -y_half],
        color='white', linewidth=2
    )

    ax.plot([0, 0], [-y_half, y_half], color='white', linewidth=2)

    ax.set_facecolor('green')
    ax.set_aspect('equal')
    penalty_area_right = plt.Rectangle(
        (x_half - 16.5, -40.32 / 2), 16.5, 40.32,                         
        ec='white',                    
        fc='none',                     
        lw=2                           
        )
    ax.add_patch(penalty_area_right) 
    penalty_area_left = plt.Rectangle(
        (-x_half, -40.32 / 2), 16.5, 40.32,                         
        ec='white',                    
        fc='none',                     
        lw=2                           
        )
    ax.add_patch(penalty_area_left)
    goal_area_right = plt.Rectangle(
        (x_half - 5.5, -18.32 / 2), 5.5, 18.32,                         
        ec='white',                    
        fc='none',                     
        lw=2                           
        )
    ax.add_patch(goal_area_right)
    goal_area_left = plt.Rectangle(
        (-x_half, -18.32 / 2), 5.5,
        18.32,
        ec='white',
        fc='none',
        lw=2
        )
    ax.add_patch(goal_area_left)    

    center_circle = Circle((0, 0), 9.15, color='white', fill=False, linewidth=2)
    ax.add_patch(center_circle)    

    plt.show()

    num_objects = reshaped_data.shape[1]

    for i in range(num_objects):
        # i番目のオブジェクトのx座標とy座標を全時間分取り出す
        x_coords = reshaped_data[:, i, 0]
        y_coords = reshaped_data[:, i, 1]

        if i == 0:  # ボール
            ax.plot(x_coords, y_coords, color='black', linewidth=3, label='Ball', zorder=10)
        else:  # 選手
            if i <= 11:
                ax.plot(x_coords, y_coords, color='red', label=f'Player_{i}')
            else:
                ax.plot(x_coords, y_coords, color='blue', label=f'Player_{i}')

    ax.set_title(os.path.basename(output_path))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)    

    plt.savefig(output_path)
    plt.close(fig) 
    print(f"テンプレートを {output_path} に保存しました")

if __name__ == "__main__":
    main()