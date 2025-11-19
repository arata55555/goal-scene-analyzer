import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

def main():
    model = joblib.load('/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-14/klusters_model_concession_ball_l/model_concession_ball_l.pkl')
    pkl_path_2023 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/goal-scene-analyzer_helios-base_vs_mars/concession_scenes_l.pkl"
    # pkl_path_2024 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/10-28/helios2024-cyrus2023/scoring_scenes_r.pkl"

    try:
        sub_scenes_2023 = joblib.load(pkl_path_2023)
        print(f"2023データを読み込みました.形状: {len(sub_scenes_2023)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path_2023}")
        return
    # try:
    #     sub_scenes_2024 = joblib.load(pkl_path_2024)
    #     print(f"2024データを読み込みました.形状: {len(sub_scenes_2024)} ")
    # except FileNotFoundError:
    #     print(f"ファイルが見つかりません: {pkl_path_2024}")
    #     return
    try:
        # all_scenes = sub_scenes_2023 + sub_scenes_2024
        all_scenes = sub_scenes_2023
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
    
    if "labels" in model:
        labels = model["labels"]
        print(f"ラベル情報が見つかりました. ラベル数: {len(labels)}")
    else:
        print("ラベル情報が見つかりません")
        labels = None

    output_dir = '/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-14/templates_concession_l/'
    os.makedirs(output_dir, exist_ok=True)

    for i, template in enumerate(templates):
        original_index = medoid_indices[i]
        output_path = os.path.join(output_dir, f'template_{i}_{original_index}.png')

        cluster_members_scenes = []
        if labels is not None:
            cluster_members_indices = np.where(labels == i)[0]
            cluster_members_indices = [idx for idx in cluster_members_indices if idx != original_index]
            cluster_members_scenes = [all_scenes[idx] for idx in cluster_members_indices]
            print(f"テンプレート {i} のクラスターメンバー数: {len(cluster_members_scenes)}")
        else:
            print(f"テンプレート {i} のクラスターメンバー情報はありません") 
        visualize_template(template, cluster_members_scenes, output_path)
    print("すべてのテンプレートの可視化が完了しました")

def visualize_template(template_data, cluster_members, output_path):
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


    for member_scene in cluster_members:
        member_n_timesteps = member_scene.shape[0]
        expected_features = 46 
        if member_scene.shape[1] != expected_features:
            print(f"警告: スキップします。メンバーの形状 {member_scene.shape} が (T, {expected_features}) ではありません。")
            continue
        try:
            member_reshaped = member_scene.reshape(member_n_timesteps, 23, 2)
        except Exception as e:
            print(f"警告: スキップします。メンバーの形状の変換中にエラーが発生しました: {e}")
            continue
        num_objects_member = member_reshaped.shape[1]
        for i in range(num_objects_member):
            x_coords = member_reshaped[:, i, 0]
            y_coords = member_reshaped[:, i, 1]
            if i == 0:  
                ax.plot(x_coords, y_coords, color='lightgreen', alpha=0.3, zorder=1, label=None)

    reshaped_data = template_data.reshape(n_timesteps, 23, 2)
    num_objects = reshaped_data.shape[1]

    for i in range(num_objects):
        x_coords = reshaped_data[:, i, 0]
        y_coords = reshaped_data[:, i, 1]

        if i == 0: 
            ax.plot(x_coords, y_coords, color='black', linewidth=3, label='Ball', zorder=10)
        else:  
            if i <= 11:
                ax.plot(x_coords, y_coords, color='red', label=f'Player_{i}', zorder=5)
            else:
                ax.plot(x_coords, y_coords, color='blue', label=f'Player_{i}', zorder=5)

    ax.set_title(os.path.basename(output_path))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)    

    plt.savefig(output_path)
    plt.close(fig) 
    print(f"テンプレートを {output_path} に保存しました")

if __name__ == "__main__":
    main()