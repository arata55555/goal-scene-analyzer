import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
import numpy as np 
import os 

def main():
    """
    メインの処理を行う関数。
    モデルとデータを読み込み、クラスタごとに可視化関数を呼び出す。
    """

    model = joblib.load('/home/arata/rcss/work/goal-scene-analyzer/data/finish/12-1/klusters_model_concession_ball_r/model_concession_ball_r.pkl')
    pkl_path_2023 = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/11-30/11-30/concession_scenes_r.pkl"
    # pkl_path_2024 = "..." # 将来的に2024年データを追加

    try:
        sub_scenes_2023 = joblib.load(pkl_path_2023)
        print(f"2023データを読み込みました.形状: {len(sub_scenes_2023)} ")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pkl_path_2023}")
        return

    # try:
    #     sub_scenes_2024 = joblib.load(pkl_path_2024)
    #     ...
    # except FileNotFoundError:
    #     ...
    
    try:
        # all_scenes = sub_scenes_2023 + sub_scenes_2024 # 2024年データも使う場合
        all_scenes = sub_scenes_2023
        print(f"全データを結合しました.形状: {len(all_scenes)} ")
    except Exception as e:
        print(f"サブシーンの結合に失敗しました: {e}")
        return

    try:
        templates = model["templates"]
        medoid_indices = model["medoid_indices"]  
        print(f"テンプレート数: {len(templates)}")
    except (AttributeError, FileNotFoundError, KeyError):
        print("モデルファイルが見つかりません、または必要なキー（templates, medoid_indices）がありません")
        return
    
    if "labels" in model:
        labels = model["labels"] 
        print(f"ラベル情報が見つかりました. ラベル数: {len(labels)}")
    else:
        print("ラベル情報が見つかりません")
        labels = None

    output_dir = '/home/arata/rcss/work/goal-scene-analyzer/data/finish/12-1/templates_lambda_40_concession_ball_r/'
    os.makedirs(output_dir, exist_ok=True)  

    for i, template in enumerate(templates):
        original_index = medoid_indices[i] 
        output_path = os.path.join(output_dir, f'template_{i}_{original_index}.png') 

        cluster_members_scenes = [] 
        if labels is not None:
            cluster_members_indices = np.where(labels == i)[0]
            
            # cluster_members_indices = [idx for idx in cluster_members_indices if idx != original_index]
            
            cluster_members_scenes = [all_scenes[idx] for idx in cluster_members_indices]
            print(f"テンプレート {i} のクラスターメンバー数: {len(cluster_members_scenes)}")
        else:
            print(f"テンプレート {i} のクラスターメンバー情報はありません") 
        
        visualize_template_heatmap(template, cluster_members_scenes, output_path)
    
    print("すべてのテンプレートの可視化が完了しました")

def visualize_template_heatmap(template_data, cluster_members, output_path):
    """
    メンバーのボール軌道をヒートマップとして描画し、代表軌道を線で上書きする関数
    """

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
        ec='white', fc='none', lw=2 
    )
    ax.add_patch(penalty_area_left)
    
    goal_area_right = plt.Rectangle(
        (x_half - 5.5, -18.32 / 2), 5.5, 18.32,
        ec='white', fc='none', lw=2
    )
    ax.add_patch(goal_area_right)
    
    goal_area_left = plt.Rectangle(
        (-x_half, -18.32 / 2), 5.5, 18.32,
        ec='white', fc='none', lw=2
    )
    ax.add_patch(goal_area_left) 

    center_circle = Circle((0, 0), 9.15, color='white', fill=False, linewidth=2)
    ax.add_patch(center_circle)

    STEP = 0.5

    all_ball_x = []
    all_ball_y = []
    all_scene_ids = []

    total_scenes = len(cluster_members)
    
    for i, scene_data in enumerate(cluster_members):
        try:
            n_timesteps = scene_data.shape[0]
            
            reshaped_scene = scene_data.reshape(n_timesteps, 23, 2)
            
            ball_x = reshaped_scene[:, 0, 0] 
            ball_y = reshaped_scene[:, 0, 1] 

            scene_x = []
            scene_y = []

            scene_x.append(ball_x[0])
            scene_y.append(ball_y[0])

            for t in range(1, len(ball_x)):
                start_x, start_y = ball_x[t-1], ball_y[t-1]
                end_x, end_y = ball_x[t], ball_y[t]

                dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

                if dist > 0:
                    num_steps = int(np.ceil(dist / STEP))
                    interp_x = np.linspace(start_x, end_x, num=num_steps+1, endpoint=False)[1:]
                    interp_y = np.linspace(start_y, end_y, num=num_steps+1, endpoint=False)[1:]

                    scene_x.extend(interp_x)
                    scene_y.extend(interp_y)
            all_ball_x.extend(scene_x)
            all_ball_y.extend(scene_y)

            ids = [i] * len(scene_x)
            all_scene_ids.extend(ids)
        except Exception as e:
            print(f"クラスターメンバーの処理中にエラーが発生しました: {e}")
            continue 

    if len(all_ball_x) > 0:

        field_extent = [-x_half, x_half, -y_half, y_half]

        weights = np.ones(len(all_ball_x))/len(all_ball_x)

        hb = ax.hexbin(
            all_ball_x,    
            all_ball_y,     
            gridsize=40,    
            extent=field_extent, 
            cmap='inferno', 
            mincnt=None,       
            alpha=0.7,
            C=all_scene_ids,
            reduce_C_function=lambda x: len(set(x)) / total_scenes
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('ball position density')
    else:
        print("クラスターメンバーにボール位置データがありません")

    ax.set_title(os.path.basename(output_path))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35) 

    plt.savefig(output_path)
    plt.close(fig) 
    print(f"テンプレートを {output_path} に保存しました")

if __name__ == "__main__":
    main()