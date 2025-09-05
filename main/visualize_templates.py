import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    model = joblib.load('/home/arata/rcss/goal-scene-analyzer/data/processed/scoring_model.pkl')
    try:
        templates = model.cluster_centers_
        print(f"テンプレート数: {len(templates)}")
    except FileNotFoundError:
        print("モデルファイルが見つかりません")
        return
    
    output_dir = '/home/arata/rcss/goal-scene-analyzer/data/results/templates/'

    for i, template in enumerate(templates):
        output_path = os.path.join(output_dir, f'template_{i}.png')
        visualize_template(template, output_path)
    print("すべてのテンプレートの可視化が完了しました。")

def visualize_template(template_data, output_path):
    n_timesteps = template_data.shape[0]

    reshaped_data = template_data.reshape(n_timesteps, 23, 2)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="gray")

    num_objects = reshaped_data.shape[1]
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_objects))

    for i in range(num_objects):
        # i番目のオブジェクトのx座標とy座標を全時間分取り出す
        x_coords = reshaped_data[:, i, 0]
        y_coords = reshaped_data[:, i, 1]

        if i == 0:  # ボール
            ax.plot(x_coords, y_coords, color='black', linewidth=3, label='Ball', zorder=10)
        else:  # 選手
            ax.plot(x_coords, y_coords, color=colors[i], label=f'Player_{i}')

    ax.set_title(os.path.basename(output_path))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)
    
    # 4. ファイルに保存
    plt.savefig(output_path)
    plt.close(fig) # メモリを解放するために図を閉じる
    print(f"テンプレートを {output_path} に保存しました。")

if __name__ == "__main__":
    main()