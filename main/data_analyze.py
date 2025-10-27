import re
from bisect import bisect_left
import numpy as np
import os
import gzip
import joblib

rcg_folder_path = "/home/arata/rcss/goal-scene-analyzer/data/goal-scene-analyzer_test/"
CYCLES_BEFORE_GOAL = 50  
OUTPUT_FILENAME_L = "final_goal_scenes_data_l.npz"
OUTPUT_FILENAME_R = "final_goal_scenes_data_r.npz"
npz_folder_path = "/home/arata/rcss/goal-scene-analyzer/data/results"

def define_scenes(cycle, goal_cycle_l, goal_cycle_r, duration = 50):
    scenes_l = []
    scenes_r = []
    nearest_play_on_l = []
    nearest_play_on_r = []

    for goal_l in goal_cycle_l:
        idx = bisect_left(cycle, goal_l)
        if idx < len(cycle):
            nearest_play_on_l.append(cycle[idx-1])
        else:
            nearest_play_on_l.append(0)

    for goal_r in goal_cycle_r:
        idx = bisect_left(cycle, goal_r)
        if idx < len(cycle):
            nearest_play_on_r.append(cycle[idx-1])
        else:
            nearest_play_on_r.append(0)

    for i,(goal_l, playon) in enumerate(zip(goal_cycle_l, nearest_play_on_l)):
        start_cycle = max(playon, goal_l - duration)
        scenes_l.append({
            "name": f"goal_{i+1}_at_cycle_{goal_l-1}",
            "start_cycle": start_cycle,
            "end_cycle": goal_l-1
        })

    for i,(goal_r, playon) in enumerate(zip(goal_cycle_r, nearest_play_on_r)):
        start_cycle = max(playon, goal_r - duration)
        scenes_r.append({
            "name": f"goal_{i+1}_at_cycle_{goal_r-1}",
            "start_cycle": start_cycle,
            "end_cycle": goal_r-1
        })

    return scenes_l, scenes_r

def extract_data(file_path, scenes_l, scenes_r):
    results_l = {scene_l['name']: [] for scene_l in scenes_l}
    results_r = {scene_r['name']: [] for scene_r in scenes_r}

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.startswith('(show'):
                continue

            match_cycle = re.search(r'\(show\s+(\d+)', line)
            if not match_cycle:
                continue
            current_cycle = int(match_cycle.group(1))

            for scene_l in scenes_l:
                if scene_l['start_cycle'] <= current_cycle <= scene_l['end_cycle']:
                    match_ball = re.search(r'\(\(b\) ([-\d\.]+) ([-\d\.]+)', line)
                    ball_x, ball_y = (float(p) for p in match_ball.groups()) if match_ball else (np.nan, np.nan)

                    raw_data = [ball_x, ball_y]
                    player_positions = [np.nan] * 44

                    players_found = re.findall(r'\(\(([lr])\s+(\d+)\)\s+\S+\s+\S+\s+([-\d\.]+)\s+([-\d\.]+)', line)
                    for team, unum, x, y in players_found:
                        unum = int(unum)
                        x, y = float(x), float(y)
                        offset = 0 if team == 'l' else 22
                        index = offset + (unum - 1) * 2
                        if index < len(player_positions):
                            player_positions[index] = x
                            player_positions[index + 1] = y 

                    raw_data.extend(player_positions)

                    results_l[scene_l['name']].append(raw_data)

                    break

            for scene_r in scenes_r:
                if scene_r['start_cycle'] <= current_cycle <= scene_r['end_cycle']:
                    match_ball = re.search(r'\(\(b\) ([-\d\.]+) ([-\d\.]+)', line)
                    ball_x, ball_y = (float(p) for p in match_ball.groups()) if match_ball else (np.nan, np.nan)

                    raw_data = [ball_x, ball_y]
                    player_positions = [np.nan] * 44

                    players_found = re.findall(r'\(\(([lr])\s+(\d+)\)\s+\S+\s+\S+\s+([-\d\.]+)\s+([-\d\.]+)', line)
                    for team, unum, x, y in players_found:
                        unum = int(unum)
                        x, y = float(x), float(y)
                        offset = 0 if team == 'l' else 22
                        index = offset + (unum - 1) * 2
                        if index < len(player_positions):
                            player_positions[index] = x
                            player_positions[index + 1] = y 

                    raw_data.extend(player_positions)

                    results_r[scene_r['name']].append(raw_data)

                    break
                
    return results_l, results_r

# def padded_data(scenes: list, target_len: int) -> np.ndarray:
#     padded_scenes = []
#     for scene in scenes:
#         if isinstance(scene, list):
#             scene = np.array(scene)
#         if scene.shape[0] < target_len:
#             padded_scene = np.pad(scene, ((0, target_len - scene.shape[0]), (0, 0)), mode='edge')
#             padded_scenes.append(padded_scene)
#         elif scene.shape[0] > target_len:
#             trimmed_scene = scene[-target_len:, :]
#             padded_scenes.append(trimmed_scene)
#         else:
#             padded_scenes.append(scene)

#     for i, scene in enumerate(padded_scenes):
#         print(f"scene {i} shape: {scene.shape}")

#     return np.stack(padded_scenes, axis=0)

def main():
    all_data_l_r = []
    all_data_l_l = []
    all_data_l_c = []
    all_data_r_r = []
    all_data_r_l = []
    all_data_r_c = []

    if not os.path.isdir(rcg_folder_path):
        print(f"エラー: 指定されたフォルダが見つかりません: {rcg_folder_path}")
        return
    for filename in os.listdir(rcg_folder_path):
        rcg_file_path = os.path.join(rcg_folder_path, filename)
        if not filename.endswith(".rcg.gz"):
            continue

        cycle = []
        goal_cycle_l = []
        goal_cycle_r = []
        with gzip.open(rcg_file_path, "rt", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"(\d+)\s+play_on", line)
                if match: cycle.append(int(match.group(1)))
                goal_match_l = re.search(r"(\d+)\s+goal_l", line)
                if goal_match_l:
                    goal_cycle_l.append(int(goal_match_l.group(1)))

                goal_match_r = re.search(r"(\d+)\s+goal_r", line)
                if goal_match_r:
                    goal_cycle_r.append(int(goal_match_r.group(1)))    

        print(f"処理を開始: {rcg_file_path}")
        
        scenes_l, scenes_r = define_scenes(cycle, goal_cycle_l, goal_cycle_r, CYCLES_BEFORE_GOAL)

        if not scenes_l and not scenes_r:
            print("ゴールシーンが見つからなかったため、処理を終了します。")
            continue

        print(f"{len(scenes_l)}個のゴールシーンを特定しました。データ抽出を開始")

        extracted_data_l, extracted_data_r = extract_data(rcg_file_path, scenes_l, scenes_r)

        if extracted_data_l:
                for scene_data_list in extracted_data_l.values():
                    if scene_data_list:
                        first_valid_y = np.nan
                        for frame in scene_data_list:
                            if not np.isnan(frame[1]):
                                first_valid_y = frame[1]
                                break
                        
                        if np.isnan(first_valid_y):
                            all_data_l_c.append(scene_data_list)
                        elif first_valid_y >= 20:
                            all_data_l_r.append(scene_data_list)
                        elif first_valid_y <= -20:
                            all_data_l_l.append(scene_data_list)
                        else:
                            all_data_l_c.append(scene_data_list)

        if extracted_data_r:
            for scene_data_list in extracted_data_r.values():
                if scene_data_list:
                    first_valid_y = np.nan
                    for frame in scene_data_list:
                        if not np.isnan(frame[1]):
                            first_valid_y = frame[1]
                            break

                    if np.isnan(first_valid_y):
                        all_data_r_c.append(scene_data_list) 
                    elif first_valid_y >= 20:
                        all_data_r_r.append(scene_data_list)
                    elif first_valid_y <= -20:
                        all_data_r_l.append(scene_data_list)
                    elif scene_data_list[0][1] <= -20:
                        all_data_r_r.append(scene_data_list)
                    else:
                        all_data_r_c.append(scene_data_list)

    if all_data_l_r:
        # scoring_scenes_numpy = padded_data(all_data_l, CYCLES_BEFORE_GOAL)
        scoring_scenes_numpy_r = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_l_r if scene]
        output_path_l = os.path.join(npz_folder_path, "scoring_scenes_r.pkl")
        joblib.dump(scoring_scenes_numpy_r, output_path_l)
        print(f"全右側得点シーン {len(scoring_scenes_numpy_r)}件を {output_path_l} に保存しました。")

    if all_data_l_l:
        # concession_scenes_numpy = padded_data(all_data_l, CYCLES_BEFORE_GOAL)
        scoring_scenes_numpy_l = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_l_l if scene]
        output_path_l = os.path.join(npz_folder_path, "scoring_scenes_l.pkl")
        joblib.dump(scoring_scenes_numpy_l, output_path_l)
        print(f"全左側得点シーン {len(scoring_scenes_numpy_l)}件を {output_path_l} に保存しました。")

    if all_data_l_c:
        # neutral_scenes_numpy = padded_data(all_data_l, CYCLES_BEFORE_GOAL)
        scoring_scenes_numpy_c = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_l_c if scene]
        output_path_l = os.path.join(npz_folder_path, "scoring_scenes_c.pkl")
        joblib.dump(scoring_scenes_numpy_c, output_path_l)
        print(f"全中央得点シーン {len(scoring_scenes_numpy_c)}件を {output_path_l} に保存しました。")

    if all_data_r_r:
        # concession_scenes_numpy = padded_data(all_data_r, CYCLES_BEFORE_GOAL)
        concession_scenes_numpy_r = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_r_r if scene]
        output_path_r = os.path.join(npz_folder_path, "concession_scenes_r.pkl")
        joblib.dump(concession_scenes_numpy_r, output_path_r)
        print(f"全右側失点シーン {len(concession_scenes_numpy_r)}件を {output_path_r} に保存しました。")

    if all_data_r_l:
        # scoring_scenes_numpy = padded_data(all_data_r, CYCLES_BEFORE_GOAL)
        concession_scenes_numpy_l = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_r_l if scene]
        output_path_r = os.path.join(npz_folder_path, "concession_scenes_l.pkl")
        joblib.dump(concession_scenes_numpy_l, output_path_r)
        print(f"全左側失点シーン {len(concession_scenes_numpy_l)}件を {output_path_r} に保存しました。")

    if all_data_r_c:
        # neutral_scenes_numpy = padded_data(all_data_r, CYCLES_BEFORE_GOAL)
        concession_scenes_numpy_c = [np.nan_to_num(np.array(scene), nan=0.0) for scene in all_data_r_c if scene]
        output_path_r = os.path.join(npz_folder_path, "concession_scenes_c.pkl")
        joblib.dump(concession_scenes_numpy_c, output_path_r)
        print(f"全中央失点シーン {len(concession_scenes_numpy_c)}件を {output_path_r} に保存しました。")

if __name__ == "__main__":
    main()