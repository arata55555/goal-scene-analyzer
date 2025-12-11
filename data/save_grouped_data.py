import re
from bisect import bisect_left
import numpy as np
import os
import gzip
import joblib

rcg_folder_path = "/mnt/data1/miyazaki/work/autogame/data/goal-scene-analyzer_heliosbase_vs_itandroids_change_1/"
CYCLES_BEFORE_GOAL = 50  
OUTPUT_FILENAME_L = "final_goal_scenes_data_l.npz"
OUTPUT_FILENAME_R = "final_goal_scenes_data_r.npz"
npz_folder_path = "/home/arata/rcss/work/goal-scene-analyzer/data/finish/12-10"

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
            "end_cycle": goal_l-1,
            "goal_cycle": goal_l
        })

    for i,(goal_r, playon) in enumerate(zip(goal_cycle_r, nearest_play_on_r)):
        start_cycle = max(playon, goal_r - duration)
        scenes_r.append({
            "name": f"goal_{i+1}_at_cycle_{goal_r-1}",
            "start_cycle": start_cycle,
            "end_cycle": goal_r-1,
            "goal_cycle": goal_r
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

def save_grouped_data(data_list_with_cycle, filename):
    grouped = {}
    for goal_cycle, scene_data in data_list_with_cycle:
        range_start = (goal_cycle // 1000) * 1000
        if range_start not in grouped:
            grouped[range_start] = []
        grouped[range_start].append(scene_data)

    for range_start, scenes in grouped.items():
        save_dir = os.path.join(npz_folder_path, str(range_start))
        os.makedirs(save_dir, exist_ok=True)
        
        scoring_scenes_numpy = [np.nan_to_num(np.array(scene), nan=0.0) for scene in scenes]
        
        output_path = os.path.join(save_dir, filename)
        joblib.dump(scoring_scenes_numpy, output_path)
        print(f"サイクル{range_start}番台: {len(scoring_scenes_numpy)}件を {output_path} に保存しました。")

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
            for scene_info in scenes_l: 
                name = scene_info['name']
                goal_cycle = scene_info['goal_cycle']
                scene_data_list = extracted_data_l.get(name)

                if scene_data_list:
                    first_valid_y = np.nan
                    for frame in scene_data_list:
                        if not np.isnan(frame[1]):
                            first_valid_y = frame[1]
                            break
                    
                    if np.isnan(first_valid_y):
                        all_data_l_c.append((goal_cycle, scene_data_list))
                    elif first_valid_y >= 20:
                        all_data_l_r.append((goal_cycle, scene_data_list))
                    elif first_valid_y <= -20:
                        all_data_l_l.append((goal_cycle, scene_data_list))
                    else:
                        all_data_l_c.append((goal_cycle, scene_data_list))

        if extracted_data_r:
            for scene_info in scenes_r:
                name = scene_info['name']
                goal_cycle = scene_info['goal_cycle']
                scene_data_list = extracted_data_r.get(name)

                if scene_data_list:
                    first_valid_y = np.nan
                    for frame in scene_data_list:
                        if not np.isnan(frame[1]):
                            first_valid_y = frame[1]
                            break

                    if np.isnan(first_valid_y):
                        all_data_r_c.append((goal_cycle, scene_data_list))
                    elif first_valid_y >= 20:
                        all_data_r_r.append((goal_cycle, scene_data_list))
                    elif first_valid_y <= -20:
                        all_data_r_l.append((goal_cycle, scene_data_list))
                    elif scene_data_list[0][1] <= -20:
                        all_data_r_r.append((goal_cycle, scene_data_list))
                    else:
                        all_data_r_c.append((goal_cycle, scene_data_list))

    if all_data_l_r:
        save_grouped_data(all_data_l_r, "scoring_scenes_r.pkl")
    if all_data_l_l:
        save_grouped_data(all_data_l_l, "scoring_scenes_l.pkl")
    if all_data_l_c:
        save_grouped_data(all_data_l_c, "scoring_scenes_c.pkl")

    if all_data_r_r:
        save_grouped_data(all_data_r_r, "concession_scenes_r.pkl")
    if all_data_r_l:
        save_grouped_data(all_data_r_l, "concession_scenes_l.pkl")
    if all_data_r_c:
        save_grouped_data(all_data_r_c, "concession_scenes_c.pkl")

if __name__ == "__main__":
    main()