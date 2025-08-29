import re
from bisect import bisect_left
import numpy as np
import os

rcg_file_path = "/home/arata/rcss/goal-scene-analyzer/data/logs/20250403061521-HELIOS_base_4-vs-Mars_1.rcg"
CYCLES_BEFORE_GOAL = 50  
OUTPUT_FILENAME_L = "final_goal_scenes_data_l.npz"
OUTPUT_FILENAME_R = "final_goal_scenes_data_r.npz"
npz_folder_path = "/home/arata/rcss/goal-scene-analyzer/data/results"

cycle = []
goal_cycle_l = []
goal_cycle_r = []

with open(rcg_file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"(\d+)\s+play_on", line)
        if match:
            cycle.append(int(match.group(1)))


        goal_match_l = re.search(r"(\d+)\s+goal_l", line)
        if goal_match_l:
            goal_cycle_l.append(int(goal_match_l.group(1)))

        goal_match_r = re.search(r"(\d+)\s+goal_r", line)
        if goal_match_r:
            goal_cycle_r.append(int(goal_match_r.group(1)))

def define_scenes(cycle, goal_cycle_l, goal_cycle_r, duration = 50):
    scenes_l = []
    scenes_r = []
    nearest_play_on_l = []
    nearest_play_on_r = []

    for goal in goal_cycle_l:
        idx = bisect_left(cycle, goal)
        if idx < len(cycle):
            nearest_play_on_l.append(cycle[idx-1])
        else:
            nearest_play_on_l.append(0)

    for goal in goal_cycle_r:
        idx = bisect_left(cycle, goal)
        if idx < len(cycle):
            nearest_play_on_r.append(cycle[idx-1])
        else:
            nearest_play_on_r.append(0)

    for i,(goal, playon) in enumerate(zip(goal_cycle_l, nearest_play_on_l)):
        start_cycle = max(playon, goal - duration)
        scenes_l.append({
            "name": f"goal_{i+1}_at_cycle_{goal}",
            "start_cycle": start_cycle,
            "end_cycle": goal
        })

    for i,(goal, playon) in enumerate(zip(goal_cycle_r, nearest_play_on_r)):
        start_cycle = max(playon, goal - duration)
        scenes_r.append({
            "name": f"goal_{i+1}_at_cycle_{goal}",
            "start_cycle": start_cycle,
            "end_cycle": goal
        })

    return scenes_l, scenes_r

def extract_data(file_path, scenes_l, scenes_r):
    results_l = {scene['name']: [] for scene in scenes_l}
    results_r = {scene['name']: [] for scene in scenes_r}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith('(show'):
                continue

            match_cycle = re.search(r'\(show(\d+)', line)
            if not match_cycle:
                continue
            current_cycle = int(match_cycle.group(1))

            for scene in scenes_l:
                if scene['start_cycle'] <= current_cycle <= scene['end_cycle']:
                    match_ball = re.search(r'\(\(b\) ([-\d\.]+) ([-\d\.]+)', line)
                    ball_x, ball_y = (float(p) for p in match_ball.groups()) if match_ball else (np.nan, np.nan)

                    raw_data = [ball_x, ball_y]
                    player_positions = [np.nan] * 44

                    players_found = re.findall(r'\(\(([lr])\s+(\d+)\)[^)]*?\s([-\d\.]+)\s+([-\d\.]+)', line)
                    for team, unum, x, y in players_found:
                        unum = int(unum)
                        x, y = float(x), float(y)
                        offset = 0 if team == 'l' else 22
                        index = offset + (unum - 1) * 2
                        if index < len(player_positions):
                            player_positions[index] = x
                            player_positions[index + 1] = y 

                    raw_data.extend(player_positions)

                    results_l[scene['name']].append(raw_data)

            
            for scene in scenes_r:
                if scene['start_cycle'] <= current_cycle <= scene['end_cycle']:
                    match_ball = re.search(r'\(\(b\) ([-\d\.]+) ([-\d\.]+)', line)
                    ball_x, ball_y = (float(p) for p in match_ball.groups()) if match_ball else (np.nan, np.nan)

                    raw_data = [ball_x, ball_y]
                    player_positions = [np.nan] * 44

                    players_found = re.findall(r'\(\(([lr])\s+(\d+)\)[^)]*?\s([-\d\.]+)\s+([-\d\.]+)', line)
                    for team, unum, x, y in players_found:
                        unum = int(unum)
                        x, y = float(x), float(y)
                        offset = 0 if team == 'l' else 22
                        index = offset + (unum - 1) * 2
                        if index < len(player_positions):
                            player_positions[index] = x
                            player_positions[index + 1] = y 

                    raw_data.extend(player_positions)

                    results_r[scene['name']].append(raw_data)

    return results_l, results_r

def main():
    print(f"処理を開始: {rcg_file_path}")
    
    scenes_l, scenes_r = define_scenes(cycle, goal_cycle_l, goal_cycle_r, CYCLES_BEFORE_GOAL)

    if not scenes_l and not scenes_r:
        print("ゴールシーンが見つからなかったため、処理を終了します。")
        return

    print(f"{len(scenes_l)}個のゴールシーンを特定しました。データ抽出を開始")

    extracted_data_l, extracted_data_r = extract_data(rcg_file_path, scenes_l, scenes_r)

    for name, data_list in extracted_data_l.items():
        extracted_data_l[name] = np.array(data_list)
    for name, data_list in extracted_data_r.items():
        extracted_data_r[name] = np.array(data_list)

    if extracted_data_l:
        output_path_l = os.path.join(npz_folder_path, OUTPUT_FILENAME_L)
        np.savez_compressed(output_path_l, **extracted_data_l)
        print(f"左チームのゴールデータを {output_path_l} に保存しました。")

    if extracted_data_r:
        output_path_r = os.path.join(npz_folder_path, OUTPUT_FILENAME_R)
        np.savez_compressed(output_path_r, **extracted_data_r)
        print(f"右チームのゴールデータを {output_path_r} に保存しました。")

if __name__ == "__main__":
    main()