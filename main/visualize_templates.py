import joblib

def main():
    model = joblib.load('/home/arata/rcss/goal-scene-analyzer/data/processed/scoring_model.pkl')
    try:
        templates = model.cluster_centers_
        print(f"テンプレート数: {len(templates)}")
    except FileNotFoundError:
        print("モデルファイルが見つかりません")
        return

if __name__ == "__main__":
    main()