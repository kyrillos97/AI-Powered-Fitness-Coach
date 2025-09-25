import os
import json
from config import DATA_DIR

def save_plan_to_json(user_id: str, diet_plan: dict, workout_plan: dict):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    file_path = os.path.join(DATA_DIR, f"{user_id}_plans.json")
    
    data_to_save = {
        "diet_plan": diet_plan,
        "workout_plan": workout_plan
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    
    print(f"Plans for user {user_id} saved successfully to {file_path}")