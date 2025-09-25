import json
import os

def split_diet_workout(input_file, output_dir="."):
    
    # Load original JSON
    with open(input_file, "r") as file:
        data = json.load(file)

    # Extract diet and workout data
    diet_data = data["diet_plan"]       # {"diet": {...}}
    workout_data = data["workout_plan"] # {"workout": {...}}

    # 🔹 Process workout_data: push rest_minutes into reps list
    for day, exercises in workout_data["workout"].items():
        for ex_name, details in exercises["exercises"].items():
            if "rest_minutes" in details:
                details["reps"].append(details["rest_minutes"])
                del details["rest_minutes"]

    # Define output file paths
    diet_file = os.path.join(output_dir, "diet_plan.json")
    workout_file = os.path.join(output_dir, "workout_plan.json")

    # Save results
    with open(diet_file, "w") as f:
        json.dump(diet_data, f, indent=4)

    with open(workout_file, "w") as f:
        json.dump(workout_data, f, indent=4)

    return diet_file, workout_file


# 🔹 Example usage:
diet_path, workout_path = split_diet_workout(
    r"D:\final_project\chatbot\user_data\9f438cac-75e0-4f70-8a56-ae4752b32e4b_plans.json",
    r"D:\final_project\chatbot\clean_json"
)

print("Diet file saved at:", diet_path)
print("Workout file saved at:", workout_path)
