import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # AI Workout
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
ASSETS_DIR = os.path.join(BACKEND_DIR, "workout_assets")

os.makedirs(ASSETS_DIR, exist_ok=True)

models_to_copy = {
    "bicep_curl": [
        "Workout-Bie/bicep_coach_3class.tflite"
    ],
    "front_shoulder": [
        "Front_Shoulder/st_gcvae_front_shoulder.pt",
        "Front_Shoulder/vae_config_front_shoulder.json"
    ],
    "shrugs": [
        "shrugs/shrug_classifier_fp16.tflite",
        "shrugs/shrug_vae_encoder.tflite",
        "shrugs/shrug_label_classes.txt",
        "shrugs/best_shrug_model_feature_stats.npz"
    ],
    "side_shoulder": [
        "side_shoulder/full_pipeline_v4.tflite",
        "side_shoulder/realtime_config_v4.json"
    ],
    "squat": [
        "squat/squat_classifier_float32.tflite",
        "squat/label_classes.txt"
    ]
}

for workout, files in models_to_copy.items():
    workout_dir = os.path.join(ASSETS_DIR, workout)
    os.makedirs(workout_dir, exist_ok=True)
    for file_path in files:
        src = os.path.join(BASE_DIR, file_path)
        if os.path.exists(src):
            dst = os.path.join(workout_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
        else:
            print(f"Warning: Missing file {src}")

print("Assets setup completed.")
