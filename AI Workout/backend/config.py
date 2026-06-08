import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "workout_assets")

CONFIG = {
    # Server configuration
    "HOST": "0.0.0.0",
    "PORT": 8000,
    "CORS_ORIGINS": ["*"],
    
    # Camera index
    "CAMERA_INDEX": 0,
    "FLIP_HORIZONTAL": True,

    # Face verification
    "FACE_MATCH_THRESHOLD": 0.5,

    # Asset paths
    "MODELS": {
        "bicep_curl": {
            "model_path": os.path.join(ASSETS_DIR, "bicep_curl", "bicep_coach_3class.tflite"),
        },
        "front_shoulder": {
            "model_path": os.path.join(ASSETS_DIR, "front_shoulder", "st_gcvae_front_shoulder.pt"),
            "config_path": os.path.join(ASSETS_DIR, "front_shoulder", "vae_config_front_shoulder.json"),
        },
        "shrug": {
            "model_path": os.path.join(ASSETS_DIR, "shrugs", "shrug_classifier_fp16.tflite"),  # alias for engine
            "classifier_path": os.path.join(ASSETS_DIR, "shrugs", "shrug_classifier_fp16.tflite"),
            "vae_path": os.path.join(ASSETS_DIR, "shrugs", "shrug_vae_encoder.tflite"),
            "labels_path": os.path.join(ASSETS_DIR, "shrugs", "shrug_label_classes.txt"),
            "stats_path": os.path.join(ASSETS_DIR, "shrugs", "best_shrug_model_feature_stats.npz"),
        },
        "side_shoulder": {
            "model_path": os.path.join(BASE_DIR, "..", "side_shoulder", "st_gcvae_side_shoulder.weights.h5"),
            "config_path": os.path.join(BASE_DIR, "..", "side_shoulder", "vae_config_side_shoulder.json"),
        },
        "squat": {
            "model_path": os.path.join(ASSETS_DIR, "squat", "squat_classifier_float32.tflite"),
            "labels_path": os.path.join(ASSETS_DIR, "squat", "label_classes.txt"),
        }
    }
}
