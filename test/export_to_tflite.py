# convert_yolo_to_tflite.py
try:
    from ultralytics import YOLO
    print("Ultralytics installed successfully!")
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# Load your YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n-pose.pt")

print("Exporting to TFLite format...")
# Export to TFLite
success = model.export(format="tflite")  # This will create a yolo11n-pose_float32.tflite file

if success:
    print("✅ Model successfully converted to TFLite!")
    print("✅ Output file: yolo11n-pose_float32.tflite")
else:
    print("❌ Conversion failed!")