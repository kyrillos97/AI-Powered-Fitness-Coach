# SpotiFit - AI Fitness Coach

Fitwise is a comprehensive Flutter-based fitness coaching application that utilizes on-device Artificial Intelligence to track user movements, count repetitions, and analyze exercise form in real-time. 

By leveraging the device's camera alongside deep learning models, Fitwise acts as a personal trainer in your pocket, ensuring you maintain proper posture and maximize the efficiency of your workouts.

## 📱 Features

- **Real-time Workout Tracking & AI Coach:** Uses the device camera to track body posture, automatically counting reps and providing real-time feedback on exercise form.
- **Personalized Onboarding:** 5-step setup wizard to configure height, weight, fitness goals, and activity levels.
- **Comprehensive Dashboard & Exercise Library:** Browse over 100+ exercises with detailed instructions. Track nutrition, macromolecules, and overall progress.
- **Bilingual & Responsive UI:** Full support for English and Arabic (RTL), paired with a modern Dark Theme featuring glass-morphism effects and Material Design 3.

## 🧠 AI & Machine Learning Pipeline

The intelligence behind Fitwise is powered by custom-trained Deep Learning models optimized for mobile edge-inference.

1. **Keypoint Data Collection (`/AI` directory):**
   - Python scripts utilize **MediaPipe Pose** to capture 33 3D body landmarks (x, y, z, visibility) from live camera feeds.
   - Intelligent scripts automatically define and segment repetitions based on joint angles (e.g., elbow/shoulder angles during a front raise), collecting large datasets of CSV keypoint data for different exercise states (perfect form, bent elbow, over range).
   
2. **Deep Learning Model Training:**
   - Neural networks are trained on the collected landmark sequences to classify the state of the exercise in real-time.
   - Currently, the project includes trained AI models for:
     - Biceps Curls
     - Front Shoulder Raises
     - Lateral Dumbbell Raises
     - Shrugs

3. **TFLite Mobile Deployment (`/app` directory):**
   - The trained models are converted to quantized **TensorFlow Lite (.tflite)** format for ultra-fast, low-latency execution strictly on the user's device.
   - The Flutter app feeds live camera frames through the pose detection pipeline, scaling and standardizing the coordinates before passing them to the TFLite model to deduce the current rep phase and form quality.

## 🏗️ Project Structure

The repository is divided into two primary domains:

- **`/app`**: The main Flutter mobile application codebase.
  - Contains all the UI screens, state management (Provider/Bloc), routing (GoRouter), and localization files.
  - Integrates Firebase for robust authentication and cloud datastore.
- **`/AI`**: The machine learning staging environment.
  - Contains Python scripts for MediaPipe data collection for various exercises (`collect_front_shoulder.py`, `ldr.py`, etc.).
  - Hosts the raw CSV datasets and the final exported `.tflite` models.

## 🚀 Getting Started

### 1. Flutter App Setup
Ensure you have the Flutter SDK (3.9.0+) installed.

```bash
cd app
flutter pub get

# Run on the platform of your choice
flutter run -d android
flutter run -d ios
```

### 2. AI Data Collection Setup (Optional)
If you wish to train new models or collect more keypoint data:

Ensure you have Python 3.8+ installed.
```bash
cd AI
pip install opencv-python mediapipe numpy

# Run the data collection script for a specific exercise
cd "Front Shoulder"
python collect_front_shoulder.py
```

## 🤝 Contributing
Contributions are welcome! Whether it's adding new exercise models in the AI pipeline or enhancing the Flutter application UI, feel free to open a pull request or submit an issue.

## 📄 License
This project is open-source and available under the MIT License.
