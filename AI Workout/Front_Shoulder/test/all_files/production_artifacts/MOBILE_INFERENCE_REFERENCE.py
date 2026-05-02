
# =============================================================================
# MOBILE INFERENCE REFERENCE CODE
# =============================================================================
# This file documents the exact inference pipeline for mobile integration.
# Adapt to your platform (Android/Kotlin, iOS/Swift, Flutter/Dart).
#
# Required files:
#   - vae_model.onnx (or .tflite)
#   - conv1d_classifier.onnx (or .tflite)
#   - mobile_config.json
#   - gb_classifier.onnx (fallback)
# =============================================================================

"""
INFERENCE LOOP (Pseudocode):

1. EVERY FRAME (30fps):
   a. Get 33 MediaPipe landmarks
   b. Extract 4 angles: arm_elevation, elbow_angle, torso_lean, shoulder_diff
   c. Compute 4 velocities: diff from previous frame
   d. Scale 8 features using vae_feature_means / vae_feature_stds
   e. Run VAE encoder → get latent (mu) [12 dims]
   f. Compute reconstruction error = MSE(input, reconstruction)
   g. Feed elevation + latent into TSMRepDetector.update()

2. WHEN REP DETECTED:
   a. Check reconstruction error < anomaly_threshold (reject glitches)
   b. Extract the rep window from buffer (start_frame to end_frame)
   c. IF using Conv1D:
      - Build window: [raw_8_features, vae_latents] = 20 channels
      - Pad/truncate to 64 frames
      - Normalize with conv1d_channel_means / conv1d_channel_stds
      - Run Conv1D classifier → logits → softmax → class + confidence
   d. IF using GradientBoosting:
      - Extract 20+ summary features from the rep window
      - Normalize with gb_feature_means / gb_feature_stds
      - Run GB classifier → class + confidence
   e. Display: "Rep N: <class> (confidence%)"

3. TSM REP DETECTOR STATE MACHINE:
   States: IDLE → RISING → TRACKING_PEAK → (rep detected) → RISING
   
   IDLE:
     Set trough = current elevation, go to RISING
   
   RISING:
     If elevation < trough: update trough
     If elevation > trough + threshold (15°): go to TRACKING_PEAK
   
   TRACKING_PEAK:
     If elevation > peak: update peak
     If elevation < peak - threshold (15°): → REP CANDIDATE
       Check duration in [25, 160] frames
       Check VAE latent similarity to previous peaks
       If valid → REP DETECTED, reset to RISING
     If duration > max_period: timeout, go to IDLE

4. LATENCY BUDGET (per frame):
   MediaPipe:     ~15ms
   Angle calc:     ~1ms
   VAE forward:    ~2ms
   TSM update:     ~1ms
   ─────────────────────
   Total per frame: ~19ms (52fps capable)
   
   Conv1D (per rep only): ~5ms
   Total per rep: ~5ms additional
"""
