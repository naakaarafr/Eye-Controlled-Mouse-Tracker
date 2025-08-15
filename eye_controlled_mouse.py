import cv2
import numpy as np
import pyautogui
from collections import deque
import time

class DiagnosticEyeTracker:
    def __init__(self):
        print("🔍 Starting Eye Tracker with Diagnostics...")
        
        # Screen setup
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False
        print(f"📺 Screen size: {self.screen_w}x{self.screen_h}")
        
        # Try multiple camera indices
        self.cap = None
        self.camera_index = None
        
        print("📹 Testing camera access...")
        for i in range(5):  # Try camera indices 0-4
            print(f"   Trying camera index {i}...")
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Camera {i} working! Frame shape: {test_frame.shape}")
                    self.cap = test_cap
                    self.camera_index = i
                    break
                else:
                    print(f"❌ Camera {i} opened but no frame")
                    test_cap.release()
            else:
                print(f"❌ Camera {i} failed to open")
                test_cap.release()
        
        if self.cap is None:
            print("❌ No working camera found!")
            print("💡 Troubleshooting tips:")
            print("   1. Make sure no other app is using the camera")
            print("   2. Check camera permissions")
            print("   3. Try unplugging and reconnecting USB camera")
            print("   4. Restart the program")
            return
        
        # Configure camera with brightness optimization
        print("⚙️ Configuring camera...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Brightness and contrast settings for better face detection
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)  # Increase brightness (0-1)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.6)    # Increase contrast (0-1)
        self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)  # Moderate saturation
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)     # Auto exposure with bias
        self.cap.set(cv2.CAP_PROP_GAIN, 0.3)        # Some gain for low light
        
        print("🔆 Applied brightness optimizations for face detection")
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"📷 Camera settings: {int(actual_width)}x{int(actual_height)} @ {actual_fps}fps")
        
        # Test Haar cascades
        print("🧠 Loading face detection models...")
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                print("❌ Haar cascades failed to load")
                return
            else:
                print("✅ Face detection models loaded successfully")
                
        except Exception as e:
            print(f"❌ Error loading cascades: {e}")
            return
        
        # Initialize tracking variables
        self.gaze_history = deque(maxlen=5)
        self.screen_history = deque(maxlen=3)
        self.center_gaze_x = 0.5
        self.center_gaze_y = 0.5
        self.sensitivity = 2.0
        self.last_mouse_x = self.screen_w // 2
        self.last_mouse_y = self.screen_h // 2
        
        # Brightness control
        self.brightness_adjustment = 0
        self.contrast_adjustment = 0
        
        print("\n🎉 Eye Tracker Ready!")
        print("👁️  Look around to move cursor")
        print("🎯 Press 'c' to recalibrate center")
        print("⚙️  Press '+' or '-' to adjust sensitivity")
        print("🔧 Press 't' to test camera feed")
        print("🔆 Press 'b'/'B' to decrease/increase brightness")
        print("🌓 Press 'n'/'N' to decrease/increase contrast")  
        print("❌ Press ESC to exit")

    def test_camera_feed(self):
        """Test camera feed for 10 seconds"""
        if self.cap is None:
            print("❌ No camera available for testing")
            return
        
        print("📹 Testing camera feed for 10 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame_count += 1
                # Flip and show frame
                frame = cv2.flip(frame, 1)
                
                # Add test info
                cv2.putText(frame, f"Camera Test - Frame #{frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {time.time() - start_time:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press any key to stop test", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Camera Test", frame)
                
                if cv2.waitKey(1) & 0xFF != 255:  # Any key pressed
                    break
            else:
                print(f"❌ Failed to read frame at {time.time() - start_time:.1f}s")
                break
        
        cv2.destroyWindow("Camera Test")
        print(f"📊 Camera test complete. Captured {frame_count} frames in {time.time() - start_time:.1f}s")

    def enhance_frame_for_detection(self, frame):
        """Enhance frame brightness and contrast for better face detection"""
        # Convert to float for processing
        enhanced = frame.astype(np.float32)
        
        # Apply brightness adjustment (-100 to +100)
        enhanced = enhanced + self.brightness_adjustment
        
        # Apply contrast adjustment (0.5 to 3.0, where 1.0 is no change)
        contrast_factor = 1.0 + (self.contrast_adjustment / 100.0)
        enhanced = enhanced * contrast_factor
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)

    def detect_pupils_simple(self, eye_roi):
        """Simple pupil detection"""
        if eye_roi.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Find darkest point (pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        pupil_x, pupil_y = min_loc
        
        # Basic validation
        h, w = gray.shape
        if 0.1 * w < pupil_x < 0.9 * w and 0.1 * h < pupil_y < 0.9 * h:
            return pupil_x, pupil_y
        
        return None

    def calculate_gaze_direction(self, pupil_x, pupil_y, eye_w, eye_h):
        """Calculate gaze direction"""
        gaze_x = pupil_x / eye_w
        gaze_y = pupil_y / eye_h
        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    def smooth_gaze(self, gaze_x, gaze_y):
        """Apply smoothing"""
        self.gaze_history.append((gaze_x, gaze_y))
        
        if len(self.gaze_history) < 2:
            return gaze_x, gaze_y
        
        recent_gazes = list(self.gaze_history)
        avg_x = sum(g[0] for g in recent_gazes) / len(recent_gazes)
        avg_y = sum(g[1] for g in recent_gazes) / len(recent_gazes)
        
        return avg_x, avg_y

    def gaze_to_screen(self, gaze_x, gaze_y):
        """Convert gaze to screen coordinates"""
        delta_x = (gaze_x - self.center_gaze_x) * self.sensitivity
        delta_y = (gaze_y - self.center_gaze_y) * self.sensitivity
        
        screen_x = self.screen_w // 2 + delta_x * (self.screen_w // 2)
        screen_y = self.screen_h // 2 + delta_y * (self.screen_h // 2)
        
        return max(0, min(self.screen_w - 1, int(screen_x))), max(0, min(self.screen_h - 1, int(screen_y)))
        """Simple pupil detection"""
        if eye_roi.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Find darkest point (pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        pupil_x, pupil_y = min_loc
        
        # Basic validation
        h, w = gray.shape
        if 0.1 * w < pupil_x < 0.9 * w and 0.1 * h < pupil_y < 0.9 * h:
            return pupil_x, pupil_y
        
        return None

    def calculate_gaze_direction(self, pupil_x, pupil_y, eye_w, eye_h):
        """Calculate gaze direction"""
        gaze_x = pupil_x / eye_w
        gaze_y = pupil_y / eye_h
        return max(0.0, min(1.0, gaze_x)), max(0.0, min(1.0, gaze_y))

    def smooth_gaze(self, gaze_x, gaze_y):
        """Apply smoothing"""
        self.gaze_history.append((gaze_x, gaze_y))
        
        if len(self.gaze_history) < 2:
            return gaze_x, gaze_y
        
        recent_gazes = list(self.gaze_history)
        avg_x = sum(g[0] for g in recent_gazes) / len(recent_gazes)
        avg_y = sum(g[1] for g in recent_gazes) / len(recent_gazes)
        
        return avg_x, avg_y

    def gaze_to_screen(self, gaze_x, gaze_y):
        """Convert gaze to screen coordinates"""
        delta_x = (gaze_x - self.center_gaze_x) * self.sensitivity
        delta_y = (gaze_y - self.center_gaze_y) * self.sensitivity
        
        screen_x = self.screen_w // 2 + delta_x * (self.screen_w // 2)
        screen_y = self.screen_h // 2 + delta_y * (self.screen_h // 2)
        
        return max(0, min(self.screen_w - 1, int(screen_x))), max(0, min(self.screen_h - 1, int(screen_y)))

    def run(self):
        """Main tracking loop"""
        if self.cap is None:
            print("❌ Cannot start - no camera available")
            return
        
        frame_count = 0
        last_time = time.time()
        consecutive_failures = 0
        
        print("\n🚀 Starting eye tracking loop...")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"⚠️  Frame read failed #{consecutive_failures}")
                
                if consecutive_failures > 10:
                    print("❌ Too many consecutive frame failures - stopping")
                    break
                
                time.sleep(0.1)  # Wait before retry
                continue
            
            consecutive_failures = 0  # Reset on successful read
            frame_count += 1
            
            # Enhance frame for better detection
            enhanced_frame = self.enhance_frame_for_detection(frame)
            
            # Flip for mirror effect
            enhanced_frame = cv2.flip(enhanced_frame, 1)
            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            
            # Additional histogram equalization for better face detection
            gray = cv2.equalizeHist(gray)
            
            # Add frame info and brightness settings
            cv2.putText(enhanced_frame, f"Frame #{frame_count} | Camera {self.camera_index}", 
                       (10, enhanced_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(enhanced_frame, f"Brightness: {self.brightness_adjustment:+d} | Contrast: {self.contrast_adjustment:+d}", 
                       (10, enhanced_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Detect faces with more sensitive settings
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,     # More sensitive scaling
                minNeighbors=3,      # Reduced from 5 for better detection
                minSize=(80, 80),    # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                cv2.putText(enhanced_frame, "No face detected - adjust brightness with 'b'/'B'", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(enhanced_frame, "Try better lighting or move closer", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(enhanced_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(enhanced_frame, "Face detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = enhanced_frame[y:y + h, x:x + w]
                
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.05,    # More sensitive for eye detection  
                    minNeighbors=3,      # Reduced for better detection
                    minSize=(15, 15),    # Smaller minimum eye size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(eyes) == 0:
                    cv2.putText(enhanced_frame, "Eyes not detected - look straight at camera", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])
                    valid_gazes = []
                    
                    for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        
                        eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                        pupil_pos = self.detect_pupils_simple(eye_roi)
                        
                        if pupil_pos:
                            px, py = pupil_pos
                            gaze_x, gaze_y = self.calculate_gaze_direction(px, py, ew, eh)
                            valid_gazes.append((gaze_x, gaze_y))
                            
                            abs_px = x + ex + px
                            abs_py = y + ey + py
                            cv2.circle(enhanced_frame, (abs_px, abs_py), 3, (0, 0, 255), -1)
                            cv2.putText(enhanced_frame, f"Pupil", (abs_px + 5, abs_py), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    if valid_gazes:
                        avg_gaze_x = sum(g[0] for g in valid_gazes) / len(valid_gazes)
                        avg_gaze_y = sum(g[1] for g in valid_gazes) / len(valid_gazes)
                        
                        smooth_gaze_x, smooth_gaze_y = self.smooth_gaze(avg_gaze_x, avg_gaze_y)
                        target_x, target_y = self.gaze_to_screen(smooth_gaze_x, smooth_gaze_y)
                        
                        try:
                            pyautogui.moveTo(target_x, target_y, duration=0.02)
                        except:
                            pass
                        
                        cv2.putText(enhanced_frame, f"Tracking: ({smooth_gaze_x:.2f}, {smooth_gaze_y:.2f})", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(enhanced_frame, f"Mouse: ({target_x}, {target_y})", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                break  # Only process first face
            
            cv2.imshow("Eye Tracker - Enhanced Detection", enhanced_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('t'):  # Test camera
                self.test_camera_feed()
            elif key == ord('c'):  # Calibrate
                if len(self.gaze_history) > 0:
                    recent_gaze = self.gaze_history[-1]
                    self.center_gaze_x, self.center_gaze_y = recent_gaze
                    print(f"✅ Calibrated center: ({self.center_gaze_x:.3f}, {self.center_gaze_y:.3f})")
            elif key == ord('+') or key == ord('='):
                self.sensitivity = min(5.0, self.sensitivity + 0.2)
                print(f"⚙️  Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('-'):
                self.sensitivity = max(0.2, self.sensitivity - 0.2)
                print(f"⚙️  Sensitivity: {self.sensitivity:.1f}")
            elif key == ord('b'):  # Decrease brightness
                self.brightness_adjustment = max(-100, self.brightness_adjustment - 10)
                print(f"🔆 Brightness: {self.brightness_adjustment:+d}")
            elif key == ord('B'):  # Increase brightness
                self.brightness_adjustment = min(100, self.brightness_adjustment + 10)
                print(f"🔆 Brightness: {self.brightness_adjustment:+d}")
            elif key == ord('n'):  # Decrease contrast
                self.contrast_adjustment = max(-50, self.contrast_adjustment - 10)
                print(f"🌓 Contrast: {self.contrast_adjustment:+d}")
            elif key == ord('N'):  # Increase contrast
                self.contrast_adjustment = min(100, self.contrast_adjustment + 10)
                print(f"🌓 Contrast: {self.contrast_adjustment:+d}")
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Eye tracker stopped")

# Run with full error handling
if __name__ == "__main__":
    try:
        tracker = DiagnosticEyeTracker()
        if hasattr(tracker, 'cap') and tracker.cap is not None:
            tracker.run()
        else:
            print("\n❌ Could not initialize tracker")
            print("💡 Common solutions:")
            print("   1. Close other camera apps (Zoom, Skype, etc.)")
            print("   2. Check camera permissions in system settings")
            print("   3. Try a different USB port")
            print("   4. Restart your computer")
            
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()