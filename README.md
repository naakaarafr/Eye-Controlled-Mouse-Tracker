# Eye-Controlled Mouse Tracker

A Python-based eye tracking system that allows you to control your computer's mouse cursor using eye movements. This project uses computer vision techniques to detect faces, eyes, and pupils in real-time, translating gaze direction into mouse movements.

## 🎯 Features

- **Real-time Eye Tracking**: Tracks eye movements using your webcam
- **Mouse Control**: Move the cursor by looking around the screen
- **Adaptive Camera Settings**: Automatic brightness and contrast optimization
- **Multiple Camera Support**: Automatically detects and uses available cameras
- **Diagnostic Tools**: Built-in camera testing and troubleshooting
- **Customizable Sensitivity**: Adjustable tracking sensitivity for different users
- **Calibration System**: Center calibration for accurate tracking
- **Visual Feedback**: Live preview with detection overlays

## 🛠️ Requirements

### Hardware
- **Webcam**: Built-in or external USB camera
- **Lighting**: Good lighting conditions for optimal face detection
- **Computer**: Windows, macOS, or Linux system

### Software Dependencies
```bash
pip install opencv-python numpy pyautogui
```

## 📦 Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pyautogui
   ```
3. **Ensure camera access**: Make sure no other applications are using your camera
4. **Run the application**:
   ```bash
   python eye_controlled_mouse.py
   ```

## 🚀 Usage

### Starting the Application
Run the script and the application will:
1. Automatically detect and configure your camera
2. Load face and eye detection models
3. Display a live video feed with tracking overlays
4. Begin mouse control when eyes are detected

### Controls

| Key | Function |
|-----|----------|
| **ESC** | Exit the application |
| **c** | Calibrate center position (look straight ahead and press) |
| **+/=** | Increase sensitivity |
| **-** | Decrease sensitivity |
| **t** | Test camera feed for 10 seconds |
| **b** | Decrease brightness |
| **B** | Increase brightness |
| **n** | Decrease contrast |
| **N** | Increase contrast |

### Getting Started
1. **Launch the application** and wait for the camera to initialize
2. **Position yourself** 2-3 feet from the camera with good lighting
3. **Look straight at the camera** and press **'c'** to calibrate center position
4. **Adjust sensitivity** using **+/-** keys until movement feels comfortable
5. **Start tracking** - move your eyes to control the cursor

## 🔧 Configuration

### Camera Settings
The application automatically configures:
- Resolution: 640x480
- Frame rate: 30 FPS
- Brightness optimization for face detection
- Contrast enhancement
- Auto-exposure adjustment

### Tracking Parameters
- **Sensitivity**: 0.2 to 5.0 (default: 2.0)
- **Smoothing**: 5-frame moving average
- **Detection**: Haar cascade classifiers for faces and eyes

## 🔍 Troubleshooting

### Common Issues

#### "No working camera found"
**Solutions:**
- Close other camera applications (Zoom, Skype, Teams, etc.)
- Check camera permissions in system settings
- Try unplugging and reconnecting USB cameras
- Restart the application
- Try running as administrator (Windows)

#### "No face detected"
**Solutions:**
- Improve lighting conditions
- Move closer to the camera (2-3 feet optimal)
- Adjust brightness with **'b'/'B'** keys
- Ensure face is clearly visible and unobstructed
- Try the camera test (**'t'** key) to verify feed quality

#### "Eyes not detected"
**Solutions:**
- Look directly at the camera
- Remove glasses if they cause glare
- Adjust contrast with **'n'/'N'** keys
- Ensure eyes are open and clearly visible
- Try different angles or distances

#### Jittery or inaccurate tracking
**Solutions:**
- Calibrate center position (**'c'** key)
- Adjust sensitivity with **+/-** keys
- Improve lighting consistency
- Minimize head movement
- Sit in a stable position

### System Requirements
- **Python 3.6+**
- **OpenCV 4.0+**
- **At least 4GB RAM** for smooth operation
- **USB 2.0+** for external cameras

## 🏗️ Technical Details

### Architecture
- **Face Detection**: Haar cascade classifier (`haarcascade_frontalface_default.xml`)
- **Eye Detection**: Haar cascade classifier (`haarcascade_eye.xml`)
- **Pupil Detection**: Minimum intensity point detection in eye regions
- **Gaze Calculation**: Relative pupil position within eye bounds
- **Smoothing**: Moving average filter for stability
- **Mouse Control**: PyAutoGUI for cursor movement

### Performance Optimization
- Multi-camera detection and selection
- Histogram equalization for better detection
- Adaptive brightness and contrast adjustment
- Efficient frame processing pipeline
- Error handling and recovery mechanisms

### Algorithms Used
1. **Face Detection**: Viola-Jones algorithm via Haar cascades
2. **Eye Segmentation**: Region of Interest (ROI) extraction
3. **Pupil Detection**: Minimum value location in Gaussian-blurred eye image
4. **Gaze Mapping**: Linear transformation from eye space to screen space
5. **Motion Smoothing**: Temporal filtering using moving averages

## 🔐 Privacy & Security

- **Local Processing**: All computation happens on your device
- **No Data Storage**: No images or tracking data are saved
- **No Network Access**: Application works entirely offline
- **Camera Control**: You have full control over camera access

## 🐛 Known Limitations

- **Lighting Dependent**: Requires good, consistent lighting
- **Single User**: Optimized for one person at a time
- **Head Movement**: Works best with minimal head movement
- **Accuracy**: Not suitable for precision tasks requiring pixel-perfect accuracy
- **Calibration**: Requires periodic recalibration for best results

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- **Advanced pupil detection algorithms**
- **Machine learning-based gaze estimation**
- **Multi-user support**
- **Click functionality** (blink detection)
- **Improved calibration procedures**
- **GUI configuration interface**

### How to Contribute
1. Fork the repository: https://github.com/naakaarafr/Eye-Controlled-Mouse-Tracker
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description of improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License allows you to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

## 🔗 Repository

**GitHub**: https://github.com/naakaarafr/Eye-Controlled-Mouse-Tracker

⭐ If you find this project helpful, please consider giving it a star!

## 🔗 Dependencies

- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for array operations
- **PyAutoGUI**: Cross-platform mouse control

## 💡 Tips for Best Results

1. **Lighting**: Use consistent, front-facing lighting
2. **Position**: Sit 2-3 feet from camera at eye level
3. **Background**: Use a plain background behind you
4. **Stability**: Keep your head relatively still
5. **Calibration**: Recalibrate when changing positions
6. **Practice**: Allow time to adapt to eye-controlled navigation

---

For questions, issues, or contributions, please refer to the troubleshooting section or create an issue in the project repository.
