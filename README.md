# Audio Software Analysis Tools

C++ tools I built for EP-381 Audio Engineering coursework. These analyze Pro Tools sessions and video content to generate audio design recommendations.

## Tools Overview

### 1. ProTools Session Analyzer (`protools_analyzer.cpp`)
**Status: In Review**

Parses Pro Tools .ptx session files and generates analysis reports. I wrote this to help validate my EP-381 project requirements.

**Features:**
- Binary file parsing of Pro Tools session format
- Track layout analysis and classification  
- Audio file detection and cataloging
- Sample rate and configuration detection
- Automated recommendations and warnings

**Usage:**
```bash
# Compile (no dependencies needed)
g++ -std=c++17 -o protools_analyzer protools_analyzer.cpp

# Run
./protools_analyzer "your_session.ptx"
```

### 2. Video-to-Audio Design Analyzer (`video_audio_analyzer.cpp`)
**Status: Needs OpenCV**

Uses computer vision to analyze video content and automatically generate audio design recommendations. This was the attempt at bridging visual analysis with audio engineering.

**Features:**
- Computer vision analysis (motion detection, scene changes, color analysis)
- HOG people detection
- Visual-to-audio characteristic mapping
- Audio engineering recommendations
- Frame-by-frame analysis reports

## Setup Instructions

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- OpenCV 4.x (for video analyzer only)

### macOS Setup

#### Install OpenCV via Homebrew:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV
brew install opencv

# Verify installation
pkg-config --cflags --libs opencv4
```

#### Compile Video Analyzer:
```bash
# Method 1: Using pkg-config (recommended)
g++ -std=c++17 $(pkg-config --cflags opencv4) -o video_analyzer video_audio_analyzer.cpp $(pkg-config --libs opencv4)

# Method 2: Manual paths (if pkg-config doesn't work)
g++ -std=c++17 -I/opt/homebrew/include/opencv4 -o video_analyzer video_audio_analyzer.cpp -L/opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_objdetect -lopencv_features2d -lopencv_video
```

### Ubuntu/Debian Setup
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev

# Compile
g++ -std=c++17 $(pkg-config --cflags opencv4) -o video_analyzer video_audio_analyzer.cpp $(pkg-config --libs opencv4)
```

### Windows Setup
```bash
# Using vcpkg (recommended)
vcpkg install opencv

# Or download OpenCV from opencv.org and set paths manually
```

## Usage Examples

### ProTools Analyzer
```bash
./protools_analyzer "EP381 P2 MyProject.ptx"
```

### Video Analyzer
```bash
# Basic usage
./video_analyzer input_video.mp4

# With custom output report
./video_analyzer Journey.mov detailed_analysis.txt

# Supported formats: MP4, MOV, AVI, MKV, WMV
```

## Sample Outputs

### ProTools Analyzer Output:
```
=== PRO TOOLS SESSION ANALYZER (C++) ===
High-performance binary analysis of Pro Tools session files

============================================================
PRO TOOLS SESSION ANALYSIS REPORT
============================================================
Session File: EP381 P2 Toubol.ptx
File Size: 2048576 bytes
Modified: Sun Aug 18 14:30:25 2024

SESSION CONFIGURATION:
------------------------------
Sample Rate: 48000 Hz (Video Standard)
Bit Depth: 24-bit
Track Count: 6
Audio Files: 15

TRACK LAYOUT:
--------------------
   1. Journey (Stereo) (Video, Stereo)
   2. Backgrounds (Stereo) (Audio, Stereo)
   3. Ambients (Stereo) (Audio, Stereo)
   4. KyActnMmnt(S) (Stereo) (Audio, Stereo)
   5. Transitions(Str) (Stereo) (Audio, Stereo)
   6. Transitions(Mn) (Audio, Mono)
```

### Video Analyzer Output:
```
================================================================
           VIDEO-TO-AUDIO DESIGN ANALYZER v1.0
================================================================
An advanced C++ application combining computer vision and audio
engineering to generate intelligent audio design recommendations
from video content analysis.

Video Analysis Initialized:
  Resolution: 1920x1080
  FPS: 24.0
  Total Frames: 2400
  Duration: 100.00 seconds

Starting comprehensive video analysis...
Progress: 100.0% (2400/2400 frames)
Analysis complete!

Analysis completed successfully in 45 seconds.
Detailed report generated: audio_design_analysis.txt
```

## Technical Specifications

### ProTools Analyzer
- **Language:** C++17
- **Dependencies:** Standard Library only
- **Performance:** Analyzes large sessions (<1MB) in milliseconds
- **Compatibility:** Cross-platform (Windows, macOS, Linux)

### Video Analyzer
- **Language:** C++17 + OpenCV 4.x
- **Performance:** Processes HD video at ~10-15 FPS
- **Memory Usage:** Optimized streaming (scales with video length)
- **Algorithms:** Lucas-Kanade optical flow, HOG detection, Canny edges
- **AI/ML:** Pre-trained people detection models

## Academic Context

These tools were developed for EP-381 Audio Engineering coursework. The ProTools analyzer helped me validate my session organization, and the video analyzer was an experimental project exploring the connection between visual content and audio design.

Skills demonstrated:
- C++ programming with modern features
- Computer vision and image processing
- Audio engineering domain knowledge
- Cross-platform development
- Binary file parsing

## License

Academic project - Camille Toubol-Fernandez, EP-381 Audio Engineering

## Future Ideas

- Real-time video processing
