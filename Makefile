# Makefile for Audio Software Analysis Tools
# Author: Camille Toubol-Fernandez
# Course: EP-381 Audio Engineering

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# OpenCV settings (try pkg-config first, fallback to manual paths)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || echo "-I/opt/homebrew/include/opencv4 -I/usr/local/include/opencv4")
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || echo "-L/opt/homebrew/lib -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_objdetect -lopencv_features2d -lopencv_video")

# Targets
PROTOOLS_TARGET = protools_analyzer
VIDEO_TARGET = video_analyzer

# Source files
PROTOOLS_SRC = protools_analyzer.cpp
VIDEO_SRC = video_audio_analyzer.cpp

# Default target
all: $(PROTOOLS_TARGET) check_opencv

# ProTools analyzer (no dependencies)
$(PROTOOLS_TARGET): $(PROTOOLS_SRC)
	@echo "Compiling ProTools Session Analyzer..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "ProTools analyzer compiled successfully!"

# Video analyzer (requires OpenCV)
$(VIDEO_TARGET): $(VIDEO_SRC)
	@echo "Compiling Video-to-Audio Design Analyzer..."
	@echo "Checking OpenCV installation..."
	@pkg-config --exists opencv4 || (echo "OpenCV not found! Please install OpenCV first." && echo "   macOS: brew install opencv" && echo "   Ubuntu: sudo apt install libopencv-dev" && exit 1)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $< $(OPENCV_LIBS)
	@echo "Video analyzer compiled successfully!"

# Check OpenCV installation
check_opencv:
	@echo "Checking OpenCV installation..."
	@if pkg-config --exists opencv4; then \
		echo "OpenCV found: $$(pkg-config --modversion opencv4)"; \
		echo "Include path: $$(pkg-config --cflags opencv4)"; \
		echo "Libraries: $$(pkg-config --libs opencv4)"; \
		echo ""; \
		echo "Ready to compile video analyzer with: make video"; \
	else \
		echo "OpenCV not found via pkg-config"; \
		echo ""; \
		echo "Installation instructions:"; \
		echo "   macOS:   brew install opencv"; \
		echo "   Ubuntu:  sudo apt install libopencv-dev"; \
		echo "   Windows: vcpkg install opencv"; \
		echo ""; \
		echo "Only ProTools analyzer is available without OpenCV"; \
	fi

# Individual targets
protools: $(PROTOOLS_TARGET)

video: $(VIDEO_TARGET)

# Test targets
test: test_protools

test_protools: $(PROTOOLS_TARGET)
	@echo "Testing ProTools analyzer..."
	@./$(PROTOOLS_TARGET) || echo "ProTools analyzer shows usage (expected behavior)"

test_video: $(VIDEO_TARGET)
	@echo "Testing Video analyzer..."
	@./$(VIDEO_TARGET) || echo "Video analyzer shows usage (expected behavior)"

# Clean up
clean:
	@echo "Cleaning up compiled files..."
	rm -f $(PROTOOLS_TARGET) $(VIDEO_TARGET)
	@echo "Clean complete!"

# Install OpenCV (macOS only)
install_opencv_mac:
	@echo "Installing OpenCV via Homebrew..."
	@command -v brew >/dev/null 2>&1 || (echo "Homebrew not found. Install from https://brew.sh" && exit 1)
	brew install opencv
	@echo "OpenCV installation complete!"
	@make check_opencv

# Help target
help:
	@echo "Audio Software Analysis Tools - Build System"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Compile ProTools analyzer and check OpenCV"
	@echo "  protools     - Compile ProTools Session Analyzer only"
	@echo "  video        - Compile Video-to-Audio Design Analyzer (requires OpenCV)"
	@echo "  check_opencv - Check OpenCV installation status"
	@echo "  test         - Run basic tests"
	@echo "  clean        - Remove compiled files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "macOS specific:"
	@echo "  install_opencv_mac - Install OpenCV via Homebrew"
	@echo ""
	@echo "Quick start:"
	@echo "  make              # Compile what's possible"
	@echo "  make protools     # ProTools analyzer (always works)"
	@echo "  make video        # Video analyzer (needs OpenCV)"
	@echo ""
	@echo "For more info, see README.md"

# Declare phony targets
.PHONY: all protools video check_opencv test test_protools test_video clean install_opencv_mac help
