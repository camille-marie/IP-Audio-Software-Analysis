#!/bin/bash
# Setup script for Audio Software Analysis Tools
# Author: Camille Toubol-Fernandez

echo "Audio Software Analysis Tools - Setup"
echo "======================================"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Homebrew found"
    fi
    
    # Install pkg-config if not present
    if ! command -v pkg-config &> /dev/null; then
        echo "Installing pkg-config..."
        brew install pkg-config
    else
        echo "pkg-config found"
    fi
    
    # Install OpenCV
    echo "Installing OpenCV..."
    if brew list opencv &> /dev/null; then
        echo "OpenCV already installed"
    else
        brew install opencv
        echo "OpenCV installed successfully"
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected"
    
    # Update package list
    echo "Updating package list..."
    sudo apt update
    
    # Install build essentials
    echo "Installing build essentials..."
    sudo apt install -y build-essential pkg-config
    
    # Install OpenCV
    echo "Installing OpenCV..."
    sudo apt install -y libopencv-dev
    
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install OpenCV manually and try again."
    exit 1
fi

echo ""
echo "Verifying installation..."

# Check OpenCV installation
if pkg-config --exists opencv4; then
    echo "OpenCV found: $(pkg-config --modversion opencv4)"
else
    echo "OpenCV verification failed"
    exit 1
fi

echo ""
echo "Compiling tools..."

# Compile ProTools analyzer
echo "Building ProTools analyzer..."
make protools

# Compile Video analyzer
echo "Building Video analyzer..."
make video

echo ""
echo "Running tests..."
make test

echo ""
echo "Setup complete!"
echo ""
echo "Available tools:"
echo "  ./protools_analyzer <session.ptx>     - Analyze Pro Tools sessions"
echo "  ./video_analyzer <video.mp4>          - Analyze video for audio design"
echo ""
echo "For more information, see README.md"
