# 3D-Cube-Projector

A computer vision application that projects a 3D cube onto a 2D plane using homography estimation and camera calibration.

## Overview

This project demonstrates the implementation of projective geometry techniques to superimpose a virtual 3D cube onto a physical colored grid. The system:

1. Detects colored markers on a predefined grid using HSV color segmentation
2. Computes the homography matrix between world and image coordinates
3. Calculates the projection matrix using camera intrinsic parameters
4. Renders a 3D cube correctly positioned on the detected grid



## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- SciPy
- Matplotlib
- scikit-image

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

- `calculate.py` - Core projection matrix and homography calculations
- `color_detection_trials.py` - Prototype code for HSV color detection
- `cube.py` - 3D cube drawing functions
- `get_points.py` - Color marker detection functions
- `main.py` - Main program that processes videos and applies cube projection
- `main_tierces.py` - Alternative approach with automatic camera calibration

## Usage

### Basic Usage

1. Ensure you have a colored grid pattern with the following colors arranged in a 3×3 grid:
   - Green (top-left), Red (top-center), Yellow (top-right)
   - Blue (middle-left), White (center), Purple (middle-right)
   - Orange (bottom-left), Cyan (bottom-center), Magenta (bottom-right)

2. Place your video file in the project directory and run:

```bash
python main.py
```

The script uses a predefined camera matrix. The output video will show a 3D cube projected onto the colored grid.

### Using Automatic Camera Calibration

If you want to calibrate the camera automatically:

1. Record a video of a checkerboard pattern (9×6 inner corners) from various angles
2. Update the video path in `main_tierces.py`
3. Run:

```bash
python main_tierces.py
```

## Testing Instructions

### Using the Provided Test Video

Run the code directly to test with the included small test video:

```bash
python main.py
```

### Using the Large Test Video

1. Download the test video from [Google Drive](https://drive.google.com/file/d/1qGRb8BAlC41i-LZpksoz-8guBA9KREjG/view?usp=sharing)
2. Place it in the local folder alongside the scripts
3. Update the input file in `main.py`:
   ```python
   video_path = "test_video_large.mp4"  # Line 28
   ```
4. Run the script

### Expected Output

- An `output.mp4` file will be generated after processing
- Execution time:
  - Approximately 15 seconds for 100 frames of `test_video_small.mp4`
  - Approximately 55 seconds for 360 frames of `test_video_large.mp4`

## Technical Details

### Camera Matrix (K)

The camera intrinsic matrix can be provided in two ways:

1. **Manual Calibration**: Using MATLAB's camera calibrator (results are stored in code)
2. **Automatic Calibration**: Using `calibrateCamera` from OpenCV on checkerboard frames

### Color Detection

The project uses HSV color space for robust detection of the colored markers:

1. Creates masks for each color using predefined HSV ranges
2. Applies morphological operations (opening/closing) to clean up the masks
3. Identifies connected components and extracts the centroid of the largest region
4. Uses these centroids as key points for homography estimation

### Homography and Projection

The homography matrix H is calculated using:
- Image points (detected colored centroids)
- World points (predefined grid coordinates)

The projection matrix P is derived from:
- Camera intrinsic matrix K
- Homography matrix H
- Calculated scaling factor α

## Improvements and Future Work

- White balance correction for better color detection
- Support for different grid patterns
- Real-time processing capabilities
- Improved robustness to lighting variations

## Contributors

- DAI Yucheng
- BOUHAI Yasser

## License

This project is licensed under the MIT License - see the LICENSE file for details.
