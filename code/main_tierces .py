
import scipy.io

import cv2
import time
from calculate import *
from cube import *
from get_points import *



# Step 1: Define a function to calibrate camera using frames from a video
def calibrate_camera_from_video(video_file, checkerboard_size, square_size, frame_skip=10):
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points in world coordinates
    img_points = []  # 2D points in image coordinates

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                obj_points.append(objp)
                img_points.append(corners_refined)

                cv2.drawChessboardCorners(frame, checkerboard_size, corners_refined, ret)
                cv2.imshow('Chessboard Corners', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) > 0:
        ret, intrinsic_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        return intrinsic_matrix, dist_coeffs
    else:
        print("No valid frames for calibration were found.")
        return None, None

# Step 2: Calibrate the camera using the video
video_file = './test1.mp4'
checkerboard_size = (9, 6)  # Inner corners per row and column
square_size = 25  # Size of each square in mm

K, dist_coeffs = calibrate_camera_from_video(video_file, checkerboard_size, square_size)
print("Calibrated Intrinsic Matrix (K):\n", K)


# Define your color ranges
color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(170, 100, 100), (179, 255, 255)],  # Secondary red (wrapping around the hue circle)
    "Orange": [(11, 100, 100), (25, 255, 255)],
    "Yellow": [(26, 100, 100), (39, 255, 255)],
    "Green": [(40, 100, 100), (85, 255, 255)],
    "Cyan": [(86, 100, 100), (100, 255, 255)],
    "Blue": [(100, 100, 100), (125, 255, 255)],
    "Purple": [(126, 100, 100), (140, 255, 255)],
    "Magenta": [(140, 100, 100), (160, 255, 255)],
    "White": [(0, 0, 200), (180, 50, 255)]
}
# Path to your video file
video_path = "test_video_small.mp4"

# Path to save the output video
output_path = "output_tierces.mp4"


ptworld = np.array([
[0, 0],  # Green (Top-left)
[1, 0],  # Red (Top-center)
[2, 0],  # Yellow (Top-right)
[0, 1],  # Blue (Middle-left)
[1, 1],  # White (Center)
[2, 1],  # Purple (Middle-right)
[0, 2],  # Orange (Bottom-left)
# [1, 2],  # Cyan (Bottom-center)
[2, 2],  # Pink (Bottom-right)
])


# #yuchang
# K = np.array([
#     [3251.08236331607, 0, 2006.77185927389],
#     [0, 3252.36090006803, 1475.73672757365],
#     [0, 0, 1]
# ])




#K = scipy.io.loadmat("K.mat")
#K=np.array(K['K'])


cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count>=500  :
        break  # End of video
    # Convert frame to uint8 format (if needed)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Process the current frame
    centroids_dict = process_frame(frame, color_ranges)
    ptimage = np.array([
        centroids_dict["Green"][-1],  # Green
        centroids_dict["Red"][-1],    # Red
        centroids_dict["Yellow"][-1], # Yellow
        centroids_dict["Blue"][-1],   # Blue
        centroids_dict["White"][-1],  # White
        centroids_dict["Purple"][-1], # Purple
        centroids_dict["Orange"][-1], # Orange
        # centroids_dict["Cyan"][-1],   # Cyan
        centroids_dict["Magenta"][-1],   # Pink
    ])
    
    H = estimerH(ptimage,ptworld)

    alpha,Rt = calcule_P(K,H)

    drawCube(frame, K, Rt, alpha, size=1, center=(0,0, 0))
    
    
    # Draw centroids on the frame
    for color, centroids in centroids_dict.items():
        for centroid in centroids:
            # Draw a red circle at each centroid
            cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 5, (255, 0, 0), -1)  # (B, G, R)



    # Write the frame with centroids to the output video
    out.write(frame)



cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {output_path}")

