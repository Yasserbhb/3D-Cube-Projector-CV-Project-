import cv2
import numpy as np

from collections import defaultdict
import time



# Define the function to process a single frame
def process_frame(frame, color_ranges):
    centroids_dict = defaultdict(list)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Special handling for combined red ranges
    red_lower1 = np.array(color_ranges["Red"][0])
    red_upper1 = np.array(color_ranges["Red"][1])
    red_lower2 = np.array(color_ranges["Red2"][0])
    red_upper2 = np.array(color_ranges["Red2"][1])

    # Create masks for both red ranges and combine them
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    combined_red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply opening and closing to the combined red mask
    kernel = np.ones((7,7), np.uint8)
    opened_mask = cv2.morphologyEx(combined_red_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask)
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
    cx, cy = centroids[largest_label]
    centroid = (cy, cx)
    centroids_dict["Red"].append(centroid)

    
    # Process other color ranges (excluding Red2, already handled)
    for color, (lower, upper) in color_ranges.items():
        if color in ["Red", "Red2"]:
            continue  # Skip as we already handled "Red"
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        
        
        # Create the mask for the current color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask)
        if num_labels>1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
            cx, cy = centroids[largest_label]
            centroid = (cy, cx)
            centroids_dict[color].append(centroid)
        

    

    return centroids_dict










