import numpy as np
import cv2

def drawCube(img, K, Rt, alpha, size=1, center=(0, 0, 0)):
    """
    Draws a 3D cube projected onto a 2D image.

    Args:
        img: Input image (will be modified in-place).
        K: Camera intrinsic matrix.
        Rt: Camera extrinsic matrix (rotation + translation).
        alpha: Scaling factor.
        size: Size of the cube.
        center: The 3D coordinates of the cube's center (x, y, z).
    """
    # Calculate the 8 vertices of the cube, shifted by the center
    x, y, z = center
    vertices = np.array([
        [x + size, y + size, z + size, 1],
        [x + size, y + size, z, 1],
        [x + size, y, z, 1],
        [x + size, y, z + size, 1],
        [x, y + size, z + size, 1],
        [x, y + size, z, 1],
        [x, y, z, 1],
        [x, y, z + size, 1]
    ])

    # Project the vertices onto the 2D image plane
    M = alpha * K @ Rt  # shape: (3, 4) if Rt is (3, 4), or (3, 4) slice of (4,4)

    # 2) Convert your vertices array to shape (4, N) for N vertices:
    #    e.g. if vertices is currently a list of shape (N, 4)
    vertices_np = np.array(vertices).T  # Now shape: (4, N)

    # 3) Multiply all vertices at once
    projected = M @ vertices_np  # shape: (3, N)

    # 4) Perform perspective divide (assuming row 2 is the z-component)
    x_vals = projected[0, :] / projected[2, :]
    y_vals = projected[1, :] / projected[2, :]

    # 5) Convert to integer tuples for OpenCV drawing
    projected_vertices = list(zip(x_vals.astype(int), y_vals.astype(int)))

    # Define the 12 edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Draw the edges of the cube on the image
    for edge in edges:
        cv2.line(img, projected_vertices[edge[0]], projected_vertices[edge[1]], (255, 0, 255), 2)

    return img
