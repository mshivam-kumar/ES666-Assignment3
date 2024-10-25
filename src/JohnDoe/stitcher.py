import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
import numpy as np
import cv2
import numpy as np
import os
import glob
from scipy.signal import convolve2d
import random
class PanaromaStitcher():

    def __init__(self, scale_percent=40, verbose=True):
        self.scale_percent = scale_percent
        self.verbose = verbose

    def resize_image(self, image):
        return resize_image_custom(image, self.scale_percent)

    def load_images_from_folder(self, folder_path):
        image_paths = sorted(glob(os.path.join(folder_path, '*')))
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {path}")
        return images

    def make_panaroma_for_images_in(self, path, lowe_ratio=0.75, max_threshold=4.0):
        image_path = sorted(glob.glob(path + os.sep + '*'))
        images = []
        for path in image_path:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {path}")
        
        print('Found {} Images for stitching'.format(len(images)))
        result_image = self.resize_image(images[0])
        
        for i in range(1, len(images)):
            current_image = self.resize_image(images[i])
            if current_image is None:
                print(f"Image {i} could not be loaded, skipping.")
                continue

            keypoints_A, features_A = self.detect_features(result_image)
            keypoints_B, features_B = self.detect_features(current_image)

            matches, homography = self.match_keypoints(keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold)

            if homography is None:
                print(f"Not enough matches between images {i-1} and {i}, skipping this image.")
                continue

            result_image = warp_images_custom(result_image, homography, (result_image.shape[0] + current_image.shape[0], result_image.shape[1] + current_image.shape[1]))

        return result_image, homography

    def detect_features(self, image):
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(image, None)
        return np.float32([kp.pt for kp in keypoints]), features

    def match_keypoints(self, keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(features_A, features_B, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            points_A = np.float32([keypoints_A[m.queryIdx] for m in good_matches])
            points_B = np.float32([keypoints_B[m.trainIdx] for m in good_matches])

            homography = find_homography_custom(points_A, points_B, threshold=max_threshold)
            return good_matches, homography
        return None, None

# Helper functions for custom operations

def resize_image_custom(image, scale_percent):
    height, width = image.shape[:2]
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x = i * (height - 1) / (new_height - 1)
            y = j * (width - 1) / (new_width - 1)
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
            r1 = (y2 - y) * image[x1, y1] + (y - y1) * image[x1, y2]
            r2 = (y2 - y) * image[x2, y1] + (y - y1) * image[x2, y2]
            resized_image[i, j] = (x2 - x) * r1 + (x - x1) * r2
    return resized_image

def perspective_transform_custom(points, homography):
    transformed_points = []
    for point in points:
        x, y = point[0]
        denominator = (homography[2, 0] * x + homography[2, 1] * y + homography[2, 2])
        x_trans = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
        y_trans = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
        transformed_points.append([x_trans, y_trans])
    return np.array(transformed_points).reshape(-1, 1, 2)

def find_homography_custom(points_A, points_B, threshold=3.0, iterations=2000):
    max_inliers = 0
    best_homography = None
    for _ in range(iterations):
        idx = random.sample(range(len(points_A)), 4)
        src_pts = points_A[idx]
        dst_pts = points_B[idx]
        A = []
        for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        inliers = 0
        for pt_a, pt_b in zip(points_A, points_B):
            x, y = pt_a
            x_trans = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
            y_trans = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
            if np.sqrt((x_trans - pt_b[0]) ** 2 + (y_trans - pt_b[1]) ** 2) < threshold:
                inliers += 1
        if inliers > max_inliers:
            max_inliers = inliers
            best_homography = H
    return best_homography if best_homography is not None else None

def warp_images_custom(image, homography, output_shape):
    height, width = output_shape
    warped_image = np.zeros((height, width, 3), dtype=image.dtype)
    for y in range(height):
        for x in range(width):
            x_prime = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / (homography[2, 0] * x + homography[2, 1] * y + homography[2, 2])
            y_prime = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / (homography[2, 0] * x + homography[2, 1] * y + homography[2, 2])
            x_prime = int(round(x_prime))
            y_prime = int(round(y_prime))
            if 0 <= x_prime < image.shape[1] and 0 <= y_prime < image.shape[0]:
                warped_image[y, x] = image[y_prime, x_prime]
    return warped_image
