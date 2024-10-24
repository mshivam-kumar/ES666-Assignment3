import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
import numpy as np
import numpy as np
import os
from scipy.signal import convolve2d
import random
class PanaromaStitcher():

    def __init__(self, scale_percent=50, verbose=True):
        self.scale_percent = scale_percent
        self.verbose = verbose

    # def log(self, message):
    #     if self.verbose:
    #         print(message)


    def resize_image(self, image):
        width = int(image.shape[1] * self.scale_percent / 100)
        height = int(image.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim)

    
    def load_images_from_folder(self, folder_path):
        image_paths = sorted(glob(os.path.join(folder_path, '*')))
        print("h1")
        images = []
        for path in image_paths:
            # Load the image
            img = cv2.imread(path)
            # print("h2")
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {path}")
        
        return images

    def make_panaroma_for_images_in(self, path, lowe_ratio=0.75, max_threshold=4.0):
        # print("h33")
        imf = path
        image_path = sorted(glob.glob(imf+os.sep+'*'))
        # images=self.load_images_from_folder(path)
        # print(images[0])
        images = []
        for path in image_path:
            # Load the image
            img = cv2.imread(path)
            # print("h2")
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {path}")
        # print("NOw")

        print('Found {} Images for stitching'.format(len(images)))
        self.say_hi()
        self.do_something()
        self.do_something_more()
        # Initialize the result with the first image
        result_image = self.resize_image(images[0])
        
        for i in range(1, len(images)):
            current_image = self.resize_image(images[i])
            if current_image is None:
                print(f"Image {i} could not be loaded, skipping.")
                continue

            keypoints_A, features_A = self.detect_features(result_image)
            keypoints_B, features_B = self.detect_features(current_image)

            # Match keypoints between the current and result images
            matches, homography = self.match_keypoints(keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold)

            if homography is None:
                print(f"Not enough matches between images {i-1} and {i}, skipping this image.")
                continue

            # Warp images to create the panorama
            result_image = self.warp_images(result_image, current_image, homography)
            
        return result_image,homography


    def make_panaroma_for_images_in(self, path, lowe_ratio=0.75, max_threshold=4.0):
        """ This function implements the stitching logic using the provided images. """
        imf = path
        image_paths = sorted(glob.glob(imf + os.sep + '*'))

        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {img_path}")

        print('Found {} Images for stitching'.format(len(images)))

        if len(images) == 0:
            raise ValueError("No images found for stitching.")

        # Initialize the result with the first image
        result_image = images[0]

        for i in range(1, len(images)):
            current_image = images[i]

            # Resize images if necessary
            result_image = self.resize_image(result_image)
            current_image = self.resize_image(current_image)

            if current_image is None:
                print(f"Image {i} could not be resized, skipping.")
                continue

            # Detect features
            keypoints_A, features_A = self.detect_features(result_image)
            keypoints_B, features_B = self.detect_features(current_image)

            # Match keypoints between the current and result images
            match_result = self.match_keypoints(keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold)
            
            print(f"Match result for images {i-1} and {i}: {match_result}")

            # Ensure we unpack the match_result correctly
            if len(match_result) == 2:
                matches, homography = match_result
            else:
                print(f"Unexpected number of return values from match_keypoints: {len(match_result)}")
                continue

            if homography is None:
                print(f"Not enough matches between images {i-1} and {i}, skipping this image.")
                continue

            # Warp images to create the panorama
            result_image = self.warp_images(result_image, current_image, homography)

        return result_image
    
    def detect_features(self, image):
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(image, None)
        return np.float32([kp.pt for kp in keypoints]), features

    def match_keypoints(self, keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold):
        # Match features between two images using BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(features_A, features_B, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            points_A = np.float32([keypoints_A[m.queryIdx] for m in good_matches])
            points_B = np.float32([keypoints_B[m.trainIdx] for m in good_matches])

            # Compute the homography matrix
            homography, _ = self.ransac_homography(points_A, points_B)
            return good_matches, homography
        return None, None

    def ensure_three_channels(self,image):
      """
      Ensure the image has 3 channels. If the image is grayscale (2D), convert it to 3 channels by duplicating the single channel.
      """
      if len(image.shape) == 2:  # Grayscale image
          return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
      return image  # Image already has 3 channels

    def warp_images(self, image_A, image_B, homography):
      # Ensure both images have 3 channels
      image_A = self.ensure_three_channels(image_A)
      image_B = self.ensure_three_channels(image_B)

      height_A, width_A = image_A.shape[:2]
      height_B, width_B = image_B.shape[:2]

      # Define corners of the first image
      corners_A = np.array([[0, 0], [0, height_A], [width_A, height_A], [width_A, 0]], dtype='float32')
      corners_A_transformed = self.perspective_transform(corners_A.reshape(-1, 1, 2), homography)

      # Prepare corners for image B
      corners_B = np.array([[0, 0], [0, height_B], [width_B, height_B], [width_B, 0]], dtype='float32')

      # Stack corners of both images
      all_corners = np.vstack((corners_A_transformed.reshape(-1, 2), corners_B))

      # Calculate bounding box for the stitched image
      [x_min, y_min] = np.int32(all_corners.min(axis=0).flatten())
      [x_max, y_max] = np.int32(all_corners.max(axis=0).flatten())

      # Log shapes for debugging 
    #   self.log(f"Corners A transformed shape: {corners_A_transformed.shape}")
    #   self.log(f"Corners B shape: {corners_B.shape}")
    #   self.log(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")

      # Create translation matrix to account for the bounding box
      translation_dist = [-x_min, -y_min]
      homography_translation = np.array([[1, 0, translation_dist[0]], 
                                          [0, 1, translation_dist[1]], 
                                          [0, 0, 1]])

      # Warp image_A to the new perspective
      result_image = self.perspective_transform(image_A, homography_translation @ homography)

      # Place image_B onto the warped image
      result_image[translation_dist[1]:translation_dist[1] + height_B, 
                  translation_dist[0]:translation_dist[0] + width_B] = image_B

      return result_image


    def normalize_points(self, points):
        """ Normalizes points so that they are centered at the origin and the average distance is sqrt(2). """
        mean = np.mean(points, axis=0)
        dists = np.linalg.norm(points - mean, axis=1)
        scale = np.sqrt(2) / np.mean(dists)
        T = np.array([[scale, 0, -scale * mean[0]],
                      [0, scale, -scale * mean[1]],
                      [0, 0, 1]])
        normalized_points = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
        return T, normalized_points[:2].T

    def estimate_homography(self, src_pts, dst_pts):
        """ Estimates homography matrix using Direct Linear Transformation (DLT). """
        A = []
        for i in range(src_pts.shape[0]):
            x, y = src_pts[i, 0], src_pts[i, 1]
            xp, yp = dst_pts[i, 0], dst_pts[i, 1]
            A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
            A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        return H / H[2, 2]  # Normalize so that H[2, 2] = 1

    def compute_inliers(self, H, src_pts, dst_pts, threshold):
        """ Computes the inliers given a homography matrix and threshold. """
        src_pts_hom = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
        projected_pts = np.dot(H, src_pts_hom.T).T
        projected_pts /= projected_pts[:, 2][:, np.newaxis]  # Normalize
        
        dists = np.linalg.norm(projected_pts[:, :2] - dst_pts, axis=1)
        inliers = np.where(dists < threshold)[0]
        
        return inliers

    def ransac_homography(self, src_pts, dst_pts, max_iter=1000, threshold=5.0):
        """ Computes homography using RANSAC. """
        best_H = None
        best_inliers = []
        
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return best_H, best_inliers
        
        for _ in range(max_iter):
            # Randomly sample 4 correspondences
            indices = np.random.choice(src_pts.shape[0], 4, replace=False)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]
            
            # Estimate homography for this subset
            H = self.estimate_homography(src_sample, dst_sample)
            
            # Compute inliers
            inliers = self.compute_inliers(H, src_pts, dst_pts, threshold)
            
            # Update the best homography if this one has more inliers
            if len(inliers) > len(best_inliers):
                best_H = H
                best_inliers = inliers
        
        # Recompute homography using all inliers
        if len(best_inliers) > 0:
            best_H = self.estimate_homography(src_pts[best_inliers], dst_pts[best_inliers])
        
        return best_H, best_inliers
        
    def perspective_transform(points, homography):
        # Ensure points are in homogeneous coordinates
        num_points = points.shape[0]
        points_homogeneous = np.hstack((points, np.ones((num_points, 1))))  # Shape (N, 3)
        
        # Apply the homography transformation
        transformed_homogeneous = homography @ points_homogeneous.T  # Shape (3, N)
        
        # Normalize the points by the last row (w')
        w = transformed_homogeneous[2, :]  # Shape (N,)
        transformed_cartesian = transformed_homogeneous[:2, :] / w  # Shape (2, N)

        # Return the transformed points as (N, 2)
        return transformed_cartesian.T  # Shape (N, 2)
    
        
    def say_hi(self):
        print('Hii From John Doe...')
    
    def do_something(self):
        print("I am computer and I am doing something...")
        return None
    
    def do_something_more(self):
        print("Let me do something more...")
        return None






















