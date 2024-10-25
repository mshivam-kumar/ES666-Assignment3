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
            homography, _ = cv2.findHomography(points_A, points_B, cv2.RANSAC, max_threshold)
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
      corners_A_transformed = cv2.perspectiveTransform(corners_A.reshape(-1, 1, 2), homography)

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
      result_image = cv2.warpPerspective(image_A, homography_translation @ homography, (x_max - x_min, y_max - y_min))

      # Place image_B onto the warped image
      result_image[translation_dist[1]:translation_dist[1] + height_B, 
                  translation_dist[0]:translation_dist[0] + width_B] = image_B

      return result_image



       

   
    
    def say_hi(self):
        print('Hii From John Doe...')
    
    def do_something(self):
        print("I am computer and I am doing something...")
        return None
    
    def do_something_more(self):
        print("Let me do something more...")
        return None

