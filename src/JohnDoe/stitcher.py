import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
import numpy as np
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
            homography,_=self.ransac_homography(points_A,points_B,max_threshold)
            return good_matches, homography
        return None, None
    
    import numpy as np

    def compute_homography_dlt(self,points_A, points_B):
        """
        Compute the homography matrix using DLT given corresponding points.
        """
        assert len(points_A) == len(points_B) and len(points_A) >= 4, "At least 4 points are required"
        
        # Construct the matrix A from point correspondences
        A = []
        for (x_A, y_A), (x_B, y_B) in zip(points_A, points_B):
            A.append([-x_A, -y_A, -1, 0, 0, 0, x_B * x_A, x_B * y_A, x_B])
            A.append([0, 0, 0, -x_A, -y_A, -1, y_B * x_A, y_B * y_A, y_B])
        
        A = np.array(A)
        
        # Perform SVD on A
        U, S, Vt = np.linalg.svd(A)
        
        # The last row of Vt (or column of V) gives the solution
        H = Vt[-1].reshape(3, 3)
        
        # Normalize to make the bottom-right element 1
        H /= H[-1, -1]
        
        return H

    def ransac_homography(self,points_A, points_B, max_threshold, max_iterations=1000):
            best_H = None
            max_inliers = 0
            n_points = len(points_A)
            best_inliers = []

            for _ in range(max_iterations):
                # Randomly sample 4 point correspondences
                indices = np.random.choice(n_points, 4, replace=False)
                sample_points_A = [points_A[i] for i in indices]
                sample_points_B = [points_B[i] for i in indices]
                
                # Compute the homography matrix for this sample
                H = self.compute_homography_dlt(sample_points_A, sample_points_B)
                
                # Count inliers
                inliers = []
                for i in range(n_points):
                    # Transform points_A[i] using H
                    p_A = np.array([*points_A[i], 1])
                    p_B_proj = H @ p_A
                    p_B_proj /= p_B_proj[2]  # Normalize
                    
                    # Compute the Euclidean distance to points_B[i]
                    p_B = np.array(points_B[i])
                    error = np.linalg.norm(p_B - p_B_proj[:2])
                    
                    if error < max_threshold:
                        inliers.append(i)
                
                # Update best homography if this one has more inliers
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_H = H
                    best_inliers = inliers

            return best_H, best_inliers

        # Example usage
        # points_A and points_B are lists of corresponding points in the two images
        # max_threshold defines the inlier distance threshold for RANSAC
        # points_A = [(x1_A, y1_A), (x2_A, y2_A), ...]
        # points_B = [(x1_B, y1_B), (x2_B, y2_B), ...]

        # H, inliers = ransac_homography(points_A, points_B, max_threshold=5.0)


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
      result_image = self.warp_perspective(image_A, homography_translation, (x_max - x_min, y_max - y_min))

      # Place image_B onto the warped image
      result_image[translation_dist[1]:translation_dist[1] + height_B, 
                  translation_dist[0]:translation_dist[0] + width_B] = image_B

      return result_image



    def warp_perspective(self,image, homography, output_shape):
        """
        Manually applies a homography transformation to an entire image.
        
        Parameters:
            image (np.ndarray): Input image (H, W, C) or (H, W) grayscale.
            homography (np.ndarray): 3x3 homography matrix.
            output_shape (tuple): (width, height) of the output image.
            
        Returns:
            np.ndarray: Transformed image with shape (height, width, C) or (height, width) for grayscale.
        """
        height, width = output_shape[1], output_shape[0]
        channels = image.shape[2] if image.ndim == 3 else 1
        output_image = np.zeros((height, width, channels), dtype=image.dtype)
        
        # Invert the homography to map output pixels back to input pixels
        homography_inv = np.linalg.inv(homography)
        
        # Iterate over every pixel in the output image
        for y_out in range(height):
            for x_out in range(width):
                # Transform output pixel (x_out, y_out) to source image coordinates
                transformed_coord = homography_inv @ np.array([x_out, y_out, 1])
                x_in, y_in = transformed_coord[:2] / transformed_coord[2]
                
                # Bilinear interpolation: Check bounds and interpolate if within input image
                if 0 <= x_in < image.shape[1] - 1 and 0 <= y_in < image.shape[0] - 1:
                    x0, y0 = int(x_in), int(y_in)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # Get the four neighboring pixel values for bilinear interpolation
                    Ia = image[y0, x0] if y0 < image.shape[0] and x0 < image.shape[1] else 0
                    Ib = image[y0, x1] if y0 < image.shape[0] and x1 < image.shape[1] else 0
                    Ic = image[y1, x0] if y1 < image.shape[0] and x0 < image.shape[1] else 0
                    Id = image[y1, x1] if y1 < image.shape[0] and x1 < image.shape[1] else 0
                    
                    # Compute the weights for each neighbor
                    wa = (x1 - x_in) * (y1 - y_in)
                    wb = (x_in - x0) * (y1 - y_in)
                    wc = (x1 - x_in) * (y_in - y0)
                    wd = (x_in - x0) * (y_in - y0)
                    
                    # Interpolate pixel values
                    output_image[y_out, x_out] = wa * Ia + wb * Ib + wc * Ic + wd * Id

        # If grayscale, remove extra channel dimension
        if channels == 1:
            output_image = output_image.reshape(height, width)

        return output_image


    def perspective_transform(self,points, homography):
        """
        Manually applies a homography transformation to a set of 2D points.
        
        Parameters:
            points (np.ndarray): A NumPy array of shape (N, 1, 2) containing N 2D points in OpenCV format.
            homography (np.ndarray): A 3x3 homography matrix.
            
        Returns:
            np.ndarray: A NumPy array of shape (N, 1, 2) containing the transformed points in OpenCV format.
        """
        # Reshape points to (N, 2) for easier manipulation, then add homogeneous coordinate
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points.reshape(num_points, 2), np.ones((num_points, 1))])

        # Apply the homography matrix to each point
        transformed_points = homography @ homogeneous_points.T  # Shape (3, N)

        # Convert from homogeneous to Cartesian coordinates
        transformed_points /= transformed_points[2, :]  # Normalize by w'

        # Reshape back to (N, 1, 2) to match OpenCV's perspectiveTransform output format
        return transformed_points[:2, :].T.reshape(-1, 1, 2)


       

   
    
    def say_hi(self):
        print('Hii From John Doe...')
    
    def do_something(self):
        print("I am computer and I am doing something...")
        return None
    
    def do_something_more(self):
        print("Let me do something more...")
        return None

