import pdb
import glob
import cv2
import os
import numpy as np
import random
from scipy.ndimage import affine_transform
from scipy.linalg import lstsq

class ImageStitcher:
    def __init__(self, scale=50, debug_mode=True):
        self.scale = scale
        self.debug = debug_mode

    def resize(self, img):
        width = int(img.shape[1] * self.scale / 100)
        height = int(img.shape[0] * self.scale / 100)
        return cv2.resize(img, (width, height))

    def load_images(self, directory):
        img_files = sorted(glob.glob(directory + os.sep + '*'))
        loaded_images = []
        for img_file in img_files:
            img = cv2.imread(img_file)
            if img is not None:
                loaded_images.append(img)
            else:
                print(f"Failed to load image: {img_file}")
        return loaded_images

    def make_panaroma_for_images_in(self, directory, ratio=0.75, threshold=3.0):
        imgs = self.load_images(directory)
        grayscale_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

        sift = cv2.SIFT_create()
        panorama = self.resize(imgs[0])
        homography_matrix = None
        
        for i in range(1, len(imgs)):
            resized_img = self.resize(imgs[i])
            if resized_img is None:
                print(f"Skipping image {i}")
                continue

            grayscale_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            grayscale_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            panorama, homography_matrix = self.stitch_pair(
                [panorama, resized_img], [grayscale_panorama, grayscale_resized_img], sift
            )

        return panorama, homography_matrix

    def stitch_pair(self, imgs, gray_imgs, sift_detector, match_ratio=0.75, blend_mode='constant_width_blend'):
        kp1, des1 = sift_detector.detectAndCompute(gray_imgs[0], None)
        kp2, des2 = sift_detector.detectAndCompute(gray_imgs[1], None)

        good_matches = self.find_keypoints(kp1, des1, kp2, des2, match_ratio)

        H_matrix = self.ransac_homography(good_matches)

        panorama = self.warp_images(imgs[0], imgs[1], H_matrix, blend_mode)
        print(H_matrix)

        return panorama, H_matrix

    def find_keypoints(self, kp1, des1, kp2, des2, ratio):
        matches = []
        for i, desc in enumerate(des1):
            distances = [np.linalg.norm(desc - d) for d in des2]
            sorted_dists = sorted(enumerate(distances), key=lambda x: x[1])
            if sorted_dists[0][1] < ratio * sorted_dists[1][1]:
                matches.append([(int(kp1[i].pt[0]), int(kp1[i].pt[1])), 
                                (int(kp2[sorted_dists[0][0]].pt[0]), int(kp2[sorted_dists[0][0]].pt[1]))])
        return matches

    def compute_homography(self, points1, points2):
        A = []
        for i in range(len(points1)):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            A.extend([[x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2],
                      [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]])
        _, _, Vt = np.linalg.svd(A)
        H = (1 / Vt[-1, -1]) * Vt[-1].reshape((3, 3))
        return H

    def ransac_homography(self, matches):
        src_pts = np.array([m[0] for m in matches])
        dst_pts = np.array([m[1] for m in matches])

        max_inliers = 0
        best_H = None
        threshold = 5
        for _ in range(2000):
            sample_indices = random.sample(range(len(matches)), 4)
            H_matrix = self.compute_homography(src_pts[sample_indices], dst_pts[sample_indices])

            inliers = 0
            for i in range(len(matches)):
                if i not in sample_indices:
                    estimated = H_matrix @ np.append(src_pts[i], 1)
                    estimated /= estimated[2]
                    if np.linalg.norm(estimated[:2] - dst_pts[i]) < threshold:
                        inliers += 1
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H_matrix
        return best_H

    def warp_images(self, img1, img2, H_matrix, blend_mode):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H_matrix)
        
        min_x, min_y = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
        max_x, max_y = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
        
        translation = np.float32([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        
        img1_transformed = cv2.warpPerspective(img1, translation @ H_matrix, (max_x - min_x, max_y - min_y))
        img2_transformed = cv2.warpPerspective(img2, translation, (max_x - min_x, max_y - min_y))

        if blend_mode == 'constant_width_blend':
            return self.constant_width_blend([img1_transformed, img2_transformed])
        else:
            return self.linear_blend([img1_transformed, img2_transformed])

    def linear_blend(self, imgs):
        img_left, img_right = imgs
        h, w = img_left.shape[:2]
        blend_img = np.copy(img_right)
        for y in range(h):
            blend_img[y, :w] = img_left[y, :w] * 0.5 + img_right[y, :w] * 0.5
        return blend_img

    def constant_width_blend(self, imgs):
        img_left, img_right = imgs
        h, w = img_left.shape[:2]
        blend_img = np.copy(img_right)
        for y in range(h):
            for x in range(w):
                alpha = max(0, min(1, 1 - abs(x - w // 2) / (w // 4)))
                blend_img[y, x] = alpha * img_left[y, x] + (1 - alpha) * img_right[y, x]
        return blend_img

    def say_hello(self):
        print('Hi From John Doe...')

    def compute(self):
        print("Computing task...")

    def additional_computation(self):
        print("Executing additional tasks...")
