import pdb
import glob
import cv2
import os
import numpy as np
import random
from scipy.ndimage import affine_transform
from scipy.linalg import lstsq
 


class PanaromaStitcher():
    def __init__(self, scale_percent=50, verbose=True):
        self.scale_percent = scale_percent
        self.verbose = verbose

    def resize_image(self, image):
        width = int(image.shape[1] * self.scale_percent / 100)
        height = int(image.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim)
    
    def load_images_from_folder(self, folder_path):
        image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image at {path}")
        return images

    def make_panaroma_for_images_in(self, path, lowe_ratio=0.75, max_threshold=3.0):
        # Load images and initialize
        image_paths = sorted(glob.glob(path + os.sep + '*'))
        images = [cv2.imread(img_path) for img_path in image_paths if cv2.imread(img_path) is not None]
        gray_images=[]
        for img in images:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(img_gray)

        SIFT_Detector = cv2.SIFT_create()
        
        result_image = self.resize_image(images[0])
        homography=None
        
        for i in range(1, len(images)):
            current_image = self.resize_image(images[i])
            if current_image is None:
                print(f"Image {i} could not be loaded, skipping.")
                continue

            # keypoints_A, features_A = self.detect_features(result_image)
            # keypoints_B, features_B = self.detect_features(current_image)

            # # Match keypoints between the current and result images
            # matches, homography = self.match_keypoints(keypoints_A, keypoints_B, features_A, features_B, lowe_ratio, max_threshold)
            

            # if homography is None:
            #     print(f"Not enough matches between images {i-1} and {i}, skipping this image.")
            #     continue

            # # Warp images to create the panorama
            # result_image = self.warp_images(result_image, current_image, homography)
            result_gray_image=cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
            current_gray_image=cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            result_image,homography=self.stitch([result_image,current_image],[result_gray_image,current_gray_image],SIFT_Detector)
        
        
        

        return result_image,homography
    
    def stitch(self, imgs, grays, SIFT_Detector, threshold = 0.75, blend = 'linearBlendingWithConstantWidth'):
        # Step1 - extract the keypoints and features by SIFT
        key_points_1, descriptors_1 = SIFT_Detector.detectAndCompute(grays[0], None)
        key_points_2, descriptors_2 = SIFT_Detector.detectAndCompute(grays[1], None)
        
        # Step2 - extract the match point with threshold (David Lowe's ratio test)
        matches = self.matchKeyPoint(key_points_1, descriptors_1, key_points_2, descriptors_2, threshold)
        
        # Step3 - fit the homography model with RANSAC algorithm
        H = self.Ransac_Homography(matches)

        # Step4 - Warp image to create panoramic image
        warp_img = self.warp(imgs[0], imgs[1], H, blend)
        print(H)#printing homography matrix
        
        return warp_img,H

    def matchKeyPoint(self, kps_1, features_1, kps_2, features_2, threshold):
        '''
        Match the Keypoints beteewn two image
        '''
        matches = []
        for i in range(len(features_1)):
            min_index, min_distance = -1, np.inf
            sec_index, sec_distance = -1, np.inf
            
            for j in range(len(features_2)):
                distance = np.linalg.norm(features_1[i] - features_2[j])
                
                if distance < min_distance:
                    sec_index, sec_distance = min_index, min_distance
                    min_index, min_distance = j, distance
                    
                elif distance < sec_distance and sec_index != min_index:
                    sec_index, sec_distance = j, distance
                    
            matches.append([min_index, min_distance, sec_index, sec_distance])

        good_matches = []
        for i in range(len(matches)):
            if matches[i][1] <= matches[i][3] * threshold:
                good_matches.append([(int(kps_1[i].pt[0]), int(kps_1[i].pt[1])), 
                                     (int(kps_2[matches[i][0]].pt[0]), int(kps_2[matches[i][0]].pt[1]))])
        
        return good_matches
    
    
    def solve_homography(self, kps_1, kps_2):
        A = []
        for i in range(len(kps_1)):
            A.append([kps_1[i, 0], kps_1[i, 1], 1, 0, 0, 0, -kps_1[i, 0] * kps_2[i, 0], -kps_1[i, 1] * kps_2[i, 0], -kps_2[i, 0]])
            A.append([0, 0, 0, kps_1[i, 0], kps_1[i, 1], 1, -kps_1[i, 0] * kps_2[i, 1], -kps_1[i, 1] * kps_2[i, 1], -kps_2[i, 1]])

        # Solve system of linear equations Ah = 0 using SVD
        u, sigma, vt = np.linalg.svd(A)
        
        # pick H from last line of vt
        H = np.reshape(vt[8], (3, 3))
        
        # normalization, let H[2,2] equals to 1
        H = (1/H.item(8)) * H
        
        return H
    
    def Ransac_Homography(self, matches):
        img1_kp = []
        img2_kp = []
        for kp1, kp2 in matches:
            img1_kp.append(list(kp1))
            img2_kp.append(list(kp2))
        img1_kp = np.array(img1_kp)
        img2_kp = np.array(img2_kp)
        
        threshold = 5
        iteration_num = 2000
        max_inliner_num = 0
        best_H = None
        
        for iter in range(iteration_num):
            random_sample_idx = random.sample(range(len(matches)), 4)
            H = self.solve_homography(img1_kp[random_sample_idx], img2_kp[random_sample_idx])

            # find the best Homography have the the maximum number of inlier
            inliner_num = 0
            
            for i in range(len(matches)):
                if i not in random_sample_idx:
                    concateCoor = np.hstack((img1_kp[i], [1])) # add z-axis as 1
                    dstCoor = H @ concateCoor.T
                    
                    # avoid divide zero number, or too small number cause overflow
                    if dstCoor[2] <= 1e-8: 
                        continue
                    
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - img2_kp[i]) < threshold):
                        inliner_num = inliner_num + 1
            
            if (max_inliner_num < inliner_num):
                max_inliner_num = inliner_num
                best_H = H

        return best_H
    
    
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(img_left_mask[i, j]) > 0):
                    linearBlending_img[i, j] = img_left[i, j]
                else:
                    linearBlending_img[i, j] = img_right[i, j]
        return linearBlending_img
    
    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # we need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 3 # constant width
        
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
                    
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) 
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            
            # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)
            
            # left 
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 0

        
        linearBlendingWithConstantWidth_img = np.copy(img_right)
        linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
        # linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(img_left_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = img_left[i, j]
                else:
                    linearBlendingWithConstantWidth_img[i, j] = img_right[i, j]
        return linearBlendingWithConstantWidth_img

                

    def warp(self, img1, img2, H, blendType):
        left_down = np.hstack(([0], [0], [1]))
        left_up = np.hstack(([0], [img1.shape[0]-1], [1]))
        right_down = np.hstack(([img1.shape[1]-1], [0], [1]))
        right_up = np.hstack(([img1.shape[1]-1], [img1.shape[0]-1], [1]))

        # Transform the corners using H
        warped_left_down = H @ left_down.T
        warped_left_up = H @ left_up.T
        warped_right_down = H @ right_down.T
        warped_right_up = H @ right_up.T

        # Determine the size of the warped image
        x1 = int(min(min(min(warped_left_down[0], warped_left_up[0]), min(warped_right_down[0], warped_right_up[0])), 0))
        y1 = int(min(min(min(warped_left_down[1], warped_left_up[1]), min(warped_right_down[1], warped_right_up[1])), 0))
        size = (img2.shape[1] + abs(x1), img2.shape[0] + abs(y1))

        # Adjust the transformation to shift images into the new canvas size
        A = np.float32([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]])
        transformed_H1 = A @ H
        transformed_H2 = A

        # warp perspective function
        warped1 = self.warpPerspective(img1, transformed_H1, size)
        warped2 = self.warpPerspective(img2, transformed_H2, size)

        # Blending
        if blendType == 'linearBlendingWithConstantWidth':
            result = self.linearBlendingWithConstantWidth([warped1, warped2])
        else:
            result = self.linearBlending([warped1, warped2])

        return result

    def warpPerspective(self, src, M, dsize):
        """ Custom implementation of warp perspective.
            src: Source image
            M: Homography matrix (3x3)
            dsize: Size of the output image (width, height)
        """
        h, w = dsize[1], dsize[0]
        dst = np.zeros((h, w, src.shape[2]), dtype=src.dtype)  # Output canvas
        
        M_inv = np.linalg.inv(M)  # Inverse transformation matrix
        
        for y in range(h):
            for x in range(w):
                # Map the destination pixel (x, y) back to source image
                source_coords = M_inv @ np.array([x, y, 1])
                source_x, source_y = source_coords[0] / source_coords[2], source_coords[1] / source_coords[2]
                
                # Bilinear interpolation for non-integer coordinates
                if 0 <= source_x < src.shape[1] - 1 and 0 <= source_y < src.shape[0] - 1:
                    x0, y0 = int(source_x), int(source_y)
                    x_diff, y_diff = source_x - x0, source_y - y0

                    # Interpolate
                    top_left = src[y0, x0]
                    top_right = src[y0, x0 + 1]
                    bottom_left = src[y0 + 1, x0]
                    bottom_right = src[y0 + 1, x0 + 1]
                    
                    pixel_value = (top_left * (1 - x_diff) * (1 - y_diff) +
                                top_right * x_diff * (1 - y_diff) +
                                bottom_left * (1 - x_diff) * y_diff +
                                bottom_right * x_diff * y_diff)
                    
                    dst[y, x] = pixel_value

        return dst

    
    def say_hi(self):
        print('Hi From John Doe...')

    def do_something(self):
        print("I am computer and I am doing something...")

    def do_something_more(self):
        print("Let me do something more...")



    