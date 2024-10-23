import cv2
import numpy as np
import os

class PanoramaStitching:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.stitcher = cv2.Stitcher.create()

    def load_images(self):
        for filename in os.listdir(self.image_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.image_folder, filename)
                image = cv2.imread(image_path)
                self.images.append(image)

    def collect_homographies(self):
        """
        Calculates homographies between pairs of images and returns them.

        Returns:
            A list of homographies, where each homography is a 3x3 transformation matrix.
        """

        num_images = len(self.images)
        homographies = []

        for i in range(num_images - 1):
            img1 = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(self.images[i + 1], cv2.COLOR_BGR2GRAY)

            # Detect SIFT keypoints and compute descriptors
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # Match keypoints using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Calculate homography using RANSAC
            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                homographies.append(homography)
            else:
                print("Not enough matches found for image pair", i, "and", i + 1)

        return homographies

    def stitch_images(self):
        homographies = self.collect_homographies()
        if homographies:
            (status, stitched_image) = self.stitcher.stitch(self.images, homographies)
            if status == cv2.STITCHER_OK:
                print("Panorama stitching successful!")
                return stitched_image
            else:
                print("Panorama stitching failed.")
                return None
        else:
            print("No homographies found.")
            return None

    def save_panorama(self, output_filename):
        stitched_image = self.stitch_images()
        if stitched_image is not None:
            cv2.imwrite(output_filename, stitched_image)
            print("Panorama saved as", output_filename)
        else:
            print("Unable to save panorama.")

if __name__ == "__main__":
    # image_folder = input("Enter the path to the image folder: ")
    output_filename = "demo"
    image_folder="Images/I1"
    stitcher = PanoramaStitching(image_folder)
    stitcher.load_images()
    stitcher.save_panorama(output_filename)