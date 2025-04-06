import argparse
import math
from typing import Literal, List, Optional, Callable, Tuple

import cv2
import random
import numpy as np


def euclidian_distance(u, v):
    return np.linalg.norm(u - v)


class Util:

    @staticmethod
    def create_video_from_image(image_path, output_video_path, num_frames=30 * 5, fps=30):
        """ Creates a debugging video from a given image path. """
        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image at {image_path} not found.")

        # Get the image dimensions
        height, width, layers = image.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Write the image multiple times to simulate a video
        for _ in range(num_frames):
            out.write(image)
        out.release()

    @staticmethod
    def create_video(frames, output_path='output.mp4', fps = 120):
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            if frame.shape != frames[0].shape:
                print(f"Frame shape is {frame.shape}")
            out.write(frame)
        out.release()

    @staticmethod
    def load_frames(path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        frames = []
        while True:
            ret, frame = cap.read()
            if frame is not None:
                frames.append(frame)
            if not ret:
                break
        return frames

    @staticmethod
    def draw_matches(image1, image2, keypoints1, keypoints2, matches, path):
        """
        Draw matches between keypoints in two images.
        """
        if keypoints1 is None or not isinstance(keypoints1[0], cv2.KeyPoint):
            raise ValueError("keypoints1 must be a list of cv2.KeyPoint objects.")
        if keypoints2 is None or not isinstance(keypoints2[0], cv2.KeyPoint):
            raise ValueError("keypoints2 must be a list of cv2.KeyPoint objects.")
        if not matches or not isinstance(matches[0], cv2.DMatch):
            raise ValueError("matches must be a list of cv2.DMatch objects.")

        # Draw the matches
        matched_image = cv2.drawMatches(
            image1, keypoints1, image2, keypoints2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(255, 255, 0),
            matchesThickness=1
        )

        # Save the matched image
        cv2.imwrite(path, matched_image)
        print(f"Matched image saved to {path}")


class Aligner:

    def __init__(self, iterations: int = 1000,
                 seed: int = None,
                 inlier_threshold=1,
                 nn_ratio=0.75):
        self.seed = seed
        self.iterations = iterations
        self.random = random.Random(seed)
        self.w = 0
        self.h = 0
        self.inlier_threshold = inlier_threshold
        self.sift = cv2.SIFT_create(sigma=3)
        self.nn_ratio = nn_ratio

    def find_matching_points(self, keypoints1, descriptors1, keypoints2, descriptors2):

        # Step 2: Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Step 3: Apply Lowe's ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < self.nn_ratio * n.distance:  # Lowe's ratio test
                good_matches.append(m)

        if len(good_matches) < 2:
            print("Not enough good matches to compute homography.")
            return None, None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return src_pts, dst_pts

    def update_seed(self, new_seed):
        self.seed = new_seed
        self.random.seed(new_seed)

    def backward_warp(self, image, matrix_inv):
        if len(matrix_inv) == 2:
            return cv2.warpAffine(image,
                                  matrix_inv,
                                  (self.w, self.h),
                                  flags=cv2.INTER_LINEAR,  # bilinear interpolation
                                  borderMode=cv2.BORDER_CONSTANT
                                  )
        else:
            return cv2.warpPerspective(image,
                                       matrix_inv,
                                       (self.w, self.h),
                                       flags=cv2.INTER_LINEAR,  # bilinear interpolation
                                       borderMode=cv2.BORDER_CONSTANT
                                       )

    def homography(self, src_points, dst_points):
        """
        Estimates the rotation and translation parameters between two sets of points.

        Args:
            src_points (np.ndarray): Nx2 array of source points.
            dst_points (np.ndarray): Nx2 array of destination points.

        Returns:
            float: Rotation angle in degrees.
            tuple: Translation (tx, ty).
        """
        v1, v2 = src_points
        vector1 = v1 - v2
        u1, u2 = dst_points
        vector2 = u1 - u2
        # Normalize the vectors
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)

        # Compute cosine and sine of the angle
        cos_theta = np.dot(vector1.flatten(), vector2.flatten())
        sin_theta = np.cross(vector1.flatten(), vector2.flatten())

        # Build the rotation matrix
        rotation_mat = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        v1 = (v1 @ rotation_mat.T).flatten()
        v2 = (v2 @ rotation_mat.T).flatten()
        u1 = u1.flatten()
        u2 = u2.flatten()

        tx = ((u1[0] - v1[0]) + (u2[0] - v2[0])) / 2
        ty = ((u1[1] - v1[1]) + (u2[1] - v2[1])) / 2
        matrix = np.array([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty],
            [0, 0, 1]
        ])


        return matrix, cos_theta, (tx, ty)

    def to_homogeneous(self, points):
        """
        Converts 2D points to homogeneous coordinates.

        Args:
            points (np.ndarray): Nx1x2 array of 2D points.

        Returns:
            np.ndarray: Nx3 array of points in homogeneous coordinates.
        """
        # Reshape points to (N, 2) if necessary
        if len(points.shape) == 3 and points.shape[1] == 1:
            points = points.reshape(-1, 2)

        # Add a column of ones to the points
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points, ones))
        return homogeneous_points

    def ransac(self, src_points, dst_points):
        """
        Implements RANSAC to find a rigid transformation (rotation + translation).

        Args:
            src_points (np.ndarray): Nx2 array of source points.
            dst_points (np.ndarray): Nx2 array of destination points.

        Returns:
            tuple: Best rotation angle (degrees), translation (tx, ty), and inlier mask.
        """
        num_points = len(src_points)
        best_inliers = 0
        best_transform = None
        src_homogeneous = self.to_homogeneous(src_points)
        dst_homogenous = self.to_homogeneous(dst_points)
        for _ in range(self.iterations):
            sample_indices = np.random.choice(num_points, 2, replace=False)
            src_sample = src_points[sample_indices]
            p1, p2 = src_sample
            p1 = p1.flatten()
            p2 = p2.flatten()
            # if abs(p1[1] - p2[1]) > 15:
            #     continue
            dst_sample = dst_points[sample_indices]

            matrix, _, (tx, ty) = self.homography(src_sample, dst_sample)
            transformed_src = np.dot(src_homogeneous, matrix.T)
            distances = np.linalg.norm(transformed_src - dst_homogenous, axis=1)
            inlier_mask = distances < self.inlier_threshold
            num_inliers = np.sum(inlier_mask)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_transform = matrix

        if best_transform is None:
            raise ValueError("RANSAC failed to find a valid transformation.")
        # best_transform = np.vstack([best_transform, np.array([0, 0, 1])])

        return best_transform

    def alignment(self, src=None,
                  dest=None,
                  key_src=None,
                  d_src=None,
                  key_dest=None,
                  d_dest=None):
        if src is not None and dest is not None:
            key_src, d_src = self.sift.detectAndCompute(src, None)
            key_dest, d_dest = self.sift.detectAndCompute(dest, None)
        src_points, dst_points = self.find_matching_points(key_src, d_src, key_dest, d_dest)
        return self.ransac(src_points, dst_points)

    def match(self, src, dest):
        keypoints1, descriptors1 = self.sift.detectAndCompute(src, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(dest, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < self.nn_ratio * n.distance:  # Lowe's ratio test
                good_matches.append(m)

        if len(good_matches) < 2:
            print("Not enough good matches to compute homography.")
            return None, None


        return keypoints1, keypoints2, good_matches


    def get_tx(self, mat):
        return mat[0, 2]

    def get_ty(self, mat):
        return mat[1, 2]

    def process_transition_matrix(self, matrix, movement_dir):
        """
        Processes a single transition matrix to extract tx and zero out the translation component.

        Args:
            args (tuple): (index, matrix)

        Returns:
            tuple: (tx value, modified matrix)
        """
        k = 1 if movement_dir == 'y' else 2
        matrix[0, k] = 0  # Set the translation component to 0
        return matrix

    def half_panorama(self, frames, start, end, direction):
        aligned = []
        tx = []
        last_matrix = np.eye(3)
        key_dest, desc_dest = None, None
        for i in range(start, end, direction):
            key_src, desc_src = self.sift.detectAndCompute(frames[i], None)
            if key_dest is None:
                key_dest, desc_dest = self.sift.detectAndCompute(frames[i - direction], None)
            matrix = self.alignment(
                key_src=key_src,
                d_src=desc_src,
                key_dest=key_dest,
                d_dest=desc_dest
            ) @ last_matrix
            c_tx = self.get_tx(matrix)
            matrix = self.process_transition_matrix(matrix, 'x')
            result = self.backward_warp(frames[i], matrix)
            aligned.append(result)
            tx.append(c_tx)
            last_matrix = matrix
            key_dest, desc_dest = key_src, desc_src
        if direction == -1:
            aligned.reverse()
            tx.reverse()
        return aligned, tx

    def align_panorama(self, frames):
        reference_frame_idx = len(frames) // 2
        h, w, _ = frames[reference_frame_idx].shape
        self.h = h
        self.w = w
        r_aligned, r_tx = self.half_panorama(frames, reference_frame_idx, -1, -1)
        l_aligned, l_tx = self.half_panorama(frames, reference_frame_idx + 1, len(frames), 1)
        aligned = r_aligned + [frames[reference_frame_idx]] + l_aligned
        gaps = r_tx + l_tx + [0]
        return aligned, gaps

    def movement_direction(self, img1, img2):
        transform = self.alignment(img1, img2)
        tx = self.get_tx(transform)
        ty = self.get_ty(transform)
        dir = 'x' if abs(tx) >= abs(ty) else 'y'
        sign = tx if abs(tx) > abs(ty) else ty
        sign /= abs(sign)
        return dir, sign


class Stitcher:

    def __init__(self, increment=0):
        self.increment = increment

    def get_band(self, i, center, prev_gap, gap):
        gap = int(abs(math.ceil(gap / 2)))
        prev_gap = int(abs(math.floor(prev_gap / 2)))
        x_start = center - prev_gap
        x_end = center + gap

        return x_start, x_end

    def get_panorama_size(self, gaps, width, height, move):
        total = 0
        for i, gap in enumerate(gaps):
            if i == 0:
                band = self.get_band(0, 0, 0, gaps[i])
            else:
                band = self.get_band(0, 0, gaps[i-1], gaps[i])
            total += band[1] - band[0]
        if move:
            return total + width + 1, height, 3
        else:
            return total , height, 3

    def stitch_panorama(self, aligned_frames, gaps, center: int, move: bool):
        height, width, _ = aligned_frames[0].shape
        new_width, new_height, channels = self.get_panorama_size(gaps, width, height, move)
        panorama = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        current = None
        for i in range(len(aligned_frames)):
            if i == 0:
                prev_gap = 0
            else:
                prev_gap = gaps[i - 1]
            start, end = self.get_band(i, center, prev_gap, gaps[i])
            if current is None:
                current = start if move else 0
            try:
                panorama[:, current: current + (end - start), ] = aligned_frames[i][:, start:end, :]
            except:
                print(current, current+ (end - start), new_width)
            panorama[:, current: current + (end - start), ] = aligned_frames[i][:, start:end, :]
            current += (end - start)
        return panorama


class PanoramaGenerator:

    @staticmethod
    def preprocess(frames, aligner):
        # axis, dir = aligner.movement_direction(frames[0], frames[1])
        # rotate = (axis == 'y')
        # reverse = (dir == -1)
        # new_frames = frames[::-1] if reverse else frames
        # if rotate:
        #     for i in range(len(frames)):
        #         new_frames[i] = cv2.rotate(new_frames[i], cv2.ROTATE_90_CLOCKWISE)
        return frames, False, False  # reverse, rotate

    @staticmethod
    def create_panoramas(aligned_frames,
                         gaps,
                         non_static,
                         panorama_centers=None):
        if len(aligned_frames) < 2:
            raise Exception("The video contains less than 2 frames and therefore is too short for panorama.")
        width = aligned_frames[0].shape[1]
        pad = int(max([abs(round(gap)) for gap in gaps])) // 2 + int(0.15*width)
        stitcher = Stitcher()
        panoramas = []
        centers = panorama_centers if panorama_centers is not None else list(range(pad, width - pad))
        for center in centers:
            panorama = stitcher.stitch_panorama(aligned_frames, gaps, center, non_static)
            panoramas.append(panorama)
        return panoramas

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program for image feature detection, alignment, and creating mosaics.")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["feature_detection", "dynamic_mosaic", "viewpoint_mosaic", 'panorama'],
        required=True,
        help="Set the program mode: 'feature_detection' for image feature detection and alignment,"
             " 'dynamic_mosaic' for creating a dynamic mosaic, 'viewpoint_mosaic' for creating a "
             "viewpoint mosaic, or panorama for creating a simple panorama."
    )

    # Optional image paths
    parser.add_argument(
        "--image_a",
        type=str,
        help="Path to the first input image (optional)."
    )

    parser.add_argument(
        "--image_b",
        type=str,
        help="Path to the second input image (optional)."
    )

    # Optional video path
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to the input video (optional)."
    )

    # Optional output path
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output file.",
        required=True
    )

    parser.add_argument(
        "--center",
        type=float,
        default=0.5,
        help="Panorama center (optional)."
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.mode == "feature_detection":
        src = cv2.imread(args.image_a)
        dst = cv2.imread(args.image_b)
        keypoints_s, keypoints_d,  matches = Aligner().match(src, dst)
        Util.draw_matches(src, keypoints_s, dst, keypoints_d,  matches, args.output_path)
        return
    frames = Util.load_frames(args.video_path)
    aligned_frames, gaps = Aligner().align_panorama(frames)
    non_static = (args.mode == "viewpoint_mosaic")
    panoramas = PanoramaGenerator.create_panoramas(aligned_frames, gaps, non_static)
    if args.mode == "dynamic_mosaic":
        panoramas.reverse()
    if args.mode == "viewpoint_mosaic":
        panoramas = panoramas + panoramas[::-1]
    Util.create_video(panoramas, args.output_path)


if __name__ == "__main__":
    main()

