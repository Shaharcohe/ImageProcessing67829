import mediapy as media
import numpy as np

#*********************************** Constants ***********************************#
TYPE_ONE = 1
TYPE_TWO = 2
VALID_TYPES = [TYPE_ONE, TYPE_TWO]
MAX_GRAYSCALE_VALUE = 256
MIN_GRAYSCALE_VALUE = 0
COLOR_FORMAT = 'gray'
NULL_SCENE_CUT = (0, 0)


#*********************************************************************************#

def detect_scene_cut(path: str, type: int) -> tuple[int, int]:
    """
    Detects the scene cut in a video by comparing consecutive frames.

    :param path: The path to the video file.
    :param type: The type of comparison to use (1 for type one comparison, 2 for type two comparison).
    :return: A tuple of integers (last frame of the first scene, first frame of the second scene).
    """
    histogram = reg_histogram if type == TYPE_ONE else cumulative_histogram
    with media.VideoReader(path, output_format=COLOR_FORMAT) as reader:
        prev_frame = histogram(reader.read())
        scene_cut = NULL_SCENE_CUT
        max_distance = 0
        index = 1
        for frame in reader:
            cur_frame = histogram(frame)
            distance = np.linalg.norm(cur_frame - prev_frame)
            if distance > max_distance:
                scene_cut = (index - 1, index)
                max_distance = distance
            index += 1
            prev_frame = cur_frame
        return scene_cut

def reg_histogram(frame):
    return np.histogram(frame, bins=MAX_GRAYSCALE_VALUE, range=(MIN_GRAYSCALE_VALUE, MAX_GRAYSCALE_VALUE))[0]

def cumulative_histogram(frame):
    return np.cumsum(reg_histogram(frame))


def main(video_path: str, video_type: int) -> tuple[int, int]:
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    if video_type not in VALID_TYPES:
        raise ValueError('Video type must be one of: ' + str(VALID_TYPES))
    return detect_scene_cut(video_path, video_type)
