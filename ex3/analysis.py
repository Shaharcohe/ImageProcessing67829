import cv2
import numpy as np

from ex3 import *

from PIL import Image
MASK_COLOR = (0, 0, 0)

def create_mask(path):
    img = cv2.imread(path)
    indicator = np.zeros(shape=img.shape, dtype=np.float32)
    value = ((abs(img[:, :, 0] - MASK_COLOR[0]) < 2) *
             (abs(img[:, :, 1] - MASK_COLOR[1]) < 2) *
             (abs(img[:, :, 2] == MASK_COLOR[2]) < 2))
    indicator[:,:,0] = value
    indicator[:,:,1] = value
    indicator[:,:,2] = value

    return 1 - indicator


def save_pyramid_results(shape, pyramid, prefix):
    for k in range(len(pyramid)):
        layer = (pyramid.layer(k, shape)  / 2) + 128
        cv2.imwrite(f'{prefix}_layer{k}.jpg', layer)

def pad_to_power_of_two(image):
    """
    Pads an image so that its dimensions are powers of 2.

    Args:
        image (np.ndarray): Input image (2D or 3D NumPy array).

    Returns:
        np.ndarray: Padded image with dimensions as powers of 2.
    """
    # Get current height and width
    height, width = image.shape[:2]

    # Calculate the nearest powers of 2
    new_height = 2 ** math.ceil(math.log2(height))
    new_width = 2 ** math.ceil(math.log2(width))

    # Calculate padding for each side
    pad_top = (new_height - height) // 2
    pad_bottom = new_height - height - pad_top
    pad_left = (new_width - width) // 2
    pad_right = new_width - width - pad_left

    if len(image.shape) == 3:  # Color image
        avg_color = tuple(map(int, image.mean(axis=(0, 1))))
    else:  # Grayscale image
        avg_color = int(image.mean())

    # Apply padding with zeros
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=avg_color
    )

    return padded_image


if __name__ == '__main__':
    # image_path = './images/wonder_woman.webp'
    # image = cv2.imread(image_path)
    # shape = image.shape[:2]
    # image = pad_to_power_of_two(image)
    # image = image.astype(np.int16)
    # pyramids = [GaussianPyramid(image), LaplacianPyramid(image)]
    # result_dirs = ['gaussian', 'laplacian']
    # for pyramid, dir in zip(pyramids, result_dirs):
    #     save_pyramid_results(shape, pyramid, dir)
    # lap_sum = pyramids[1].sum()
    # cv2.imwrite('laplacian_sum.jpg', lap_sum)
    mars = './images/galaxy2.webp'
    mars = cv2.imread(mars)
    mars = mars.astype(np.int16)
    shape = mars.shape[:2]
    new_shape = (2 ** round(math.log2(shape[0])), 2 ** round(math.log2(shape[1])))
    new_shape = (int(new_shape[1]), int(new_shape[0]))
    mars = cv2.resize(mars, new_shape)
    gr = './images/Givat_ram_entrance_night_1.jpg'
    gr = cv2.imread(gr)
    gr = gr.astype(np.int16)
    gr = cv2.resize(gr, new_shape)

    mask = create_mask('./images/new_mask.png')
    cv2.imwrite('new_mask.jpg', mask * 255)
    mask = mask.astype(np.int16)
    mask = cv2.resize(mask, new_shape)
    # mask = GaussianPyramid(mask).layer(2, shape)
    blended = BlendedPyramid(gr, mars, mask).sum()
    blended = cv2.resize(blended, gr.shape[:2][::-1])
    cv2.imwrite('mars_blended.jpg', blended)

    old = cv2.imread('./images/maapilim.jpg').astype(np.int16)
    new = cv2.imread('images/tlv.jpg').astype(np.int16)
    old = cv2.resize(old, dsize=(new.shape[1], new.shape[0]))
    print(old.shape)
    print(new.shape)
    old = pad_to_power_of_two(old)
    new = pad_to_power_of_two(new)
    cv2.imwrite('tel aviv.jpg', HybridPyramid(new, old, threshold=0.27).sum())
    shape = list(old.shape)
    shape[1] /= 2
    shape[1] = int(shape[0])
    half = np.zeros(shape, dtype=np.float32)
    mask = np.ones_like(old)
    mask[:, :shape[1], :] = half
    mask = GaussianPyramid(mask).layer(5, mask.shape)
    cv2.imwrite('tel aviv blend.jpg', BlendedPyramid(new, old, mask).sum())



