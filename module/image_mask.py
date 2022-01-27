import cv2
import numpy as np
from .image_utils import *

# 画像をマスク
def mask_image(img, img_mask, reverse=False):
    height, width = img.shape[0:2]

    if reverse:
        _img_mask = cv2.bitwise_not(img_mask)
    else:
        _img_mask = img_mask

    if _img_mask.shape != (width, height):
        _img_mask = cv2.resize(_img_mask.astype(np.uint8), dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    elif len(_img_mask.shape) == 3:
        _img_mask = gray_scale(_img_mask)
    
    return cv2.bitwise_and(img, img, mask=_img_mask).astype(np.uint8)


#画像をマスクして値を測定
def count_masked_area_intensity(img, img_mask, reverse=False):
    masked_img = mask_image(img, img_mask, reverse)
    return masked_img.sum()

#画像をマスクして値を測定 (3D)
def count_masked_region_intensity_3d(imgs, imgs_mask, reverse=False):
    result = 0
    for i, (img, img_mask) in enumerate(zip(imgs, imgs_mask)):
        result += count_masked_area_intensity(img, img_mask, reverse)
    return int(result)