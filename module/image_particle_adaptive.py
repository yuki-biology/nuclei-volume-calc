from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
import scipy.ndimage as ndimage
import cv2

from .image_utils import separate_image


DEFAILT_BLOCK_SIZE = 1/2.25 * 40

def to_odd(i):
    i = int(i)
    if i % 2 == 0:
        return i + 1
    return i


def adaptive_threshold(img, block_size=DEFAILT_BLOCK_SIZE, C=5, mag=5):
    block_size = to_odd(block_size)

    res = cv2.bitwise_not(img)

    # # ブロックサイズを変更する
    h, w = res.shape[:-1]
    res = cv2.resize(res, (w*mag, h*mag))

    # ノイズ除去
    res = cv2.medianBlur(res, 5)

    # ガウシアンフィルタでぼかす
    res = cv2.GaussianBlur(res,(block_size*mag, block_size*mag),0)

    # 二値化
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res  = cv2.adaptiveThreshold(res, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size*mag, C)


    res = cv2.bitwise_not(res)
    
    #穴埋め
    res = ndimage.binary_fill_holes(res).astype("uint8") * 255

    #リサイズ
    res = cv2.resize(res, (w, h))
    return res


#バッチ処理
def batch_binarize_images(images, block_size=DEFAILT_BLOCK_SIZE, C=5, mag=5):
    img_stack = cv2.vconcat(images)
    img_th = adaptive_threshold(img_stack, block_size, C, mag)
    # 画像を分離
    imgs_spq = separate_image(img_th, split_x=len(images))
    return imgs_spq