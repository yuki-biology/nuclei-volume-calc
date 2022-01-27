import cv2
import numpy as np
import matplotlib.pyplot as plt

from .stats import show_image

def to_odd(i):
    i = int(i)
    if i % 2 == 0:
        return i + 1
    return i

def watershed_separate(_img, kernel_size = 10):
    kernel_size = to_odd(kernel_size)

    opening = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
    print("\ropening", end="")
    #show_image(opening)

    # モルフォロジー演算のDilationを使う
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    sure_bg = cv2.dilate(_img,kernel,iterations=2)
    print("\rsure_bg", end="")
    #show_image(sure_bg)

    dist_transform = cv2.distanceTransform(_img,cv2.DIST_L2,5)
    print("\rdist_transform", end="")
    #show_image(dist_transform)

    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    #print('\r閾値（距離変換で得られた値の最大値×0.5）:',ret, end="")
    #show_image(sure_fg)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    print("\runknown", end="")
    #show_image(unknown)

    # foregroundの1オブジェクトごとにラベル（番号）を振っていく
    ret, markers = cv2.connectedComponents(sure_fg)
    print("\rmarkers", end="")
    #show_image(markers)

    markers += 1
    np.unique(markers,return_counts=True)

    markers[unknown==255] = 0
    # "ヒント"であるmarkersをwatershedに適応する
    markers = cv2.watershed(opening,markers)
    print("\rwatershed", end="")
    #show_image(markers)

    img = np.zeros(shape=opening.shape, dtype=np.uint8)
    img[markers == -1] = [255,255,255]
    res = -img+opening
    res[res==1] = 0
    return res