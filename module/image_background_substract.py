import cv2
import numpy as np

def background_substract_KNN(images, mag=5, kernel_size=1):
    fgbg = cv2.createBackgroundSubtractorKNN()

    shape = images[0].shape
    h, w = shape[:2]
    images_result = []

    #一回目の解析 (最初の７枚は正常に認識されないので、この結果は使わない)
    for i, img in enumerate(images):
        print(f"\ranalyzing bsg: {i}", end="")

        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (w*mag, h*mag))
        img = cv2.GaussianBlur(img,(5,5),0)    
        fgmask = fgbg.apply(img)
    print("--done")

    #折返しで再解析 (こっちを使う)
    for i, img in enumerate(reversed(images)):
        print(f"\rre-analyzing bsg: {i}", end="")
        img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (w*mag, h*mag))
        img = cv2.GaussianBlur(img,(5,5),0)
        fgmask = fgbg.apply(img)            

        kernel = np.ones((int(mag*kernel_size), int(mag*kernel_size) ),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.resize(fgmask, (w, h))
        images_result.append(fgmask)
    print("--done")

    return images_result


#背景を除去 (背景差分ではなく背景除去ってことだったのかもしれない)
def background_substract_open(image, kernel_size=1):
    #kernel = np.ones((int(kernel_size), int(kernel_size) ),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size) ))
    background = cv2.dilate(image, kernel)
    substracted = cv2.subtract(image, background)
    return background, substracted

def batch_background_substract_open(images, kernel_size=1):
    result_b = []
    result_s = []
    for img in images:
        b, s = background_substract_open(img, kernel_size)
        result_b.append(b)
        result_s.append(s)
    return result_b, result_s