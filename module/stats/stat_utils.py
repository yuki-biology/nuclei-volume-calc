import pandas as pd
import scipy.stats

# 有意性の判断
def significance(p):
    if p < 0.001:
        return "*** (p < 0.001)"
    elif p < 0.001:
        return "** (p < 0.01)"
    elif p < 0.05:
        return "* (p < 0.05)"
    else:
        return "n.s."
    
# 対応のあるT検定
def t_test_rel(a,b, option = "two-sided"):
    t,p = scipy.stats.ttest_rel(a, b)

    # a > b となる確率
    if option == "single-sided":
        p = p / 2
        if t < 0:
            p = 1 - p
        else:
            p = p

    return t, p, significance(p)

# T検定
def t_test(a,b, option = "two-sided"):
    t,p = scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
    
    # a > b となる確率
    if option == "single-sided":
        p = p / 2
        if t < 0:
            p = 1 - p
        else:
            p = p
    
    return t, p, significance(p)

# 対応のあるT検定 (増減を検出)
def t_test_rel_compare(a,b):
    t, p, s = t_test_rel(a,b)
    if p < 0.05:
        _t, _p, _s = t_test_rel(a,b, "single-sided")
        if _p < 0.05:
            return _t, _p, _s + ", decreased"
        
        _t, _p, _s = t_test_rel(b, a, "single-sided")
        if _p < 0.05:
            return _t, _p, _s + ", inccreased"
    
    return t, p, s

# 相関検定 (ピアソン相関検定)
def corr_test(x,y):
    r, p = scipy.stats.pearsonr(x,y)
    return r, p , significance(p)


#大津の二値化
import cv2
import numpy as np
def otsu_binarization(img):
    img = np.array(img, dtype=np.uint16)
    threshold, result = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    result = result.astype(np.uint8)
    result = np.where(result == 255, True, False)
    return threshold, result

def triangle_binarization(img):
    img = np.array(img, dtype=np.uint8)
    threshold, result = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)
    result = result.astype(np.uint8)
    result = np.where(result == 255, True, False)
    return threshold, result

