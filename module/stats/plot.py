import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import decimal
import cv2

def show_image(img):
    #画像の表示
    plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    print("image info")
    print("shape: ", img.shape, ",", "max:", np.amax(img))
    plt.show()



def gaussian_plot(title, data):

    from scipy.optimize import curve_fit

    #numpyでヒストグラムを作成
    hist = np.histogram(data, bins=25, range=(3.0, 15.0))
    #ヒストグラムの中央値の配列を作成
    diff = np.diff(hist[1])
    hist_center = hist[1][:-1] + diff

    #フィッティング用のガウス関数を作成
    def Gaussian(x, *params):
        amp = params[0]
        wid = params[1]
        ctr = params[2]
        y = amp * np.exp( -((x - ctr)/wid)**2)
        return y

    #ガウシアンでフィッティング
    guess = [5000, 4, 9]
    params, cov = curve_fit(Gaussian, hist_center, hist[0], p0=guess)

    #強度, 半値幅, 平均値を表示
    fmhm = 2 * np.sqrt(2*np.log(2)) * params[1]
    print("Amplitude = "+str(params[0]))
    print("FMHM = "+str(fmhm))
    print("Mean = "+str(params[2]))

    #ヒストグラムと分布関数(ガウシアン)を重ねて表示
    plt.title(title)
    plt.bar(hist_center, hist[0], width=0.3)
    x = np.linspace(5,15, 1000)
    plt.plot(x, Gaussian(x, *params), c="red",lw=2, ls="--")
    
    return plt


def drop_hazure(array, max_error = 1.5):
    sample_data = pd.Series(array)
    # 第1四分位値
    Q1 = sample_data.quantile(0.25)
    # 第3四分位値
    Q3 = sample_data.quantile(0.75)
    # 第1四分位値 と 第3四分位値 の範囲
    IQR = Q3 - Q1
    # 下限値として、Q1 から 1.5 * IQRを引いたもの 
    LOWER_Q = Q1 - max_error * IQR
    # 上限値として、Q3 に 1.5 * IQRをたしたもの 
    HIGHER_Q = Q3 + max_error * IQR

    # 四分位数の観点から、外れ値を除外する
    return sample_data[(LOWER_Q <= sample_data) & (sample_data <= HIGHER_Q)]


def t_test(a,b):
    import scipy.stats

    t,p = scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
    significance = ""

    if p < 0.001:
        significance = "*** (p < 0.001)"
    elif p < 0.001:
        significance = "** (p < 0.01)"
    elif p < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "n.s."

    return t, p, significance

def _decimal(num, prec=3):
    with decimal.localcontext() as ctx:
        ctx.prec     = prec                      # 有効桁数を設定
        ctx.rounding = decimal.ROUND_HALF_UP    # いわゆる四捨五入を丸め方に設定

        # 有効桁数は即座に適用される
        n = ctx.create_decimal(str(num))
        return n


class Particles_Plot():
    def __init__(self, result):
        self.data = {
            "Areas": {
                "title": "Area (px*px)",
                "data": [result["area"] for result in result]
            },
            "Circularities": {
                "title": "Circularity",
                "data": [result["circularity"] for result in result]
            },
            "Eq_Diameters": {
                "title": "Equal Diameters (px)",
                "data": [result["eq_diameter"] for result in result]
            },
            "Ex_Volume": {
                "title": "Expected Volumes (px*px*px)",
                "data": [result["expected_volume"] for result in result]
            }
        }


    def histogram(self):
        fig = plt.figure(figsize=(8,6))
        
        plt.subplot(2,2,1)
        plt.title(self.data["Areas"]["title"])
        plt.hist(self.data["Areas"]["data"], bins=10, rwidth=0.7)
        
        plt.subplot(2,2,2)
        plt.title(self.data["Circularities"]["title"])
        plt.hist(self.data["Circularities"]["data"], bins=10, rwidth=0.7)
        
        plt.subplot(2,2,3)
        plt.title(self.data["Eq_Diameters"]["title"])
        plt.hist(self.data["Eq_Diameters"]["data"], bins=10, rwidth=0.7)

        plt.subplot(2,2,4)
        plt.title(self.data["Ex_Volume"]["title"])
        plt.hist(self.data["Ex_Volume"]["data"], bins=10, rwidth=0.7)

        return plt

    def gaussian_plot(self, key):
        return gaussian_plot(self.data[key]["title"], self.data[key]["data"])


def plot_compare_single(title, data_0, data_1, label_0="Negative", label_1="Positive"):
    t, p, s = t_test (data_0, data_1)

    fig = plt.figure(figsize=(12,12))

    ax = plt.subplot(2,2,1)
    plt.title(f"{title}\n {s}")
    ax.set_xticklabels([label_0, label_1])
    ax.boxplot( (data_0, data_1 ) , showmeans=True)
    
    return plt


def plot_histogram(data):
    fig = plt.figure(figsize=(12,12))
    
    for i, data in enumerate(data):    
        plt.subplot(2,2,i+1)
        plt.title(data["title"])
        plt.hist(data["values"], bins=10, rwidth=0.7)

    return plt

def plot_compare(data, label_0="positive", label_1="negative"):
    fig = plt.figure(figsize=(12,12))
    
    for i, data in enumerate(data.values()): 
        data_0 = data["values"]["positive"]
        data_1 = data["values"]["negative"]
        
        t, p, s = t_test (data_0, data_1)

        ax = plt.subplot(3,2,i+1)
        plt.title(f'{data["title"]},  {s}')
        ax.set_xticklabels([label_0, label_1])
        ax.boxplot( (data_0, data_1) , showmeans=True, sym="+")

    return plt


def Compare_Mosaic(data_positive, data_negative):
    data = [data_negative, data_positive]

    fig = plt.figure(figsize=(12,12))


    data0 = drop_hazure(data_negative["Areas"]["data"])
    data1 = drop_hazure(data_positive["Areas"]["data"])
    
    t, p, s = t_test (data0, data1)

    ax = plt.subplot(2,2,1)
    plt.title(f'{data_negative["Areas"]["title"]},  {s}')
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)


    data0 = drop_hazure(data_negative["Circularities"]["data"])
    data1 = drop_hazure(data_positive["Circularities"]["data"])

    t, p, s = t_test (data0, data1)
    ax = plt.subplot(2,2,2)
    plt.title(f'{data_negative["Circularities"]["title"]},  {s}')
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)

    data0 = drop_hazure(data_negative["Eq_Diameters"]["data"])
    data1 = drop_hazure(data_positive["Eq_Diameters"]["data"])

    t, p, s = t_test (data0, data1)
    ax = plt.subplot(2,2,3)
    plt.title(f'{data_negative["Eq_Diameters"]["title"]},  {s}')
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)

    data0 = drop_hazure(data_negative["Ex_Volume"]["data"])
    data1 = drop_hazure(data_positive["Ex_Volume"]["data"])

    t, p, s = t_test (data0, data1)
    ax = plt.subplot(2,2,4)
    plt.title(f'{data_negative["Ex_Volume"]["title"]},  {s}')
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)

    return plt

if __name__ == "__main__":

    with open("result/all_result.json") as f:
        data = json.load(f)
    
    fig = plt.figure(figsize=(12,12))


    data0 = drop_hazure(data[0]["Areas"])
    data1 = drop_hazure(data[1]["Areas"])
    
    t, p, s = t_test (data0, data1)

    ax = plt.subplot(2,2,1)
    plt.title(f"Area [px^2],  {s}")
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)


    data0 = drop_hazure(data[0]["Circularities"])
    data1 = drop_hazure(data[1]["Circularities"])

    t, p, s = t_test (data0, data1)
    ax = plt.subplot(2,2,2)
    plt.title(f"Circularity,  {s}")
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)


    data0 = drop_hazure(data[0]["Eq_diameters"])
    data1 = drop_hazure(data[1]["Eq_diameters"])

    t, p, s = t_test (data0, data1)
    ax = plt.subplot(2,2,3)
    plt.title(f"Eq. Diameters [px],  {s}")
    ax.set_xticklabels(['GFP-Negative', 'GFP-Positive'])
    ax.boxplot( (data0, data1 ) , showmeans=True)

    plt.savefig( "result/all_result_plot.jpg" )
    plt.show()
