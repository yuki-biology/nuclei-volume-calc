from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


def convert_to_csv(target_folder, target_title):
    target_filename = os.path.join(target_folder, target_title + ".json")

    output_filename = os.path.join(target_folder, target_title + ".csv")
    output_stat_filename = os.path.join(target_folder, target_title + "_stat.csv")
    
    df = pd.read_json(target_filename)
    print(df)

    cols = df.columns
    cols = [i for i in cols if "Intensity" in i]

    for i in cols:
        df[i + "_per_Volume_um3"] = df[i] / df["Volume_um3"]

    for i in cols:
        if i != "Intensity_Cy5":
            df[i + "_per_Intensity_Cy5"] = df[i] / df["Intensity_Cy5"]
    
    if "Intensity_FITC" in cols:
        th_all, res = otsu_binarization(df["Intensity_FITC"])
        th_per, res = otsu_binarization(df["Intensity_FITC_per_Volume_um3"])
        
        df["FITC_positive"] = df["Intensity_FITC_per_Volume_um3"] > th_per
        
        df_pos = df[df["FITC_positive"]]
        df_neg = df[df["FITC_positive"] != True]

        print(df_pos)
        print(df_neg)

        df_stat = pd.DataFrame(index = df.columns, columns = [
            "t-test p-value", "significance", "t-test p-value (pos > neg)", "significance (pos > neg)", "t-test p-value (neg > pos)" , "significance (neg > pos)",
            "pos_mean", "neg_mean", "pos_std", "neg_std", "pos_median", "neg_median", "pos_min", "neg_min", "pos_max", "neg_max"
        ])

        for i in df_stat.index:
            if i != "Id" or i != "FITC_positive":
                v_pos = df_pos[i]
                v_neg = df_neg[i]
                t, p, s = t_test(v_pos, v_neg)
                df_stat.loc[i, "t-test p-value"] = p
                df_stat.loc[i, "significance"] = s
                t, p, s = t_test(v_pos, v_neg, option="single-sided")
                df_stat.loc[i, "t-test p-value (pos > neg)"] = p
                df_stat.loc[i, "significance (pos > neg)"] = s
                t, p, s = t_test(v_neg, v_pos, option="single-sided")
                df_stat.loc[i, "t-test p-value (neg > pos)"] = p
                df_stat.loc[i, "significance (neg > pos)"] = s
                df_stat.loc[i, "pos_mean"] = v_pos.mean()
                df_stat.loc[i, "neg_mean"] = v_neg.mean()
                df_stat.loc[i, "pos_std"] = v_pos.std()
                df_stat.loc[i, "neg_std"] = v_neg.std()
                df_stat.loc[i, "pos_median"] = v_pos.median()
                df_stat.loc[i, "neg_median"] = v_neg.median()
                df_stat.loc[i, "pos_min"] = v_pos.min()
                df_stat.loc[i, "neg_min"] = v_neg.min()
                df_stat.loc[i, "pos_max"] = v_pos.max()
                df_stat.loc[i, "neg_max"] = v_neg.max()
    
        print(df_stat)

        df_res = pd.concat([df_pos, df_neg])
        df_res.to_csv(output_filename)
        df_stat.to_csv(output_stat_filename)

    else:
        df_stat = pd.DataFrame(index = df.columns, columns = [
            "mean", "std", "median", "min", "max"
        ])
        for i in df_stat.index:
            v = df[i]
            df_stat.loc[i, "mean"] = v.mean()
            df_stat.loc[i, "std"] = v.std()
            df_stat.loc[i, "median"] = v.median()
            df_stat.loc[i, "min"] = v.min()
            df_stat.loc[i, "max"] = v.max()

        print(df_stat)

        df_res = df
        df_res.to_csv(output_filename)
        df_stat.to_csv(output_stat_filename)



if __name__ == "__main__":
    target_title = "result_Cy5_Ad_Th"
    
    import glob
    files = glob.glob("./result/OIB/**")
    print(files)

    @handle_elaped_time(len(files))
    def iterarion(i):
        print("converting: " + files[i])
        target_folder = files[i]
        convert_to_csv(target_folder, target_title)

    iterarion()