from TICC_solver import TICC
import os
import load_data
import numpy as np
import sys


def load_data_from_dir(input_file):
    distance_file_ = os.path.join(input_file, "distance.npy")
    rssi_file_ = os.path.join(input_file, "rssi.npy")
    sd_file_ = os.path.join(input_file, "sd.npy")
    sl_file_ = os.path.join(input_file, "sl.npy")
    Data = load_data.prepare_data(distance_file_, rssi_file_, sd_file_, sl_file_)
    (m, n) = Data.shape  # m: num of observations, n: size of observation vector
    return Data, m, n


# fname = "data/108384_106840"
fname = "data/108384_106836"
# Get data into proper format
times_series_arr, time_series_rows_size, time_series_col_size = load_data_from_dir(fname)

ticc = TICC(window_size=3, number_of_clusters=7, beta=100000, maxIters=100,
            write_out_file=False, prefix_string="result/", visible=True)
cluster_assignment, cluster_MRFs, bic = ticc.fit(times_series_arr)

print(cluster_assignment.tolist())
print("bic:", bic)

"""
λ是nwxnw维矩阵
λ确定描述每个簇的MRFs中的稀疏度; β是鼓励相邻子序列被分配到同一个集群的平滑性惩罚
"""
