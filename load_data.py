# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def load_data(distance, rssi, sd, sl):
    distance_npy = np.round(np.load(distance), 4)
    rssi_npy = np.round(np.load(rssi), 4)
    sd_npy = np.round(np.load(sd), 4)
    sl_npy = np.round(np.load(sl), 4)
    return distance_npy, rssi_npy, sd_npy, sl_npy


def get_time_stamp_set(distance_npy, rssi_npy, sd_npy, sl_npy):
    distance_time = distance_npy[:, 0].tolist()
    # print(distance_time)
    rssi_time = rssi_npy[:, 0].tolist()
    # print(rssi_time)
    sd_time = sd_npy[:, 0].tolist()
    # print(sd_time)
    sl_time = sl_npy[:, 0].tolist()
    # print(sl_time)
    set_data_1 = set.union(set(distance_time), set(rssi_time))
    print(len(set_data_1))
    set_data_2 = set.union(set(sd_time), set(sl_time))
    print(len(set_data_2))
    set_data_3 = set.union(set_data_2, set_data_1)
    print(len(set_data_3))
    return set_data_3


def get_start_end(distance_npy, rssi_npy, sd_npy, sl_npy):
    distance_time = distance_npy[:, 0].tolist()
    start_time = distance_time[0]
    end_time = distance_time[-1]
    # print(distance_time)
    rssi_time = rssi_npy[:, 0].tolist()
    # print(rssi_time)
    sd_time = sd_npy[:, 0].tolist()
    # print(sd_time)
    sl_time = sl_npy[:, 0].tolist()
    # print(sl_time)

    if start_time < sd_time[0]:
        start_time = sd_time[0]
    if start_time < sl_time[0]:
        start_time = sl_time[0]

    if end_time > sd_time[-1]:
        end_time = sd_time[-1]
    if end_time > sl_time[-1]:
        end_time = sl_time[-1]

    return start_time, end_time


def process_data(distance_npy, sd_npy, sl_npy, rssi_npy, start_time, end_time):
    distance_len = distance_npy.shape[0]
    sd_list = []
    distance_list = []
    sl_list = []
    rssi_list = []
    distance_index = 0
    sl_index = 0
    sd_index = 0
    rssi_index = 0
    while distance_index < distance_len:
        s_time = distance_npy[distance_index][0]
        if s_time < start_time:
            distance_index = distance_index + 1
            continue
        elif s_time > end_time:
            break
        else:
            sub_sd_value, sd_index = get_insert_value(s_time, distance_npy[distance_index + 1][0], sd_npy, sd_index)
            sd_list.append([s_time, sub_sd_value])

            sub_sl_value, sl_index = get_insert_value(s_time, distance_npy[distance_index + 1][0], sl_npy, sl_index)
            sl_list.append([s_time, sub_sl_value])

            sub_rssi_value, rssi_index = get_insert_rssi_value(s_time, distance_npy[distance_index + 1][0], rssi_npy,
                                                               rssi_index)
            rssi_list.append([s_time, sub_rssi_value])

            distance_list.append(distance_npy[distance_index])
            distance_index = distance_index + 1
    # return np.log(np.array(distance_list)), np.array(sd_list), np.log(np.array(sl_list)), np.array(rssi_list)
    return np.array(distance_list), np.array(sd_list), np.array(sl_list), np.array(rssi_list)


def get_insert_value(start_time, end_time, values, index):
    temp_value = []
    # for _ in values:
    while index < len(values):
        _ = values[index]
        s_time = _[0]
        if end_time >= s_time >= start_time:
            temp_value.append(_[1])
            index = index + 1
            continue
        if s_time > end_time:
            break
        if s_time < start_time:
            index = index + 1
            continue
    if len(temp_value) > 0:
        return sum(temp_value) * 1.0 / len(temp_value), index
    else:
        return values[index][1], index


def get_insert_rssi_value(start_time, end_time, values, index):
    temp_value = []
    # for _ in values:
    while index < len(values):
        _ = values[index]
        s_time = _[0]
        if start_time > 6.37:
            # print(s_time)
            pass
        if end_time >= s_time >= start_time:
            temp_value.append(_[1])
            index = index + 1
            continue
        if s_time > end_time:
            break
        if s_time < start_time:
            index = index + 1
            continue
    if len(temp_value) > 0:
        return sum(temp_value) * 1.0 / len(temp_value), index
    else:
        if index >= len(values) or values[index][0] > end_time or values[index][0] < start_time:
            return 0, index
        else:
            return values[index][1], index


def get_start_end_interval(set_data_time):
    data_list = list(set_data_time)
    data_list.sort()
    start = None
    end = None
    inter = None
    cur = None
    for _ in data_list:
        if start is None or start > _:
            start = _
        if end is None or end < _:
            end = _
        if cur is None:
            cur = _
        else:
            if inter is None or inter < _ - cur:
                inter = _ - cur
                print(_ - cur)
            cur = _
    return start, end, inter


def plot_test(distance_npy, rssi_npy, sd_npy, sl_npy):
    valid_sd_list = sd_npy[:, 0]
    plt.plot(sd_npy[:, 0], sd_npy[:, 1], '.', color='silver', label='sd of truck')
    plt.plot(sl_npy[:, 0], sl_npy[:, 1], 'm.', label='sl of truck')
    plt.plot(distance_npy[:, 0], distance_npy[:, 1], 'g.',
             label='distance between truck and excavator')
    plt.plot(rssi_npy[:, 0], rssi_npy[:, 1], 'r.', label='rssi of excavator')
    plt.legend(loc='upper right')
    plt.show()


def plot_test_1(distance_npy, sd_npy, sl_npy):
    valid_sd_list = sd_npy[:, 0]
    plt.plot(sd_npy[:, 0], sd_npy[:, 1], '.', color='silver', label='sd of truck')
    plt.plot(sl_npy[:, 0], sl_npy[:, 1], 'm.', label='sl of truck')
    plt.plot(distance_npy[:, 0], distance_npy[:, 1], 'g.',
             label='distance between truck and excavator')
    plt.legend(loc='upper right')
    plt.show()


def prepare_data(distance_file, rssi_file, sd_file, sl_file):
    distance_npy, rssi_npy, sd_npy, sl_npy = load_data(distance_file, rssi_file, sd_file, sl_file)
    start_time, end_time = get_start_end(distance_npy, rssi_npy, sd_npy, sl_npy)
    d, sd, sl, rssi = process_data(distance_npy, sd_npy, sl_npy, rssi_npy, start_time, end_time)
    # ks_data = np.array([d[:, 1], sd[:, 1], sl[:, 1], rssi[:, 1]])
    # ks_data = np.array([d[:, 1], sd[:, 1], sl[:, 1]])
    ks_data = np.array([savgol_filter(d[:, 1], 9, 2), savgol_filter(sd[:, 1], 9, 2), savgol_filter(sl[:, 1], 9, 2)])
    # ks_data = np.array([savgol_filter(d[:, 1], 9, 2)])
    ks_data = ks_data.T
    return ks_data


if __name__ == "__main__":
    distance_file_ = "data/108801_106864/distance.npy"
    rssi_file_ = "data/108801_106864/rssi.npy"
    sd_file_ = "data/108801_106864/sd.npy"
    sl_file_ = "data/108801_106864/sl.npy"

    ticc_data = prepare_data(distance_file_, rssi_file_, sd_file_, sl_file_)
    print(ticc_data.shape)

    distance_npy_, rssi_npy_, sd_npy_, sl_npy_ = load_data(distance_file_, rssi_file_, sd_file_, sl_file_)
    # np.savetxt('distance', distance_npy_, fmt='%f')
    # np.savetxt('sd', sd_npy_, fmt='%f')

    print(distance_npy_.shape)
    plot_test(distance_npy_, rssi_npy_, sd_npy_, sl_npy_)

    set_data_ = get_time_stamp_set(distance_npy_, rssi_npy_, sd_npy_, sl_npy_)
    start_time_, end_time_ = get_start_end(distance_npy_, rssi_npy_, sd_npy_, sl_npy_)
    print(start_time_, end_time_)

    d_, sd_, sl_, rssi_ = process_data(distance_npy_, sd_npy_, sl_npy_, rssi_npy_, start_time_, end_time_)
    plot_test_1(d_, sd_, sl_)

    ks_data = np.array([d_[:, 1], sd_[:, 1], sl_[:, 1]])
    ks_data = ks_data.T
    start, end, inter = get_start_end_interval(set_data_)
    print(start, end, inter)
