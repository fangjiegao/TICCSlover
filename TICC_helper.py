import numpy as np


def getTrainTestSplit(m, num_blocks, num_stacked):
    """
    - m: number of observations有多少行数据
    - num_blocks: window_size + 1
    - num_stacked: window_size
    Returns:
    - sorted list of training indices
    """
    # Now splitting up stuff
    # split1 : Training and Test
    # split2 : Training and Test - different clusters
    training_percent = 1
    # list of training indices  从0-m-num_blocks+1中随机选择(m-num_stacked)*training_percent个数据
    training_idx = np.random.choice(
        m - num_blocks + 1, size=int((m - num_stacked) * training_percent), replace=False)
    # Ensure that the first and the last few points are in
    training_idx = list(training_idx)
    if 0 not in training_idx:
        training_idx.append(0)
    if m - num_stacked not in training_idx:
        training_idx.append(m - num_stacked)
    training_idx = np.array(training_idx)
    return sorted(training_idx)


def viterbi_decode_updateClusters(LLE_node_vals, switch_penalty_trans, max_or_min="min"):
    """
    Viterbi算法求最优路径
    其中 LLE_node_vals.shape=[seq_len, num_labels],
        switch_penalty_trans.shape=[num_labels, num_labels].
    """
    seq_len, num_labels = len(LLE_node_vals), len(switch_penalty_trans)
    labels = np.arange(num_labels).reshape((1, -1))
    scores = LLE_node_vals[0].reshape((-1, 1))
    paths = labels
    if max_or_min == "min":
        for t in range(1, seq_len):
            observe = LLE_node_vals[t].reshape((1, -1))
            M = scores + switch_penalty_trans + observe
            scores = np.min(M, axis=0).reshape((-1, 1))
            idxs = np.argmin(M, axis=0)
            paths = np.concatenate([paths[:, idxs], labels], 0)
        best_path = paths[:, scores.argmin()]
        return best_path
    if max_or_min == "max":
        for t in range(1, seq_len):
            observe = LLE_node_vals[t].reshape((1, -1))
            M = scores + switch_penalty_trans + observe
            scores = np.max(M, axis=0).reshape((-1, 1))
            idxs = np.argmax(M, axis=0)
            paths = np.concatenate([paths[:, idxs], labels], 0)
        best_path = paths[:, scores.argmax()]
        return best_path


def gen_switch_penalty_trans(clusters_num, switch_penalty):
    switch_penalty_trans = np.ones(shape=(clusters_num, clusters_num), dtype=float)
    switch_penalty_trans = switch_penalty_trans * switch_penalty
    row, col = np.diag_indices_from(switch_penalty_trans)
    switch_penalty_trans[row, col] = 0
    return switch_penalty_trans


def computeBIC(K, T, clustered_points, inverse_covariances, empirical_covariances):
    """
    empirical covariance and inverse_covariance should be dicts
    K is num clusters
    T is num samples
    """
    mod_lle = 0
    threshold = 2e-5
    clusterParams = {}
    # for cluster, clusterInverse in inverse_covariances.items():
    for cluster in range(len(inverse_covariances)):
        clusterInverse = inverse_covariances[cluster]
        mod_lle += np.log(np.linalg.det(clusterInverse)) - np.trace(
            np.dot(empirical_covariances[cluster], clusterInverse))
        clusterParams[cluster] = np.sum(np.abs(clusterInverse) > threshold)
    curr_val = -1
    non_zero_params = 0
    for val in clustered_points:
        if val != curr_val:
            non_zero_params += clusterParams[val]
            curr_val = val
    return non_zero_params * np.log(T) - 2 * mod_lle
