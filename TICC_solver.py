import collections
import errno
import multiprocessing
import os
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import pandas as pd
from sklearn import mixture
from TICC_helper import *
from sklearn_GraphicalLassoCV import get_GraphicalLassoCV_for_multiprocessing


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5,
                 beta=400, maxIters=1000, write_out_file=False,
                 prefix_string="", compute_BIC=True, cluster_reassignment=20,
                 biased=False, visible=True, convergence=0.95):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - write_out_file: (bool) if true, prefix_string is output file dir
            - prefix_string: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.clusters_bias = 1000
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased
        self.visible = visible
        self.trained_model = None
        self.times_series_arr = None
        self.convergence = convergence
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit(self, times_series_arr):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.maxIters > 0  # must have at least one iteration

        # Get data into proper format
        # self.times_series_arr, time_series_rows_size, time_series_col_size = load_data_from_dir(input_file)
        self.times_series_arr = times_series_arr
        time_series_rows_size, time_series_col_size = times_series_arr.shape
        ############
        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split, time_series_rows_size:有多少行数据
        training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks,
                                             self.window_size)  # indices of the training samples
        # print(training_indices)
        num_train_points = len(training_indices)

        # Stack the training data，根据窗口整理数据数据
        complete_D_train = self.stack_training_data(self.times_series_arr, time_series_col_size, num_train_points,
                                                    training_indices)

        # Initialization
        # Gaussian Mixture
        # E step init
        gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full", reg_covar=1e-4)
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        # K-means
        # kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0).fit(complete_D_train)
        # clustered_points_kmeans = kmeans.labels_  # todo, is there a difference between these two?
        # kmeans_clustered_pts = kmeans.labels_

        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration
        empirical_covariances = None  # 经验协方差
        old_computed_covariance = None
        train_cluster_inverse = None

        # PERFORM TRAINING ITERATIONS
        for iters in range(self.maxIters):
            print("ITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)  # {cluster(标签): [point indices](数据索引)}
            for point, cluster_num in enumerate(clustered_points):  # 使用GMM初始化的结果
                train_clusters_arr[cluster_num].append(point)  # 每个类对应的数据片段是那个(index)
            # 每个类的数据量有多少
            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            # M step
            temp_train_cluster_inverse, computed_covariance, temp_empirical_covariances = \
                self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train, len_train_clusters,
                                    time_series_col_size, train_clusters_arr)

            # update old computed covariance
            if old_computed_covariance is None:
                old_computed_covariance = computed_covariance
            else:
                for key in computed_covariance.keys():
                    if computed_covariance[key] is not None:
                        old_computed_covariance[key] = computed_covariance[key]

            # update train cluster inverse
            if train_cluster_inverse is None:
                train_cluster_inverse = temp_train_cluster_inverse
            else:
                for cluster in range(len(train_cluster_inverse)):
                    if temp_train_cluster_inverse[cluster] is not None:
                        train_cluster_inverse[cluster] = temp_train_cluster_inverse[cluster]

            # update empirical covariances
            if empirical_covariances is None:
                empirical_covariances = temp_empirical_covariances
            else:
                for cluster in range(len(empirical_covariances)):
                    if cluster in temp_empirical_covariances.keys() and temp_empirical_covariances[cluster] is not None:
                        empirical_covariances[cluster] = temp_empirical_covariances[cluster]

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  # 'computed_covariance': computed_covariance,  # 计算出的协方差
                                  'computed_covariance': old_computed_covariance,  # 计算出的协方差
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,  # 不同族逆协方差对应的均值
                                  'complete_D_train': complete_D_train,  # 分窗好的数据
                                  'time_series_col_size': time_series_col_size}  # 数据长度
            # E step
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                 range(self.number_of_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                  (self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]

            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            # if np.array_equal(old_clustered_points, clustered_points):
            if old_clustered_points is not None and sum(old_clustered_points == clustered_points) * 1.0 / len(
                    old_clustered_points) > self.convergence:
                print("CONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if self.visible:
            self.write_plot(clustered_points, str_NULL)

        if self.compute_BIC:
            bic = computeBIC(self.number_of_clusters, time_series_rows_size, clustered_points, train_cluster_inverse,
                             empirical_covariances)
            return clustered_points, train_cluster_inverse, bic

        return clustered_points, train_cluster_inverse

    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,self.num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,self.num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,self.num_clusters)
        print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def write_plot(self, clustered_points, str_NULL):
        # Save a figure of segmentation
        plt.figure()
        times_series_arr_T = self.times_series_arr.T
        c = 0
        for _ in times_series_arr_T:
            c += 0.05
            plt.scatter(range(len(_) - self.window_size + 1),
                        np.zeros(len(_) - self.window_size + 1) + self.clusters_bias, s=10, c=clustered_points,
                        marker='s')
            plt.plot(range(len(_)), _, color=(c, c, c))

            temp = _[0:len(clustered_points)]
            plt.scatter(range(len(temp)), temp, s=50, c=clustered_points, marker='s')

        if self.write_out_file:
            plt.savefig(
                str_NULL + "switch_penalty = " + str(self.switch_penalty) + ".jpg")
        plt.show()
        plt.close("all")
        print("Done writing the figure")

    def smoothen_clusters(self, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        # print("computed_covariance:", computed_covariance)
        for cluster in range(self.number_of_clusters):
            # print("cluster:", cluster)
            cov_matrix = computed_covariance[
                             self.number_of_clusters, cluster
                         ][
                         0:(self.num_blocks - 1) * n, 0:(self.num_blocks - 1) * n
                         ]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in range(self.number_of_clusters):
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]  # 减去均值
                    inv_cov_matrix = inv_cov_dict[cluster]  # 逆协方差
                    log_det_cov = log_det_dict[cluster]  # log det (Θ)
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),  # ll(X_t ,Θ_i)是Xt来自簇 i 的对数似然值
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle
        print("end the smoothening ALGORITHM")
        return LLE_all_points_clusters

    # 计算每个族的逆协方差矩阵Θ
    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                       len_train_clusters, n, train_clusters_arr):
        optRes = multiprocessing.Manager().list()  # 所有的逆协方差矩阵
        for i in range(self.number_of_clusters):
            optRes.append(None)

        temp_computed_covariance = multiprocessing.Manager().dict()
        for cluster in range(self.number_of_clusters):
            temp_computed_covariance[self.number_of_clusters, cluster] = None

        job_process = []
        empirical_covariances = {}
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]  # 属于第cluster类的数据有多少个
            if cluster_length != 0:
                indices = train_clusters_arr[cluster]  # 当前类下的所有index
                if cluster_length >= self.cluster_reassignment:
                    D_train = np.zeros([cluster_length, self.window_size * n])  # X[window_size*传感器个数]
                else:
                    lack_num = self.cluster_reassignment - cluster_length
                    D_train = np.zeros([cluster_length + lack_num, self.window_size * n])  # X[window_size*传感器个数]
                    point = indices[0]
                    for i in range(1, lack_num + 1):
                        if point + i < complete_D_train.shape[0]:
                            D_train[i + cluster_length - 1, :] = complete_D_train[point + i, :]
                        else:
                            D_train[i + cluster_length - 1, :] = complete_D_train[point - i, :]

                # D_train = complete_D_train[indices] 浅拷贝有bug
                for i in range(cluster_length):  # 不同类别的数据分别整理
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.number_of_clusters, cluster] = \
                    np.mean(D_train, axis=0)[(self.window_size - 1) * n:self.window_size * n].reshape([1, n])

                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)

                empirical_covariances[cluster] = np.cov(np.transpose(D_train), bias=self.biased)
                # print("D_train:", D_train.shape, cluster_length)
                job_process.append(multiprocessing.Process(target=get_GraphicalLassoCV_for_multiprocessing, args=(
                    D_train, optRes, temp_computed_covariance, cluster, self.number_of_clusters),
                                                           name="进程-%s" % cluster))

        for p in job_process:
            p.start()
        for p in job_process:
            p.join()
        # print("train_clusters:", temp_computed_covariance)
        return optRes, temp_computed_covariance, empirical_covariances

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def prepare_out_directory(self):
        str_NULL = self.prefix_string + "lam_sparse=" + "maxClusters=" + str(
            self.number_of_clusters + 1) + "/"
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL))
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return str_NULL

    def predict_clusters(self, test_data=None):
        """
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        """
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING, 状态矩阵
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        (T, num_clusters) = lle_all_points_clusters.shape
        switch_penalty_trans = gen_switch_penalty_trans(num_clusters, self.switch_penalty)  # 转移矩阵
        clustered_points = viterbi_decode_updateClusters(lle_all_points_clusters, switch_penalty_trans)

        return clustered_points
