import numpy as np
from dp_means import dp_means_clustering


def inf_mix_chaining(X_query, frame_mus, dp_lambda, knn=1):
    """
    Infinite mixture chaining algorithm to assign a novel argument to the most appropriate frame.
    :param X_query: embeddings of the query arguments
    :param frame_mus: list of support argument centroids for every frame f,
    each array has shape (N_support_f, embedding_dim)
    :param dp_lambda: hyper-parameter controlling the tradeoff between memory cost and recovering cost
    :param knn: number of nearest neighbors to consider during inference
    :return:
    """

    frame_weights = np.zeros((len(X_query), len(frame_mus)))
    centroid_idx = []
    for i in range(len(X_query)):
        x_query = X_query[i]
        centroid_idx_i = []
        for j in range(len(frame_mus)):
            mus_j = frame_mus[j]
            sq_dists = np.square(x_query - mus_j).sum(1)
            knn_sq_dist_id = np.argsort(sq_dists)[:knn]
            min_sq_dist_id = knn_sq_dist_id[0]
            mean_knn_sq_dists = sq_dists[knn_sq_dist_id].mean()
            if mean_knn_sq_dists < dp_lambda:
                frame_weights[i, j] = np.exp(-mean_knn_sq_dists)
                centroid_idx_i.append(min_sq_dist_id)
            else:
                frame_weights[i, j] = np.exp(-dp_lambda)
                centroid_idx_i.append(-1)
        centroid_idx.append(np.array(centroid_idx_i))

    return frame_weights, centroid_idx


# def exemplar_chaining(x_query, X_support, frame_counts, k_nn=1):
#     """
#     Exemplar chaining method with k-nearest-neighbor inference.
#     :param x_query:
#     :param X_support:
#     :param frame_counts:
#     :param k_nn:
#     :return:
#     """
#     frame_weights = []
#     knn_idx = []
#     for i in range(len(X_support)):
#         X_support_i = X_support[i]
#         qs_dists = np.square(x_query - X_support_i).sum(1)
#         sorted_X_support_idx = np.argsort(qs_dists)
#         k_nn_qs_dists = qs_dists[:min(k_nn, len(X_support_i))]
#         knn_X_support_idx = sorted_X_support_idx[:k_nn]
#         knn_idx.append(knn_X_support_idx)
#         frame_weight = np.sum(np.exp(-k_nn_qs_dists)) * frame_counts[i]
#         frame_weights.append(frame_weight)
#
#     frame_weights = np.array(frame_weights)
#     predicted_frame_idx = np.argmax(frame_weights)
#
#     return predicted_frame_idx, frame_weights, knn_idx


