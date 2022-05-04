import pickle
import sys
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir = '/home/jingyihe/scratch/inf_mix_chaining/'
dp_lambdas = np.arange(0., 52., 2.) / 100


# todo: fix bug when having two data points with identical embeddings

def dp_means_clustering(X_support, dp_lambda):
    """
    DP-Means algorithm of clustering embedded support arguments of a frame.
    :param X_support: embedded support arguments of shape (N_support_args, embedding_dim)
    :param dp_lambda: hyper-parameter controlling the tradeoff between memory cost and recovering cost
    :return: mus: mixture centroids of shape (N_mix_centroids, embedding_dim)
    """

    # initialize a uniform cluster label for all support arguments
    cluster_idx = np.zeros(len(X_support))
    old_cluster_idx = np.ones(len(X_support))
    mus = X_support.mean(0).reshape(1, -1)
    n_clusters = 1
    n_iters = 0

    while np.sum(cluster_idx - old_cluster_idx) != 0:
        old_cluster_idx = np.copy(cluster_idx)
        for i in range(len(X_support)):
            x_i = X_support[i]
            sq_dists = np.square(x_i - mus).sum(1)  # shape (n_clusters)
            min_sq_dist_cluster_idx = np.argmin(sq_dists)
            min_sq_dist = sq_dists[min_sq_dist_cluster_idx]
            if min_sq_dist > dp_lambda:
                # create a new cluster unless we've got a distinct label for each x
                if n_clusters < len(X_support):
                    new_mu = x_i
                    cluster_idx[i] = len(mus)
                    mus = np.concatenate([mus, new_mu.reshape(1, -1)], axis=0)
            else:
                cluster_idx[i] = min_sq_dist_cluster_idx
        cluster_idx = cluster_idx - np.min(cluster_idx)
        n_clusters = len(np.unique(cluster_idx))

        new_mus = []
        for i in range(n_clusters):
            mu_i = X_support[np.where(cluster_idx == i)].mean(0)
            new_mus.append(mu_i)

        mus = new_mus
        n_iters += 1

        # print('old clusters: ', old_cluster_idx)
        # print('new clusters: ', cluster_idx)
        # print(np.sum(cluster_idx - old_cluster_idx))

    return mus, cluster_idx, n_iters


def compute_silhouette_scores(noun_histwords_embeddings, noun2idx, decade, dp_lambdas):
    silhouette_scores = np.zeros(len(dp_lambdas))
    eval_df = pd.read_csv(data_dir + 'eval_dataframes_dropped/evaluation_df_{}.csv'.format(decade))

    for dp_lambda_idx in tqdm(range(len(dp_lambdas))):
        cluster_idx_dp = pickle.load(
            open(data_dir + 'cluster_idx_dropped/dp_means_cluster_idx_{}_{}.p'.format(decade, dp_lambda_idx), 'rb')
        )
        silhouette_scores_lambda = []
        for idx, row in eval_df.iterrows():
            support_nouns = eval(row['support nouns'])
            X_support = np.stack([noun_histwords_embeddings[noun2idx[noun]] for noun in support_nouns])

            n_unique_clusters = len(np.unique(cluster_idx_dp[idx]))
            if n_unique_clusters <= 1 or n_unique_clusters >= len(X_support):
                sil_f = -1.0
            else:
                sil_f = silhouette_score(X_support, cluster_idx_dp[idx])
            silhouette_scores_lambda.append(sil_f)
        silhouette_scores[dp_lambda_idx] = np.array(silhouette_scores_lambda).mean()

    return silhouette_scores


def main_1():
    decade = int(sys.argv[1])
    pca_dim = 30

    noun2idx = pickle.load(open(data_dir + 'histwords/unreduced/histwords_noun2idx.p', 'rb'))
    noun_histwords_embeddings = np.load(
        data_dir + 'histwords/reduced/noun_histwords_embeddings_{}_{}.npy'.format(decade, pca_dim)
    )
    eval_df = pd.read_csv(data_dir + 'eval_dataframes/evaluation_df_{}.csv'.format(decade))

    for i in tqdm(range(len(dp_lambdas))):
        dp_lambda = dp_lambdas[i]
        dp_cluster_idx = []
        for _, row in eval_df.iterrows():
            support_nouns = eval(row['support nouns'])
            X_support = np.stack([noun_histwords_embeddings[noun2idx[noun]] for noun in support_nouns])
            mus, cluster_idx, n_iters = dp_means_clustering(X_support, dp_lambda)
            dp_cluster_idx.append(cluster_idx)

        pickle.dump(dp_cluster_idx, open(
            data_dir + 'cluster_idx/dp_means_cluster_idx_{}_{}.p'.format(decade, i), 'wb')
                    )


def main_2():
    decade = int(sys.argv[1])
    pca_dim = 30
    noun_histwords_embeddings = np.load(
        data_dir + 'histwords/reduced/noun_histwords_embeddings_{}_{}.npy'.format(decade, pca_dim)
    )
    noun2idx = pickle.load(open(data_dir + 'histwords/unreduced/histwords_noun2idx.p', 'rb'))
    silhouette_scores = compute_silhouette_scores(noun_histwords_embeddings, noun2idx, decade, dp_lambdas)
    print(silhouette_scores)
    np.save(data_dir + 'silhouette/dp_means_silhouette_scores_{}.p'.format(decade), silhouette_scores)


if __name__ == '__main__':
    main_1()
    # main_2()
