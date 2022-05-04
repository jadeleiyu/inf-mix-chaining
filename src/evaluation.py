import pickle
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

dp_lambdas = np.arange(10., 52., 2.) / 100
decades = list(range(1860, 2000, 10))


def model_evaluation(dp_lambda_idx, decade, noun2idx, noun_vocab, noun2count,
                     noun_histwords_embeddings, n_samples_frame=-1, n_samples_noun=100):
    eval_df = pd.read_csv(data_dir + 'eval_dataframes/evaluation_df_{}.csv'.format(decade))
    cluster_idx_dp = pickle.load(
        open(data_dir + 'cluster_idx/dp_means_cluster_idx_{}_{}.p'.format(decade, dp_lambda_idx), 'rb')
    )

    if decade <= 1950:
        acc_threshold = 0.55
    else:
        acc_threshold = 0.50

    failed_frame_idx = []
    accs_dp_all = []
    accs_dp_1nn = []
    accs_proto = []
    accs_exemplar = []
    accs_1nn = []
    accs_freq = []
    accs_random = []
    n_clusters = []

    if n_samples_frame == -1:
        sampled_df_frac = eval_df
    else:
        sampled_df_frac = eval_df.sample(n=n_samples_frame, replace=True)
    for i, row in tqdm(sampled_df_frac.iterrows(), total=len(sampled_df_frac['frame'])):
        try:
            query_nouns = eval(row['query nouns'])
            support_nouns = eval(row['support nouns'])
            support_noun_counts = np.array(eval(row['support noun count up to {}s'.format(decade)]))
            attested_nouns = set(eval(row['attested nouns']))
            unattested_nouns = [n for n in noun_vocab if n not in attested_nouns]
            X_support = np.stack([noun_histwords_embeddings[noun2idx[noun]] for noun in support_nouns])
            frame_mus_dp_i = []
            frame_mus_counts_i = []
            n_clusters_i = len(np.unique(cluster_idx_dp[i]))
            n_clusters.append(n_clusters_i)

            for j in range(n_clusters_i):
                support_idx_i_j = np.where(cluster_idx_dp[i] == j)[0]
                frame_mus_dp_i.append(X_support[support_idx_i_j].mean(0))
                frame_mus_counts_i.append(support_noun_counts[support_idx_i_j].sum())
            frame_proto_mu = X_support.mean(0)
            frame_mus_dp_i = np.array(frame_mus_dp_i)

            accs_dp_all_i = []
            accs_dp_1nn_i = []
            accs_proto_i = []
            accs_exemplar_i = []
            accs_1nn_i = []
            accs_freq_i = []
            accs_random_i = []

            for _ in range(n_samples_noun):
                samlped_negative_nouns = random.sample(unattested_nouns, len(query_nouns))
                candidate_nouns = query_nouns + samlped_negative_nouns
                ws_dp_all = []
                ws_dp_1nn = []
                ws_proto = []
                ws_exemplar = []
                ws_1nn = []
                ws_freq = []
                ws_random = []
                for noun in candidate_nouns:
                    sq_dists_dp = np.square(frame_mus_dp_i - noun_histwords_embeddings[noun2idx[noun]]).sum(1)
                    # sq_dists_proto = np.square(frame_proto_mu - noun_histwords_embeddings[noun2idx[noun]]).sum(0)
                    # sq_dists_exemplar = np.square(X_support - noun_histwords_embeddings[noun2idx[noun]]).sum(1)

                    ws_dp_all.append(np.mean(np.exp(-sq_dists_dp) * frame_mus_counts_i))
                    ws_dp_1nn.append(np.sort(np.exp(-sq_dists_dp) * frame_mus_counts_i)[-1])
                    # ws_proto.append(np.exp(-sq_dists_proto))
                    # ws_exemplar.append(np.mean(np.exp(-sq_dists_exemplar) * support_noun_counts))
                    # ws_1nn.append(np.sort(np.exp(-sq_dists_exemplar) * support_noun_counts)[-1])
                    # ws_freq.append(noun2count[noun])
                    # ws_random.append(np.random.rand())

                sorted_cand_idx_dp_all = np.argsort(-np.array(ws_dp_all))
                sorted_cand_idx_dp_1nn = np.argsort(-np.array(ws_dp_1nn))
                # sorted_cand_idx_proto = np.argsort(-np.array(ws_proto))
                # sorted_cand_idx_exemplar = np.argsort(-np.array(ws_exemplar))
                # sorted_cand_idx_1nn = np.argsort(-np.array(ws_1nn))
                # sorted_cand_idx_freq = np.argsort(-np.array(ws_freq))
                # sorted_cand_idx_random = np.argsort(-np.array(ws_random))

                for k in range(1, len(candidate_nouns)):
                    accs_dp_all_i.append(
                        np.where(sorted_cand_idx_dp_all < len(query_nouns), 1., 0.)[:k].sum() / k)
                    accs_dp_1nn_i.append(
                        np.where(sorted_cand_idx_dp_1nn < len(query_nouns), 1., 0.)[:k].sum() / k)
                    # accs_proto_i.append(
                    #     np.where(sorted_cand_idx_proto < len(query_nouns), 1., 0.)[:k].sum() / k)
                    # accs_exemplar_i.append(
                    #     np.where(sorted_cand_idx_exemplar < len(query_nouns), 1., 0.)[:k].sum() / k)
                    # accs_1nn_i.append(
                    #     np.where(sorted_cand_idx_1nn < len(query_nouns), 1., 0.)[:k].sum() / k)
                    # accs_freq_i.append(
                    #     np.where(sorted_cand_idx_freq < len(query_nouns), 1., 0.)[:k].sum() / k)
                    # accs_random_i.append(
                    #     np.where(sorted_cand_idx_random < len(query_nouns), 1., 0.)[:k].sum() / k)

            # if np.array(accs_dp_all_i).mean() <= acc_threshold:
            #     failed_frame_idx.append(i)
            # else:
            accs_dp_all.append(np.array(accs_dp_all_i).mean())
            accs_dp_1nn.append(np.array(accs_dp_1nn_i).mean())
            # accs_proto.append(np.array(accs_proto_i).mean())
            # accs_exemplar.append(np.array(accs_exemplar_i).mean())
            # accs_1nn.append(np.array(accs_1nn_i).mean())
            # accs_freq.append(np.array(accs_freq_i).mean())
            # accs_random.append(np.array(accs_random_i).mean())
        except Exception as e:
            pass

    return np.array(accs_dp_all), np.array(accs_dp_1nn), np.array(accs_proto), np.array(accs_exemplar), \
           np.array(accs_1nn), np.array(accs_freq), np.array(accs_random), failed_frame_idx, np.array(n_clusters)


if __name__ == '__main__':
    data_dir = '/home/jingyihe/scratch/inf_mix_chaining/'

    job_id = int(sys.argv[1])
    dp_lambda_idx = job_id % 21
    decade_idx = int(job_id / 21)
    decade = decades[decade_idx]
    dp_lambda = dp_lambdas[dp_lambda_idx]

    n_samples_frame = -1
    n_samples_noun = 200
    pca_dim = 30
    # dp_lambda = 0.26
    # dp_lambda_idx = 8

    noun2idx = pickle.load(open(data_dir + 'histwords/unreduced/histwords_noun2idx.p', 'rb'))
    noun_histwords_embeddings = np.load(
        data_dir + 'histwords/reduced/noun_histwords_embeddings_{}_{}.npy'.format(decade, pca_dim)
    )
    noun2total_count = pickle.load(open(data_dir + 'gsn_noun2total_count.p', 'rb'))
    top_5000_nouns = pickle.load(open(data_dir + 'gsn_top_5000_nouns.p', 'rb'))
    noun_vocab = set(top_5000_nouns)

    accs_dp_all, accs_dp_1nn, accs_proto, accs_exemplar, accs_1nn, accs_freq, accs_random, failed_frame_idx, n_clusters \
        = model_evaluation(
        dp_lambda_idx, decade, noun2idx, noun_vocab, noun2total_count,
        noun_histwords_embeddings, n_samples_frame, n_samples_noun
    )

    print('decade {}, dp lambda: {}'.format(decade, dp_lambdas[dp_lambda_idx]))

    # print('decade {}, dp lambda: {}'.format(decade, dp_lambda))
    #     print('mean number of clusters per frame: {}'.format(n_clusters.mean()))
    print('using 30d HistWords embeddings')
    print()
    print('AUC by dp-all model: ', accs_dp_all.mean())
    print('AUC by dp-1nn model: ', accs_dp_1nn.mean())
    # print('AUC by prototype model: ', accs_proto.mean())
    # print('AUC by exemplar model: ', accs_exemplar.mean())
    # print('AUC by 1nn model: ', accs_1nn.mean())
    # print('AUC by frequency baseline: ', accs_freq.mean())
    # print('AUC by random baseline: ', accs_random.mean())

    # pickle.dump(
    #     failed_frame_idx,
    #     open(data_dir + 'failed_frames/failed_frame_idx_{}_{}.p'.format(decade, dp_lambda_idx), 'wb')
    # )
