import pickle
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def baseline_evaluation(decade, noun2idx, noun_vocab, noun2count,
                        noun_histwords_embeddings, n_samples_noun=100):
    eval_df = pd.read_csv(data_dir + 'eval_dataframes/evaluation_df_{}.csv'.format(decade))

    n_frames = len(eval_df['frame'])

    accs_proto = np.ones(n_frames) * (-1.0)
    accs_exemplar = np.ones(n_frames) * (-1.0)
    accs_1nn = np.ones(n_frames) * (-1.0)
    accs_random = np.ones(n_frames) * (-1.0)

    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df['frame'])):
        try:
            query_nouns = eval(row['query nouns'])
            support_nouns = eval(row['support nouns'])
            support_noun_counts = np.array(eval(row['support noun count up to {}s'.format(decade)]))
            attested_nouns = set(eval(row['attested nouns']))
            unattested_nouns = [n for n in noun_vocab if n not in attested_nouns]
            X_support = np.stack([noun_histwords_embeddings[noun2idx[noun]] for noun in support_nouns])

            frame_proto_mu = X_support.mean(0)
            accs_proto_i = []
            accs_exemplar_i = []
            accs_1nn_i = []
            accs_random_i = []

            for _ in range(n_samples_noun):
                samlped_negative_nouns = random.sample(unattested_nouns, len(query_nouns))
                candidate_nouns = query_nouns + samlped_negative_nouns

                ws_proto = []
                ws_exemplar = []
                ws_1nn = []
                ws_random = []
                for noun in candidate_nouns:
                    sq_dists_proto = np.square(frame_proto_mu - noun_histwords_embeddings[noun2idx[noun]]).sum(0)
                    sq_dists_exemplar = np.square(X_support - noun_histwords_embeddings[noun2idx[noun]]).sum(1)

                    ws_proto.append(np.exp(-sq_dists_proto))
                    ws_exemplar.append(np.mean(np.exp(-sq_dists_exemplar) * support_noun_counts))
                    ws_1nn.append(np.sort(np.exp(-sq_dists_exemplar) * support_noun_counts)[-1])
                    ws_random.append(np.random.rand())

                sorted_cand_idx_proto = np.argsort(-np.array(ws_proto))
                sorted_cand_idx_exemplar = np.argsort(-np.array(ws_exemplar))
                sorted_cand_idx_1nn = np.argsort(-np.array(ws_1nn))
                sorted_cand_idx_random = np.argsort(-np.array(ws_random))

                for k in range(1, len(candidate_nouns)):
                    accs_proto_i.append(
                        np.where(sorted_cand_idx_proto < len(query_nouns), 1., 0.)[:k].sum() / k)
                    accs_exemplar_i.append(
                        np.where(sorted_cand_idx_exemplar < len(query_nouns), 1., 0.)[:k].sum() / k)
                    accs_1nn_i.append(
                        np.where(sorted_cand_idx_1nn < len(query_nouns), 1., 0.)[:k].sum() / k)
                    accs_random_i.append(
                        np.where(sorted_cand_idx_random < len(query_nouns), 1., 0.)[:k].sum() / k)

            # if np.array(accs_dp_all_i).mean() <= acc_threshold or np.array(accs_dp_1nn_i).mean() <= acc_threshold:
            #     failed_frame_idx.append(i)
            # else:
            accs_proto[i] = np.array(accs_proto_i).mean()
            accs_exemplar[i] = np.array(accs_exemplar_i).mean()
            accs_1nn[i] = np.array(accs_1nn_i).mean()
            accs_random[i] = np.array(accs_random_i).mean()
        except Exception as e:
            pass

    return accs_proto, accs_exemplar, accs_1nn, accs_random


if __name__ == '__main__':
    data_dir = '/home/jingyihe/scratch/inf_mix_chaining/'

    decade = int(sys.argv[1])
    n_samples_noun = 200
    pca_dim = 30

    noun2idx = pickle.load(open(data_dir + 'histwords/unreduced/histwords_noun2idx.p', 'rb'))
    noun_histwords_embeddings = np.load(
        data_dir + 'histwords/reduced/noun_histwords_embeddings_{}_{}.npy'.format(decade, pca_dim)
    )
    noun2total_count = pickle.load(open(data_dir + 'gsn_noun2total_count.p', 'rb'))
    top_5000_nouns = pickle.load(open(data_dir + 'gsn_top_5000_nouns.p', 'rb'))
    noun_vocab = set(top_5000_nouns)

    accs_proto, accs_exemplar, accs_1nn, accs_random = baseline_evaluation(
        decade, noun2idx, noun_vocab, noun2total_count,
        noun_histwords_embeddings, n_samples_noun
    )

    print('decade {}'.format(decade))

    # print('decade {}, dp lambda: {}'.format(decade, dp_lambda))
    print('using 30d HistWords embeddings')
    print()

    print('AUC by prototype model: ', accs_proto.mean())
    print('AUC by exemplar model: ', accs_exemplar.mean())
    print('AUC by 1nn model: ', accs_1nn.mean())
    print('AUC by random baseline: ', accs_random.mean())

    np.save(data_dir + 'eval_results/acc_proto_{}.npy'.format(decade), accs_proto)
    np.save(data_dir + 'eval_results/acc_exemplar_{}.npy'.format(decade), accs_exemplar)
    np.save(data_dir + 'eval_results/acc_1nn_{}.npy'.format(decade), accs_1nn)
    np.save(data_dir + 'eval_results/acc_random_{}.npy'.format(decade), accs_random)
