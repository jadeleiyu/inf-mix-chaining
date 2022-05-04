import pickle
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import gzip
from nltk import WordNetLemmatizer


def noun_embeddings_pca(data_dir, dims):
    for decade in tqdm(range(1860, 2000, 10)):
        noun_histwords_embeddings = np.load(
            data_dir + 'histwords/noun_histwords_embeddings_{}.npy'.format(decade)
        )
        for dim in dims:
            pca = PCA(n_components=dim)
            explained_ratio = pca.explained_variance_ratio_.sum()
            noun_histwords_embeddings_reduced = pca.fit_transform(noun_histwords_embeddings)

            print('decade {}, PCA with {} components explains {} of the variance'.format(
                decade, dim, explained_ratio
            ))

            np.save(noun_histwords_embeddings_reduced,
                    data_dir + 'histwords/noun_histwords_embeddings_{}_{}.npy'.format(decade, dim))


def get_frame_data_from_gsn(gsn_file_id, gsn_dir, candidate_nouns, candidate_verbs,
                            rel_types, preps, start_year=1800, end_year=2000):
    lemmatizer = WordNetLemmatizer()
    frame_df = {
        'verb': [],
        'relation': [],
        'noun': [],
    }

    for decade in range(start_year, end_year + 10, 10):
        col_name = 'count in {}s'.format(str(decade))
        frame_df[col_name] = []

    with gzip.open(gsn_dir + 'verb_args_{}.gz'.format(gsn_file_id), 'rb') as f:
        content = f.read()

    for line in tqdm(content.decode("utf-8").split('\n'), position=0, leave=True):
        try:
            line_stats = line.split('\t')
            head_verb = lemmatizer.lemmatize(line_stats[0], pos='v')
            if head_verb in candidate_verbs:
                syntactic_ngram = line_stats[1]
                tagged_words = syntactic_ngram.split(' ')
                words_with_pos = [ngram_lemmatize(lemmatizer, tagged_word) for tagged_word in
                                  tagged_words]
                if words_with_pos:
                    nouns = [word for (word, pos) in words_with_pos if
                             word in candidate_nouns and pos[0] == 'N']
                    if nouns:
                        noun = nouns[0]
                        words = [t[0] for t in words_with_pos]
                        noun_idx = words.index(noun)
                        relation = rel_extract(tagged_words, noun_idx, rel_types, preps)
                        if relation:
                            frame_df['verb'].append(head_verb)
                            frame_df['noun'].append(noun)
                            frame_df['relation'].append(relation)
                            counts_by_decade = [0] * 21
                            for x in line_stats[3:]:
                                year = int(x.split(',')[0])
                                if year >= start_year:
                                    decade_idx = int((year - start_year) / 10)
                                    count = int(x.split(',')[1])
                                    counts_by_decade[decade_idx] += count
                            for k in range(len(counts_by_decade)):
                                decade = start_year + k * 10
                                col_name = 'count in {}s'.format(str(decade))
                                frame_df[col_name].append(counts_by_decade[k])
        except Exception as e:
            pass
    frame_df = pd.DataFrame(frame_df)
    return frame_df


def ngram_lemmatize(lemmatizer, tagged_word):
    if tagged_word.split('/')[1]:
        if tagged_word.split('/')[1][0] == 'V':
            return lemmatizer.lemmatize(tagged_word.split('/')[0], pos='v'), tagged_word.split('/')[1]
        elif tagged_word.split('/')[1][0] == 'N':
            return lemmatizer.lemmatize(tagged_word.split('/')[0], pos='n'), tagged_word.split('/')[1]
        else:
            return tagged_word.split('/')[0], tagged_word.split('/')[1]


def rel_extract(tagged_words, noun_idx, rel_types, preps):
    # as/IN/mark/3 water/NN/nsubj/3 covers/VBZ/advcl/0 the/DT/det/5 sea/NN/dobj/3
    # ['as/IN/mark/3', 'water/NN/nsubj/3', 'covers/VBZ/advcl/0',
    # 'for/IN/prep/3', 'the/DT/det/5', 'sea/NN/dobj/4']
    relation = []
    # words = [tagged_word.split('/')[0] for tagged_word in tagged_words]
    current_idx = noun_idx
    current_node = tagged_words[current_idx]
    next_idx = int(current_node.split('/')[-1]) - 1
    while next_idx != -1:
        rel = current_node.split('/')[-2]
        prep = current_node.split('/')[0]
        if rel in rel_types:
            relation.append(current_node.split('/')[-2])
        elif rel == 'prep' and prep in preps:
            relation.append('.'.join([current_node.split('/')[-2], current_node.split('/')[0]]))
        else:
            return None
        current_idx = next_idx
        current_node = tagged_words[current_idx]
        next_idx = int(current_node.split('/')[-1]) - 1
    if len(relation) == 1:
        return relation[0]
    elif len(relation) == 2:
        return '_'.join(relation)
    return None


def eval_dfs_prep(decade, grouped_frame_df, noun_vocab,
                  min_frame_total_count=50000,
                  min_n_support=10,
                  max_n_support=1000,
                  min_support_noun_cur_count=100):

    eval_df = {
        'frame': [],
        'support nouns': [],
        'support noun count up to {}s'.format(decade): [],
        'attested nouns': []
    }

    for index, row in grouped_frame_df.iterrows():
        final_decade_acc_counts = eval(row["total counts up to {}s".format(2000)])
        frame_total_count = sum(final_decade_acc_counts)
        if frame_total_count >= min_frame_total_count:
            nouns = eval(row['noun'])

            support_nouns_f = []
            support_noun_counts_f = []
            attested_nouns_f = []
            cur_decade_acc_counts = eval(row["total counts up to {}s".format(decade)])
            for j in range(len(nouns)):
                if nouns[j] in noun_vocab:
                    attested_nouns_f.append(nouns[j])
                    if cur_decade_acc_counts[j] >= min_support_noun_cur_count:
                            support_nouns_f.append(nouns[j])
                            support_noun_counts_f.append(cur_decade_acc_counts[j])
            sorted_support_noun_idx = np.argsort(-np.array(support_noun_counts_f))
            if len(support_nouns_f) >= min_n_support:
                if len(support_nouns_f) > max_n_support:
                    support_nouns_f = [support_nouns_f[j] for j in sorted_support_noun_idx[:max_n_support]]
                    support_noun_counts_f = [support_noun_counts_f[j] for j in sorted_support_noun_idx[:max_n_support]]
                frame = '-'.join([row['verb'], row['relation']])
                eval_df['frame'].append(frame)
                eval_df['support nouns'].append(support_nouns_f)
                eval_df['support noun count up to {}s'.format(decade)].append(support_noun_counts_f)
                eval_df['attested nouns'].append(attested_nouns_f)

    eval_df = pd.DataFrame(eval_df)
    return eval_df


def main_gsn():
    gsn_file_id = int(sys.argv[1])
    data_dir = '/home/jingyihe/scratch/inf_mix_chaining/'
    gsn_dir = data_dir + 'gsn/'
    noun_threshold = 20000
    verb_threshold = 200000
    candidate_nouns = pickle.load(open(
        data_dir + 'eval_dataframes/gsn_candidate_nouns_{}.p'.format(noun_threshold), 'rb'))
    candidate_verbs = pickle.load(open(
        data_dir + 'eval_dataframes/gsn_candidate_verbs_{}.p'.format(verb_threshold), 'rb'))

    rel_types = {'nsubj', 'dobj', 'iobj', 'pobj'}
    preps = {'in', 'by', 'to', 'with', 'on', 'from', 'for', 'at', 'as', 'like', 'of', 'into', 'about', 'under'}

    df = get_frame_data_from_gsn(gsn_file_id, gsn_dir, candidate_nouns, candidate_verbs,
                                 rel_types, preps)
    df.to_csv(gsn_dir + 'gsn_ungrouped_df_{}.csv'.format(gsn_file_id), index=False)


def main_eval_df():
    decade = int(sys.argv[1])
    data_dir = '/home/jingyihe/scratch/inf_mix_chaining/'
    grouped_frame_df = pd.read_csv(data_dir + 'gsn/gsn_grouped_df.csv')
    noun_vocab = set(pickle.load(open(data_dir + 'gsn_top_5000_nouns.p', 'rb')))
    eval_df = eval_dfs_prep(decade, grouped_frame_df, noun_vocab)
    eval_df.to_csv(data_dir + 'eval_dataframes/evaluation_df_{}.csv'.format(decade), index=False)


if __name__ == '__main__':
    main_eval_df()
