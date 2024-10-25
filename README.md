This repository includes code and data for analyses in the following work:

Yu, L. and Xu, Y. (in press) Infinite mixture chaining: An
efficiency-based framework for the dynamic construction of word meaning.
Open Mind: Discoveries in Cognitive Science.

## English verb-noun compositions
Python files to reproduce experiment results in Case Study 1 (English verb-noun compositions) can be found in the ```verb/``` folder.
To run experiments:
1. Run ```verb/gsn_download.py``` to download Google Syntactic Ngram data.
2. Run ```verb/data_prep.py``` to generate evaluation dataframes of verb-noun composition usages for each decade.
3. Run ```inf-mix/dp_means.py``` to infer semantic cluster centroids for verbs across all decades.
4. Run ```inf-mix/chaining.py``` to predict novel pairings of query nouns and head verbs via inf-mix chaining.
5. Run ```inf-mix/baselines.py``` to predict novel pairings of query nouns and head verbs via baseline models.
6. Run ```figures/plot-results.ipynb``` to produce figures of decade-wise predictive accuracy and the accuracy-complexity trade-off plot.

## English adjective-noun compositions
Python files to reproduce experiment results in Case Study 2 (English adjective-noun compositions) can be found in the ```adjective/``` folder.
To run experiments:
1. Run ```adjective/gsn_download.py``` to download Google Syntactic Ngram data.
2. Run ```adjective/data_prep.py``` to generate evaluation dataframes of adjective-noun composition usages for each decade.
3. Run ```inf-mix/dp_means.py``` to infer semantic cluster centroids for adjective across all decades.
4. Run ```inf-mix/chaining.py``` to predict novel pairings of query nouns and head adjectivea via inf-mix chaining.
5. Run ```inf-mix/baselines.py``` to predict novel pairings of query nouns and head adjectivea via baseline models.
6. Run ```figures/plot-results.ipynb``` to produce figures of decade-wise predictive accuracy and the accuracy-complexity trade-off plot.

## Chinese classifier-noun compositions
Python files to reproduce experiment results in Case Study 3 (Chinese classifier-noun compositions) can be found in the ```classifier/``` folder.
To run experiments:
1. Run ```classifier/evaluate.py``` to predict novel pairings of query nouns and head adjectivea via inf-mix chaining and baseline models. Historical data of noun-classifier usages by (Habibi et al., 2020) can be found in ```super_words-chi-Luis-YbyY(w2v-en).pkl```.
2. Run ```figures/plot-results.ipynb``` to produce figures of decade-wise predictive accuracy and the accuracy-complexity trade-off plot.


## Diachronic lexical semantic change
Code and data to reproduce results in Case Study 4 can be found in the ```lsc/``` folder.
To reproduce results in Case Study 4, follow ```lsc/readme.txt```.

All result figures can be found in the ```figures``` folder, together with a ```plot-results.ipynb``` to produce these figures.

## Citation
```
@article{yu2024infmix,
  title={Infinite mixture chaining: A probabilistic framework for the adaptive construction of word meaning},
  author={Lei Yu and Yang Xu},
  journal={Open Mind (To appear)},
  year={2024}
}
```
