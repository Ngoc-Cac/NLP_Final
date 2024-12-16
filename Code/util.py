import os.path as osp

import numpy as np

from string import punctuation
from collections import Counter

from sklearn.metrics import silhouette_samples
from pyvi.ViTokenizer import tokenize
from wordcloud import WordCloud

with open(osp.join('..', 'Misc', 'vietnamese-stopwords.txt'), encoding='utf-8') as file:
    stopwords = list(i[:-1] for i in file.readlines())
    
punctuation += '...'
def viet_tokenize(article: str) -> list[str]:
  tokens = []
  text = article.replace('\n', ' ')
  for token in tokenize(text).split():
    if (token in punctuation) or\
        token.isnumeric():
       continue
    tokens.append(token.lower().replace('_', ' '))
  return tokens


def draw_wordcloud(articles: list[str], ax, **kwargs):
    wordcloud = WordCloud(**kwargs)

    words = [i for article in articles 
             for i in viet_tokenize(article)
             if i not in stopwords]
    frequencies_dict = Counter(words)
    wordcloud.generate_from_frequencies(frequencies_dict)

    ax.imshow(wordcloud)
    ax.axis("off")

    return ax

def draw_silhouette(ax, data, labels, title):
    y_lower = 10
    silhouette_sum = 0
    total_sample = 0

    sil_coefs = silhouette_samples(data, labels)

    for cluster in np.unique(labels):
        cluster_sil = sil_coefs[labels == cluster].copy()
        cluster_sil.sort()
        sample_size = cluster_sil.shape[0]

        silhouette_sum += np.sum(cluster_sil)
        total_sample += sample_size

        y_upper = y_lower + sample_size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         alpha=.9)
        ax.text(-0.05, y_lower + 0.5 * sample_size, str(cluster))
        
        y_lower = y_upper + 10

    sil_score = silhouette_sum / total_sample
    ax.axvline(x=sil_score, color=(1, 0, 0), linestyle='--')
    ax.text(sil_score + .01, 0, f'{sil_score:.2f}')

    ax.set_xlabel('Silhouette coefficient')
    ax.set_ylabel('Cluster')

    ax.set_yticks([])
    ax.set_title(title)

    return ax