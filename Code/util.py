import os.path as osp

from string import punctuation

from pyvi import ViTokenizer

with open(osp.join('..', 'Misc', 'vietnamese-stopwords.txt'), encoding='utf-8') as file:
    stopwords = list(i[:-1] for i in file.readlines())
    
punctuation += '...'
def viet_tokenize(article: str) -> list[str]:
  tokens = []
  text = article.replace('\n', '')
  for token in ViTokenizer.tokenize(text).split():
    if (token in punctuation) or\
        token.isnumeric():
       continue
    tokens.append(token.lower().replace('_', ' '))
  return tokens