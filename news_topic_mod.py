# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 00:34:59 2017

@author: sahebsingh
"""

from __future__ import print_function

""" Import all Necessary Modules. """

from scipy.spatial import distance
import numpy as np
from gensim import corpora, models, matutils
import sklearn.datasets
import nltk.stem
from collections import defaultdict
import nltk.corpus
from stop_words import get_stop_words

class DirectText(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

""" Specifying Stemmers and Stop Words. """

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(get_stop_words('english'))

""" Importing the Dataset """

dataset = sklearn.datasets.fetch_20newsgroups(subset = "test")

otexts = dataset.data
texts = dataset.data

""" Preprocessing data """

texts = [t.split() for t in texts]
# Converting texts to lower case
texts = [map(lambda w: w.lower(), t) for t in texts]
# Filtering texts which has set("+-.?!()>@012345689") in it.
texts = [filter(lambda s: not len(set("+-.?!()>@012345689") & set(s)), t)
         for t in texts]
texts = [filter(lambda s: (len(s) > 3) and (s not in stopwords), t) 
        for t in texts]
texts = [map(english_stemmer.stem, t) for t in texts]


""" Storing Words as a Dictionary. """

usage = defaultdict(int)

for t in texts:
    for w in t:
        usage[w] += 1        
        
""" Filtering words """

limit = len(texts)/10
too_common = [w for w in usage if usage[w]>limit]
too_common = set(too_common)
texts = [filter(lambda s: s not in too_common, t) for t in texts]

""" Creating lda topic model """           

corpus = corpora.BleiCorpus('/Users/sahebsingh/Documents/books/Machine Learning/chap4/ap 2/ap.dat',
'/Users/sahebsingh/Documents/books/Machine Learning/chap4/ap 2/vocab.txt')

model = models.ldamodel.LdaModel(
                  corpus,
                  num_topics=100,
                  id2word=corpus.id2word,
                  alpha=None)

topics = matutils.corpus2dense(model[corpus], num_terms = model.num_topics)
pairwise = distance.squareform(distance.pdist(topics))
largest = pairwise.max()

for i in range(len(topics)):
    pairwise[i, i] = largest+1

print(otexts[1])
print()
print()
print()
print(otexts[pairwise[1].argmin()])

        


           
















































