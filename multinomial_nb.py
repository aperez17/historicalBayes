#!/usr/bin/env python

from __future__ import division

import math
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
import copy
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import BernoulliNB

from collections import defaultdict

# Global class labels.
GLOBAL_LABLES = {}

# Path to dataset
if len(sys.argv) < 3:
    print "YOU NEED TO SPECIFY YOUR LOCAL CORPUS DIRECTORY AHHHHH!!!"
    print "i.e. /Users/alexperez/Documents/cs580/project/historicalBayes/preProcessedCorpus"
    sys.exit(1)
PATH_TO_DATA = sys.argv[1]
NUM_PERIODS = int(sys.argv[2])

if NUM_PERIODS == 3:
    GLOBAL_LABLES = {
        'EARLY_TO_MID_1700': '1710-1780',
        'LATE_TO_MID_1800': '1780-1850',
        'MID_TO_EARLY_1900': '1850-1920'
    }
elif NUM_PERIODS == 5:
    GLOBAL_LABLES = {
        'EARLY_1700': '1710-1752',
        'LATE_1700': '1752-1794',
        'EARLY_1800': '1794-1836',
        'LATE_1800': '1836-1878',
        'EARLY_1900': '1878-1921'
    }
elif NUM_PERIODS == 7:
    GLOBAL_LABLES = {
        'EARLY_1700':'1710-1740',
        'MID_1700': '1740-1770',
        'LATE_1700': '1770-1800',
        'EARLY_1800': '1800-1830',
        'MID_1800': '1830-1860',
        'LATE_1800': '1860-1890',
        'EARLY_1900': '1890-1921'
    }
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    tokens = re.split('[ !?.\"]', doc)
    lowered_tokens = map(lambda t: t.lower(), tokens)
    return tokens

class SVMClassifier:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        self.data = DataFrame({'text': [], 'class': []})
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        initial_doc_counts = {}
        for label in GLOBAL_LABLES:
            initial_doc_counts[label] = 0.0
        self.class_total_doc_counts = initial_doc_counts

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = initial_doc_counts

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        initial_word_dicts = {}
        for label in GLOBAL_LABLES:
            initial_word_dicts[label] = defaultdict(float)
        self.class_word_counts = initial_word_dicts

        self.pipeline = Pipeline([
            ('vectorizer',  CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)),
            ('classifier',  MultinomialNB()) ])


    def parse_text(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs

        #print "Starting training with paths %s and %s" % (pos_path, neg_path)
        labeled_paths = []
        for label in GLOBAL_LABLES:
            path = os.path.join(TRAIN_DIR, GLOBAL_LABLES[label])
            labeled_paths.append((path, GLOBAL_LABLES[label]))
        print labeled_paths
        for (p, label) in labeled_paths:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            rows = []
            index = []
            for f in filenames:
                if f != '.DS_Store':
                    with open(os.path.join(p,f),'r') as doc:
                        content = doc.read()
                        text = tokenize_doc(content)
                        rows.append({'text': text, 'class': label})
                        index.append(f)
            data_frame = DataFrame(rows, index=index)
            self.data = self.data.append(data_frame)
        self.data = self.data.reindex(np.random.permutation(self.data.index))

    def fit_naive_bayes(self):
        self.pipeline.fit(self.data['text'].values, self.data['class'].values)
        examples = ['adams jackson', "lincoln johnson", "arthur cleveland"]
        print self.pipeline.predict_proba(['thou thy thee art ye hast hence thine doth dost oft'])
        print self.pipeline.predict(['thou thy thee art ye hast hence thine doth dost oft'])
        predictions = self.pipeline.predict(examples)
        print predictions # [1, 0]


    def k_fold(self):
        k_fold = KFold(n=len(self.data), n_folds=10)
        scores = []
        initial_arr = []
        sample_arr = []
        for i in range(NUM_PERIODS):
            inner_arr = []
            for i in range(NUM_PERIODS):
                inner_arr.append(0)
            sample_arr = inner_arr
            initial_arr.append(inner_arr)
        confusion = np.array(initial_arr)
        labels = []
        for label in GLOBAL_LABLES:
            labels.append(GLOBAL_LABLES[label])
        precTotal = 0
        recallTotal = 0
        fscoreTotal = 0
        scoreCount = 0
        for train_indices, test_indices in k_fold:
            train_text = self.data.iloc[train_indices]['text'].values
            train_y = self.data.iloc[train_indices]['class'].values

            test_text = self.data.iloc[test_indices]['text'].values
            test_y = self.data.iloc[test_indices]['class'].values

            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)
            confusion += confusion_matrix(test_y, predictions, labels=labels)
            precision, recall, fscore, support = score(test_y, predictions, average='weighted', labels=labels)
            precTotal += precision
            recallTotal += recall
            fscoreTotal += fscore
            scoreCount += 1

        print('Total texts classified:', len(self.data))
        print('Precision', precTotal/scoreCount)
        print('Recall', recallTotal/scoreCount)
        print('Fscore', fscoreTotal/scoreCount)
        print('Confusion matrix:')
        print(confusion)

    def fit_bernouilli(self):
        self.pipeline = Pipeline([
            ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2), tokenizer=lambda doc: doc, lowercase=False)),
            ('classifier',         BernoulliNB(binarize=0.0)) ])
        self.fit_naive_bayes

    def train_model(self, num_docs=None):
        self.parse_text(num_docs)
        self.fit_naive_bayes()
        self.k_fold()


if __name__ == '__main__':
    sv = SVMClassifier()
    sv.train_model(num_docs=1000000)
    #evaluate_nb_model()
