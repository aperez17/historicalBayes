#!/usr/bin/env python

from __future__ import division

import math
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
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
EARLY_TO_MID_1700 ='1710-1780'
LATE_TO_MID_1800 = '1780-1850'
MID_TO_EARLY_1900 = '1850-1920'

# Path to dataset
if len(sys.argv) < 2:
    print "YOU NEED TO SPECIFY YOUR LOCAL CORPUS DIRECTORY AHHHHH!!!"
    print "i.e. /Users/alexperez/Documents/cs580/project/historicalBayes/preProcessedCorpus"
    sys.exit(1)
PATH_TO_DATA = sys.argv[1]
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")

categories = [EARLY_TO_MID_1700, LATE_TO_MID_1800, MID_TO_EARLY_1900]

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
        self.class_total_doc_counts = { EARLY_TO_MID_1700: 0.0,
                                        LATE_TO_MID_1800 : 0.0,
                                        MID_TO_EARLY_1900: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { EARLY_TO_MID_1700: 0.0,
                                         LATE_TO_MID_1800: 0.0,
                                         MID_TO_EARLY_1900: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { EARLY_TO_MID_1700: defaultdict(float),
                                   LATE_TO_MID_1800 : defaultdict(float),
                                   MID_TO_EARLY_1900: defaultdict(float) }

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

        early_to_mid_1700_path = os.path.join(TRAIN_DIR, EARLY_TO_MID_1700)
        late_to_mid_1800_path = os.path.join(TRAIN_DIR, LATE_TO_MID_1800)
        mid_to_early_1900_path = os.path.join(TRAIN_DIR, MID_TO_EARLY_1900)
        #print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [ (early_to_mid_1700_path, EARLY_TO_MID_1700), (late_to_mid_1800_path, LATE_TO_MID_1800), (mid_to_early_1900_path, MID_TO_EARLY_1900) ]:
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
        predictions = self.pipeline.predict(examples)
        print predictions # [1, 0]

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN EARLY CLASS:", self.class_total_doc_counts[EARLY_TO_MID_1700]
        print "NUMBER OF DOCUMENTS IN LATE CLASS:", self.class_total_doc_counts[LATE_TO_MID_1800]
        print "NUMBER OF DOCUMENTS IN MID CLASS:", self.class_total_doc_counts[MID_TO_EARLY_1900]
        print "NUMBER OF TOKENS IN EARLY CLASS:", self.class_total_word_counts[EARLY_TO_MID_1700]
        print "NUMBER OF TOKENS IN LATE CLASS:", self.class_total_word_counts[LATE_TO_MID_1800]
        print "NUMBER OF TOKENS IN MID CLASS:", self.class_total_word_counts[MID_TO_EARLY_1900]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        for word in bow:
            newWordCount = 0
            if word in self.class_word_counts[label]:
                newWordCount = self.class_word_counts[label][word]
            else:
                # New word
                self.vocab.add(word)
            self.class_total_word_counts[label] += 1
            self.class_word_counts[label][word] = newWordCount + bow[word]
        self.class_total_doc_counts[label] += 1


    def k_fold(self):
        k_fold = KFold(n=len(self.data), n_folds=6)
        scores = []
        confusion = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        precTotal = [0,0,0]
        recallTotal = [0,0,0]
        fscoreTotal = [0,0,0]
        supportTotal = [0,0,0]
        scoreCount = 0
        for train_indices, test_indices in k_fold:
            train_text = self.data.iloc[train_indices]['text'].values
            train_y = self.data.iloc[train_indices]['class'].values

            test_text = self.data.iloc[test_indices]['text'].values
            test_y = self.data.iloc[test_indices]['class'].values

            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)
            confusion += confusion_matrix(test_y, predictions)
            # score = f1_score(test_y, predictions, pos_label=EARLY_TO_MID_1700)
            precision, recall, fscore, support = score(test_y, predictions)
            for i in range(len(precTotal)):
                precTotal[i] = precTotal[i] + precision[i]
                recallTotal[i] = recallTotal[i] + recall[i]
                fscoreTotal[i] = fscoreTotal[i] + fscore[i]
                supportTotal[i] = supportTotal[i] + support[i]
            scoreCount += 1

        for i in range(len(precTotal)):
            precTotal[i] = precTotal[i]/scoreCount
            recallTotal[i] = recallTotal[i]/scoreCount
            fscoreTotal[i] = fscoreTotal[i]/scoreCount
            supportTotal[i] = supportTotal[i]/scoreCount
        print('Total texts classified:', len(self.data))
        print('Precision', precTotal)
        print('Recall', recallTotal)
        print('Fscore', fscoreTotal)
        print('Support:', supportTotal)
        print('Confusion matrix:')
        print(confusion)

    def fit_bernouilli(self):
        self.pipeline = Pipeline([
            ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2), tokenizer=lambda doc: doc, lowercase=False)),
            ('classifier',         BernoulliNB(binarize=0.0)) ])
        self.fit_naive_bayes

    def train_model(self, num_docs=None):
        self.parse_text(num_docs)
        #self.fit_naive_bayes()
        #self.k_fold()
        self.fit_bernouilli()
        self.k_fold()


if __name__ == '__main__':
    sv = SVMClassifier()
    sv.train_model(num_docs=1000000)
    #evaluate_nb_model()
