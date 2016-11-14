#!/usr/bin/env python

from __future__ import division

import math
import os
import sys
import re
import operator
import numpy as np
import matplotlib.pyplot as plt

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


def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = re.split('[ !?.\"]', doc)
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        if token is not '':
            bow[token] += 1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
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


    def train_model(self, num_docs=None):
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
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

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


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not
        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def better_tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not
        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = better_tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Implement me!
        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        wordCount = self.class_word_counts[label][word]
        totalCount = self.class_total_word_counts[label]
        try:
            return (wordCount / totalCount)
        except:
            0

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Implement me!
        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        wordCount = self.class_word_counts[label][word] + alpha
        totalCount = self.class_total_word_counts[label] + (alpha * len(self.vocab))
        try:
            return (wordCount / totalCount)
        except:
            0


    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        likelihood = 0.0
        for word in bow:
            likelihood += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))
        return likelihood

    def log_prior(self, label):
        """
        Implement me!
        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        prior = math.log(self.class_total_doc_counts[label]) - (math.log(self.class_total_doc_counts[EARLY_TO_MID_1700] + self.class_total_doc_counts[LATE_TO_MID_1800] + self.class_total_doc_counts[MID_TO_EARLY_1900]))
        return prior

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!
        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        log_prior = self.log_prior(label)
        log_likelihood = self.log_likelihood(bow, label, alpha)
        return log_prior + log_likelihood

    def classify(self, bow, alpha):
        """
        Implement me!
        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)
        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        early_prob = self.unnormalized_log_posterior(bow, EARLY_TO_MID_1700, alpha)
        late_prob = self.unnormalized_log_posterior(bow, LATE_TO_MID_1800, alpha)
        mid_prob = self.unnormalized_log_posterior(bow, MID_TO_EARLY_1900, alpha)
        max_label = EARLY_TO_MID_1700
        prob_dict = {
            EARLY_TO_MID_1700: early_prob,
            LATE_TO_MID_1800: late_prob,
            MID_TO_EARLY_1900: mid_prob
        }
        for label in prob_dict:
            if prob_dict[label] > prob_dict[max_label]:
                max_label = label
        return max_label


    def evaluate_classifier_accuracy(self, alpha):
        """
        Implement me!
        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        early_to_mid_1700_path = os.path.join(TRAIN_DIR, EARLY_TO_MID_1700)
        late_to_mid_1800_path = os.path.join(TRAIN_DIR, LATE_TO_MID_1800)
        mid_to_early_1900_path = os.path.join(TRAIN_DIR, MID_TO_EARLY_1900)
        #print "Starting testing with paths %s and %s" % (pos_path, neg_path)
        correctCount = 0.0
        totalCount = 0.0
        for (p, label) in [ (early_to_mid_1700_path, EARLY_TO_MID_1700), (late_to_mid_1800_path, LATE_TO_MID_1800), (mid_to_early_1900_path, MID_TO_EARLY_1900) ]:
            filenames = os.listdir(p)
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow = tokenize_doc(content)
                    if self.classify(bow, alpha) is label:
                        correctCount += 1
                    totalCount += 1
        return (correctCount / totalCount)


def evaluate_nb_model() :
    print 'Evaluation'
    print "Accuarcy pseduo param at 1: " + (str(nb.evaluate_classifier_accuracy(1) * 100) + "%")
    print ''
    print "PROB OF 'WAR' EARLY: ", nb.p_word_given_label_and_psuedocount('machine', EARLY_TO_MID_1700, 1)
    print "PROB OF 'WAR' LATE: ", nb.p_word_given_label_and_psuedocount('machine', LATE_TO_MID_1800, 1)
    print "PROB OF 'WAR' MID: ", nb.p_word_given_label_and_psuedocount('machine', MID_TO_EARLY_1900, 1)
    counter = 0
    fig, ax = plt.subplots()
    # top_words = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']
    top_words = ['thou', 'thy','thee', 'art', 'ye', 'hast', 'hence', 'thine', 'doth', 'dost', 'oft']
    freq_lists = []
    labels = []
    for label in nb.class_word_counts:
        labels.append(label)
        freq_list = []
        label_words = nb.class_word_counts[label]
        for word in top_words:
            freq_list.append(nb.p_word_given_label_and_psuedocount(word, label, 1))
        freq_lists.append(freq_list)
    X = np.arange(len(top_words))
    rects = []
    for i in range(len(freq_lists)):
        color = 'b'
        if i is 1:
            color = 'g'
        elif i is 2:
            color = 'r'
        rects.append(ax.bar(X + (i * 0.25), freq_lists[i], color = color, width = 0.25)[0])
    ax.set_xticks(X + .25)
    ax.set_xticklabels(top_words)
    ax.legend(rects, labels)

    plt.show()
    print '[done.]'

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model(num_docs=1000000)
    evaluate_nb_model()
