import re
from nltk import word_tokenize
import numpy as np
import itertools
from scipy.spatial.distance import cosine
import torch


"""
Tools for preparing/preprocessing the dataset, and creating pairwise features
"""


class Dataset:
    """
    The annotated data in Elsner & Charniak (2008) was split into 'dev' and 'pilot' sets. I used the dev set 
        for train, and split the pilot set 50/50 for dev/test

        Parameters: paths to data files
    """
    def __init__(self, dev_data_path, pilot_data_path):
        dev_raw_data = [line for line in open(dev_data_path, 'r').readlines()]
        pilot_raw_data = [line for line in open(pilot_data_path, 'r').readlines()]

        self.train_set = []
        self.dev_set = []
        self.test_set = []
        self.lexicon = {}

        # build the datasets and lexicon

        for ex in dev_raw_data:
            self.add_train_entry(ex)

        for ex in pilot_raw_data[:250]:
            self.add_unseen_entry(ex, self.dev_set)

        for ex in pilot_raw_data[250:]:
            self.add_unseen_entry(ex, self.test_set)

    def create_example(self, line):
        # input: string, a chat message from the dataset
        # output: tuple of conversation label, time, speaker, action type, and message contents
        conv = re.search(r'^T-?\d*', line).group(0)
        time = re.sub(conv + ' ', '', re.search(r'^T-?\d*\s\d*', line).group(0))
        txt = re.sub(r'(^T-?\d*\s\d*\s)|\n', '', line)

        speaker = re.search(r'^[\w#]*\s', txt).group(0)
        type = re.search(r'\s[*:]\s\s', txt).group(0)
        utterance = re.sub(r'^\w*\s[*:]\s*', '', txt)

        if re.search(r'\*', type):
            type = 0
        else:
            type = 1

        # tokenize the utterance
        utterance = ' '.join(word_tokenize(utterance.lower()))
        utterance = re.sub(r'[^a-z ]', '', utterance)

        return conv, int(time), speaker.rstrip(), type, utterance

    def add_lexicon_entry(self, word):
        # if a word is in the lexicon, increment its frequency
        # otherwise, make a new entry
        if word in self.lexicon.keys():
            self.lexicon[word] += 1
        else:
            self.lexicon[word] = 1

    def add_train_entry(self, line):
        # create an example tuple, update the lexicon with unseen words
        entry = self.create_example(line.rstrip())
        self.train_set.append(entry)

        # add typed words to the lexicon
        for w in entry[4].split(' '):
            self.add_lexicon_entry(w)

    def add_unseen_entry(self, line, dset):
        # create an example tuple for the dev/test sets, substituting unseen tokens with [UNK]
        # dset: self.dev_set or self.test_set
        entry = self.create_example(line.rstrip())

        for w in entry[4].split(' '):
            if w not in self.lexicon.keys():
                entry = (entry[0], entry[1], entry[2], entry[3], re.sub('\W%s\W' % w, ' [UNK] ', entry[4]))
                self.add_lexicon_entry('[UNK]')
        dset.append(entry)


class FeatureMaker:
    """
    A class for featurizing a tuple example. Requires: data from the Dataset class, and a number for time/probability bins

    Parameters: a reference to a Dataset object
    """
    def __init__(self, dataset):
        self.train = dataset.train_set
        self.lexicon = dataset.lexicon
        self.n_tokens = sum(self.lexicon.values())
        self.bins = 15
        self.time_scale = self.get_time_scale()
        self.unigram_scale = self.get_unigram_scale()
        self.embedder = Embeddings()


    def get_time_scale(self):
        times = [t[1] for t in self.train]
        times.sort()
        dist = times[-1] - times[0]
        return np.logspace(0, np.log10(dist), num=self.bins)

    def bin_time(self, dist):
        for i in range(len(self.time_scale)):
            if self.time_scale[i] < dist:
                return i
        return len(self.time_scale)

    def get_unigram_scale(self):
        # Create a logarithmic scale, and normalize to 0-1 using softmax
        scale = np.logspace(0, 1, num=self.bins)
        return np.exp(scale) / np.sum(np.exp(scale))

    def bin_unigram(self, prob):
        # return the index of the bin that represents the input probability
        if prob == 0:
            return 0
        elif prob == 1:
            return self.bins - 1
        else:
            for i in range(1, self.bins-1):
                if self.unigram_scale[i-1] < prob < self.unigram_scale[i]:
                    return i

    def get_unigram_prob(self, token):
        # use the frequency in the training dataset to estimate token probabilities
        try:
            count = self.lexicon[token]
        except KeyError:
            count = self.lexicon['[UNK]']
        return count / self.n_tokens

    def check_speaker(self, b1, b2):
        # do the chats share the same speaker?
        return 1 if b1[2] == b2[2] else 0

    def check_mention(self, b1, b2):
        # does one user mention the other?
        return 1 if re.search(b1[2].lower(), b2[4]) or re.search(b2[2].lower(), b1[4]) else 0

    def check_question(self, b1, b2):
        # Is there a question mark in either chat?
        return 1 if re.search('\?', b1[4]) or re.search('\?', b2[4]) else 0

    def repeat_i(self, b1, b2, i):
        # The number of shared tokens at unigram probability i
        # first, find shared words
        seq = []
        for t in b1[4].split(' '):
            if re.search(t, b2[4]):
                seq.append(t)
        for t in b2[4].split(' '):
            if re.search(t, b1[4]):
                seq.append(t)
                
        # Then, find how many of them are at unigram probability i
        num_shared = 0
        for t in seq:
            if self.bin_unigram(self.get_unigram_prob(t)) == i:
                num_shared += 1

        return num_shared

    def sum_cosine_i(self, b1, b2, i):
        # sum of all cosine similarities for words in bucket i
        # similar to repeat_i, except we want all the words in either chat at probability i, 
        #   and find the sum distance of the cartesian product
        b1_at_i = []
        for t in b1[4].split(' '):
            if self.bin_unigram(self.get_unigram_prob(t)) == i:
                b1_at_i.append(self.embedder.encode(t))

        b2_at_i = []
        for t in b2[4].split(' '):
            if self.bin_unigram(self.get_unigram_prob(t)) == i:
                b2_at_i.append(self.embedder.encode(t))

        sum_cosine = 0
        for i, j in itertools.product(b1_at_i, b2_at_i):
            sum_cosine += cosine(i, j)

        return sum_cosine

    def calc_sentence_cosine(self, b1, b2):
        # Cosine distance of the sentence-level embeddings for the chats (i.e. output layer of BERT for the whole sentence)
        b1_embed = self.embedder.encode(b1[4])
        b2_embed = self.embedder.encode(b2[4])
        dist = cosine(b1_embed, b2_embed)
        return dist

    def get_feature_vector(self, b1, b2, baseline=False):
        # If we're creating a baseline model, leave out the BERT-related features
        if baseline:
            # 4 non-lexical features, 1 label, and n_bins repeat_i features
            vec = np.zeros(5 + self.bins)
        else:
            # 4 non-lexical features, 1 label, 1 cosine distance feature, n_bins repeat_i features, and n_bins sum_i features
            vec = np.zeros(6 + 2*self.bins)

        # time
        vec[0] = self.bin_time(abs(b1[1] - b2[1]))

        # same speaker?
        vec[1] = self.check_speaker(b1, b2)

        # mention?
        vec[2] = self.check_mention(b1, b2)

        # question?
        vec[3] = self.check_question(b1, b2)

        # unigram frequencies
        for i in range(len(self.unigram_scale)):
            vec[4+i] = self.repeat_i(b1, b2, i)

        if not baseline:
            # sentence cosine distance
            vec[4+self.bins] = self.calc_sentence_cosine(b1, b2)

            # cosine sim
            for i in range(len(self.unigram_scale)):
                vec[5+self.bins+i] = self.sum_cosine_i(b1, b2, i)

            vec[5+2*self.bins] = 1 if b1[0] == b2[0] else 0
        else:
            vec[4+self.bins] = 1 if b1[0] == b2[0] else 0

        return vec


"""
Model for creating BERT embeddings given a token
"""
class Embeddings:
    def __init__(self):
        # use Torch Hub to load a pretrained BERT model and tokenizer
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

    def encode(self, sent):
        indexed_tokens = self.tokenizer.encode(sent, add_special_tokens=True)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            encoded_layers = self.model(tokens_tensor)
            mat = encoded_layers['pooler_output']
            return mat.numpy().flatten()