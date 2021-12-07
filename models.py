from featurize import Dataset, FeatureMaker
from copy import deepcopy
import numpy as np
from sklearn import linear_model


"""
The classifier and partitioner that make up the thread detection model
"""


class ConvClassifier:
    def __init__(self):
        self.dataset = Dataset('IRC/dev/linux-dev-0X.annot', 'IRC/pilot/linux-pilot-0X.annot')
        self.featurizer = FeatureMaker(self.dataset)

    def train_from_file(self, features_path_train, features_path_dev):
        self.train_data = np.loadtxt(features_path_train, delimiter='\t')
        self.dev_data = np.loadtxt(features_path_dev, delimiter='\t')

        self.train_X = self.train_data[...,:-1]
        self.train_y = self.train_data[...,-1]

        self.dev_X = self.dev_data[...,:-1]
        self.dev_y = self.dev_data[...,-1]

        self.model = linear_model.LogisticRegression(solver='newton-cg').fit(self.train_X, self.train_y)
        y_hat = self.model.predict(self.train_X)
        score = np.sum(y_hat == self.train_y) / len(self.train_y)
        print('Training accuracy: {:.3f}'.format(score))

        dev_y_hat = self.model.predict(self.dev_X)
        dev_score = np.sum(dev_y_hat == self.dev_y) / len(self.dev_y)
        print('Dev accuracy: {:.3f}'.format(dev_score))

    def create_feature_files(self, save_train_path, save_dev_path, is_baseline):
        train_pairs = self.get_pairs(self.dataset.train_set)
        self.featurize(train_pairs, save_train_path, is_baseline)

        dev_pairs = self.get_pairs(self.dataset.dev_set)
        self.featurize(dev_pairs, save_dev_path, is_baseline)

    def get_pairs_for(self, b, dset):
        # Find all possible example pairs for a given utterance: If another utterance is within 50 seconds, it's considered eligible
        pairs = []
        exs = deepcopy(dset)
        exs.remove(b)
        for e in exs:
            if abs(b[1] - e[1]) < 50:
                pairs.append((b, e))
        return pairs

    def get_pairs(self, dset):
        # get pairs for every example in a dataset
        pairs = []
        for b in dset:
            for p in self.get_pairs_for(b, dset):
                pairs.append(p)
        return pairs

    def featurize(self, pairs, save_txt, is_baseline):
        # create vectors for a list of tuples, and save to a file
        arr = self.featurizer.get_feature_vector(*pairs[0], baseline=is_baseline)
        
        n = 1
        for p in pairs[1:]:
            arr = np.vstack([arr, self.featurizer.get_feature_vector(*p, baseline=is_baseline)])
            print('Featurizing {} of {}'.format(n, len(pairs)))
            n += 1

        np.savetxt(save_txt, arr, delimiter='\t', fmt='%.3f')


class Partitioner:
    """
    Partition a dataset of example-tuples into multiple conversations.
    Parameters: a reference to a ConvClassifier and a bool to determine if only baseline features should be used
    """
    def __init__(self, classf, is_baseline):
        self.featurizer = classf.featurizer
        self.linear_model = classf.model

        self.test_data = classf.dataset.test_set
        self.gold_conv = self.get_gold_clusters()
        self.is_baseline = is_baseline

        # partition the test data using the model, then use the gold clusters to predict a final partitioning accuracy
        sys_convos = self.partition()
        print('Test set average overlap: {:.3f}'.format(self.conv_overlap(sys_convos)))

    def partition(self):
        # attempt to completely segment the test set. It needs a reference to a Classifier object's trained linear model
        convos = {}
        for entry in self.test_data:
            print('Entry {} of {}'.format(self.test_data.index(entry) + 1, len(self.test_data)))
            # if empty, add first entry to a new conversation
            if not convos:
                convos[len(convos)] = [entry]
            # otherwise, test entry against the last entry in all conversations
            else:
                # get prediction probabilities for all of the clusters, and choose the max. If all predictions are less than 0, create a new cluster
                votes = np.zeros(len(convos))
                for i in range(len(convos)):
                    pair = self.featurizer.get_feature_vector(convos[i][-1], entry, baseline=self.is_baseline)[:-1]
                    proba = self.linear_model.predict_proba(pair.reshape(1,-1))
                    votes[i] = proba[0][1]
                votes = votes - 0.5
                if np.any(votes > 0):
                    convos[np.argmax(votes)].append(entry)
                else:
                    convos[len(convos)] = [entry]
        return convos

    def get_gold_clusters(self):
        # create a dictionary that represents the "gold standard" conversation clustering
        gold_convos = {}
        for entry in self.test_data:
            if entry[0] not in gold_convos.keys():
                gold_convos[entry[0]] = [entry]
            else:
                gold_convos[entry[0]].append(entry)
        return gold_convos

    def conv_overlap(self, sys):
        # evaluate the model's partitioning accuracy: return the average overlap 
        n_pairs = 0
        tot_avg_overlap = 0

        # find the gold conversation with the highest overlap, add to the total
        for sys_entry in sys.values():
            max_overlap = 0
            for gold_entry in self.gold_conv.values():
                overlap = set(sys_entry).intersection(set(gold_entry))
                overlap = len(overlap) / len(gold_entry)
                if overlap > max_overlap:
                    max_overlap = overlap
            n_pairs += 1
            tot_avg_overlap += max_overlap

        return tot_avg_overlap / n_pairs