from sys import argv
from models import ConvClassifier, Partitioner
from os.path import isfile


"""
Top-level script for running the 'chat disentanglement' model.
"""


def train_and_featurize(train_file, dev_file, is_baseline):
    # initialize classifier
    model = ConvClassifier()

    # if the feature files exist, train the model. Otherwise, featurize first
    if isfile(train_file) and isfile(dev_file):
        print('=' * 30)
        model.train_from_file(train_file, dev_file)
        print('=' * 30)
        part_acc = Partitioner(model, is_baseline)
    else:
        model.create_feature_files(train_file, dev_file, is_baseline)
        print('=' * 30)
        model.train_from_file(train_file, dev_file)
        print('=' * 30)
        part_acc = Partitioner(model, is_baseline)


if __name__ == "__main__":
    # Some very primitive command-line interaction
    try:
        if argv[1] == 'baseline':
            train_and_featurize('baseline_features_train.tsv', 'baseline_features_dev.tsv', is_baseline=True)
        elif argv[1] == 'new':
            train_and_featurize('new_features_train.tsv', 'new_features_dev.tsv', is_baseline=False)
        else:
            raise IndexError
    except IndexError:
        print("Error: invalid arguments.")