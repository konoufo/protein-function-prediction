from os.path import join as pjoin, dirname as pdirname, abspath as pabspath
import os
import argparse
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prec_rec_f
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


class GOClassifier:
    def __init__(self, X, y, random_seed=11, test_size=0.25, *args, **kwargs):
        ind = np.arange(X.shape[0])
        np.random.seed(random_seed)
        np.random.shuffle(ind)
        self.X = X[ind]
        self.y = y[ind]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed)
        self.random_seed = random_seed
        self.args = args
        self.kwargs = kwargs
        self.clf = None

    def fit(self, X=None, y=None):
        X_ = X if X is not None else self.X_train
        y_ = y if y is not None else self.y_train
        self.clf = MultiOutputClassifier(SGDClassifier(
            alpha=0.0001, max_iter=1000, tol=1e-3, random_state=self.random_seed, *self.args, **self.kwargs))
        self.clf.fit(X_, y_)
        return self.clf

    def predict(self, X=None):
        assert self.clf is not None
        X_ = X if X is not None else self.X
        return self.clf.predict(X_)
    
    def test_predict(self):
        return self.predict(X=self.X_test)

    def score(self, X, y):
        assert self.clf is not None
        return self.clf.score(X, y)

    def test_score(self):
        assert self.clf is not None
        return self.clf.score(self.X_test, self.y_test)

    def train_score(self):
        assert self.clf is not None
        return self.clf.score(self.X_train, self.y_train)


if __name__ == '__main__':
    # Usage:
    # $ python goclassifier.py
    parser = argparse.ArgumentParser(description='Tissue-specific Protein Function Prediction from embeddings')
    parser.add_argument('-d', '--data-dir', dest='datadir', type=str)
    parser.add_argument('-t', '--tissues-file', dest='tissues_file', type=str)
    parser.set_defaults(
        datadir=pjoin(os.getcwd(), 'data'),
        tissues_file=pjoin(os.getcwd(), 'data', 'tissues.list'))
    args = parser.parse_args(sys.argv[1:])
    data_dir = args.datadir
    # with open(args.tissues_file, 'r') as f:
    #     tissues = f.read().split('\n')
    tissues = ['leukocyte']
    random_seed = 11

    # load ohmnet embeddings
    dim_size = 128
    embs_file = pjoin(data_dir, 'leaf_vectors.emb')
    dtemb = np.loadtxt(embs_file, delimiter=' ', skiprows=1,
        dtype={'names': tuple(['id'] + ['d{}'.format(i + 1) for i in range(dim_size)]),
               'formats': tuple(['U64'] + [np.float] * dim_size)})
    for name in tissues:
        print('Function prediction on tissue "{}"'.format(name))
        # Find all target files for this tissue, one file per biological function
        target_files = os.listdir(pjoin(data_dir, 'bio-tissue-labels'))
        target_files = [f for f in target_files if f.startswith(name)]
        if not target_files:
            print('Skipping tissue "{}". No function labels available.'.format(name))
            continue
        # Load target data which maps gene id to functions
        y = None
        for f in target_files:
            tmp = np.loadtxt(pjoin(data_dir, 'bio-tissue-labels', f), skiprows=1, dtype=np.int32, delimiter='\t')
            if y is not None:
                assert tmp.shape[0] == y.shape[0]
                y = np.concatenate((y, tmp[:, 1:2]), axis=1)
            else:
                y = tmp
        tissue_embs = dtemb[np.core.defchararray.startswith(dtemb['id'], 'data_edgelists_{}'.format(name))]
        assert tissue_embs.shape[0], 'No embeddings for tissue {}'.format(name)
        emb_ids = np.core.defchararray.replace(tissue_embs['id'], 'data_edgelists_{}.edgelist__'.format(name), '')
        emb_ids = np.array(emb_ids, dtype=np.int32)
        # Only keep embeddings with known target
        available_target = np.isin(emb_ids, y[:, 0], assume_unique=True)
        X = np.zeros((tissue_embs[available_target].shape[0], dim_size))
        for i in range(dim_size):
            X[:, i] = tissue_embs['d{}'.format(i + 1)][available_target]
        y = y[np.isin(y[:, 0], emb_ids, assume_unique=True), 1:]

        # fit new classifier using learnt embeddings
        fun_clf = GOClassifier(X, y, random_seed=random_seed, loss='modified_huber', test_size=.25)
        fun_clf.fit()

        print('Train score ({} ohmnet embeddings): {}'.format(fun_clf.X_train.shape[0], fun_clf.train_score()))
        print('Test score: {}'.format(fun_clf.test_score()))
        y_pred = fun_clf.test_predict()
        print('Number of predicted positives: {}'.format(y_pred[y_pred == 1].sum()))
        scores = prec_rec_f(fun_clf.y_test, y_pred, labels=np.arange(y_pred.T.shape[0]), average='micro')
        print('Precision, Recall, F-score: {}'.format(scores))
