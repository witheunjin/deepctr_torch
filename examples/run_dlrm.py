import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.dlrm import DLRM


def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]
 
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender, 'score': score}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    return x, y, feature_columns


if __name__ == "__main__":
    x, y, feature_columns = get_xy_fd()
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DLRM(feature_columns, device=device)
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=3, epochs=10, verbose=2, validation_split=0.0)
