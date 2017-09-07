import numpy as np

class MaxFeatureMap(object):
    def __init__(self, n_top=9):
        self.max_vals_ind = np.ones(n_top)
        self.n_top = n_top

    def update(self, blob):
        data = blob[0]
        n_channels = data.shape[0]
        # normalization
        data = (data - data.min()) / (data.max() - data.min())

        sum_act = np.sum(np.sum(data,1),1)
        ind = np.argsort(sum_act)
        self.max_vals_ind = ind[-self.n_top-1:-1]

class MaxActivationPoint(object):
    def __init__(self, n_top):
        self.max_loc = -np.ones((n_top, 3), dtype='int')
        self.n_top = n_top

    def update(self, blob):
        data = blob[0]
        width = data[0].shape[1]
        self.max_feature_map = MaxFeatureMap(self.n_top)
        self.max_feature_map.update(blob)
        count = 0
        for ind in self.max_feature_map.max_vals_ind:
            flatten_ind = np.argmax(data[ind])
            self.max_loc[count,0] = flatten_ind/width
            self.max_loc[count,1] = flatten_ind%width
            self.max_loc[count,2] = ind
            count += 1