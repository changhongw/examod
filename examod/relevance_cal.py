import torch
import numpy as np
import glob, h5py, os
from zennit.attribution import Gradient

class relevance_cal():
    def __init__(self,
                test_dir='SOL-0.9HQ-PMT/scat1d/test/',
                rule_types=None,
                model=None,
                class_abbrv=None,
                ):
        super().__init__()

        self.test_dir = test_dir
        self.rule_types = rule_types
        self.model = model
        self.class_abbrv = class_abbrv

    def feature_relevance_class_rule(self, class_name, rule):

        test_files = glob.glob(self.test_dir + "/**/*" + self.class_abbrv[class_name] + "*S1S2.npy", recursive = True)
        input_feature_all = torch.from_numpy(np.load(test_files[0]))
        input_feature_all = input_feature_all.view(1, input_feature_all.shape[0], -1)
        
        for file in test_files[1:]:
            feature = torch.from_numpy(np.load(file))
            feature = feature.view(1, feature.shape[0], -1)
            input_feature_all = torch.cat((input_feature_all, feature), 0)

        # u-log for input
        mu = torch.tensor(np.load('SOL-0.9HQ-PMT/scat1d/stats/mu_S1S2.npy'))
        c = torch.tensor([1e-1])
        data = torch.log1p(input_feature_all / (c[None, :, None] * mu[None, :, None] + 1e-8))

        # create the attributor, specifying model and composite
        with Gradient(model=self.model, composite=rule) as attributor:
            # compute the model output and attribution
            output, _ = attributor(data)
            pred = torch.zeros(data.shape[0], len(self.class_abbrv))
            for k in range(len(pred)):
                pred[k, output.softmax(-1).argmax(-1)[k]] = 1
            
            _ , relevance_all = attributor(data, pred)

        return input_feature_all, relevance_all

    def feature_relevance_all(self):
        with h5py.File("results/feature_relevance.hdf5", "w") as f:
            for rule in self.rule_types:
                for class_name in self.class_abbrv:
                    feature, relevance = self.feature_relevance_class_rule(class_name, self.rule_types[rule])
                    feature_path = os.path.join(rule, class_name, 'feature')
                    relevance_path = os.path.join(rule, class_name, 'relevance')
                    f.create_dataset(feature_path, data = feature)
                    f.create_dataset(relevance_path, data = relevance)