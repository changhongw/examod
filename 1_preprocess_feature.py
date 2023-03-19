import os 
import numpy as np, torch
from tqdm import tqdm
import gin
import librosa

from examod.SOLdataset import SOLdatasetModule
from kymatio.torch import Scattering1D
from examod.utils import make_directory, normalize_audio

gin.enter_interactive_mode()

class Extractor():
    def __init__(self,
                 output_dir,
                 data_module):
        self.output_dir = output_dir
        self.data_module = data_module
        self.mu_S1 = []
        self.mu_S1S2 = []

    def get_loaders(self):
        loaders = [('training', self.data_module.train_ds),
                   ('validation', self.data_module.val_ds),
                   ('test', self.data_module.test_ds)]
        return loaders

    def stats(self):
        print('Computing Mean Stat ...')
        self.mu_S1 = np.mean(np.stack(self.mu_S1), 0)
        self.mu_S1S2 = np.mean(np.stack(self.mu_S1S2), 0)
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu_S1'), self.mu_S1)
        np.save(os.path.join(stats_path, 'mu_S1S2'), self.mu_S1S2)

class Scat1DExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 scat1d_kwargs={
                    'shape': 2**18,
                    'J': 13,
                    'T': 2**13,
                    'Q': (8, 2)},
                freq_min = 32,  # freq below which the modulations not audible
                sr=44100): 
        super().__init__(output_dir, data_module)
        self.output_dir = output_dir
        self.data_module = data_module
        self.scat1d_kwargs = scat1d_kwargs

        self.scat1d = Scattering1D(**scat1d_kwargs).cuda()
        meta = self.scat1d.meta()

        # carrier center frequencies condition >= freq_min
        freq_cnd = meta['xi'][:, 0] * sr >= freq_min

        # indices for scat coefficient to extract 
        idxs_S1 = np.where(np.logical_and(meta['order'] == 1, freq_cnd))[0]  # order 1
        idxs_S1S2 = np.where(np.logical_and(meta['order'] != 0, freq_cnd))[0] # order 1 and 2

        self.idxs = (idxs_S1, idxs_S1S2)

    def run(self):

        loaders = self.get_loaders()

        for subset, loader in loaders:
            subset_dir = os.path.join(self.output_dir, subset)
            make_directory(subset_dir)
            print(f'Extracting Scat1D for {subset} set ...')
            for idx, item in tqdm(enumerate(loader)):
                audio, _, fname = item
                audio = librosa.util.fix_length(audio, size=self.scat1d_kwargs['shape'])
                audio = normalize_audio(audio)
                audio = torch.tensor(audio).cuda()

                Sx = self.scat1d(audio)
                S1x = Sx[self.idxs[0]]
                S1S2x = Sx[self.idxs[1]]
                if subset == 'training':
                    # collect integrated over time
                    self.mu_S1.append(S1x.mean(dim=-1).cpu().numpy())
                    self.mu_S1S2.append(S1S2x.mean(dim=-1).cpu().numpy())

                out_path = os.path.join(subset_dir, os.path.split(fname)[-1])
                np.save(out_path + '_S1', S1x.cpu().numpy())
                np.save(out_path + '_S1S2', S1S2x.cpu().numpy())


def process_SOLdataset(data_dir='/home/changhongw/datasets/SOL-0.9HQ-PMT',
                      feature='scat1d_s1s2', out_dir='SOL-0.9HQ-PMT/'):
    """ extract SOL-PMT dataset scattering coefficients and stats, and save to disk
    Args:
        data_dir: source data directory for SOL-PMT audio files
        feature: 'scat1d'
        output_dir_id: optional identifier to append to the output dir name
    """

    output_dir = os.path.join(os.getcwd(), out_dir + feature.split('_')[0])
    make_directory(out_dir)
    make_directory(output_dir)
    data_module = SOLdatasetModule(data_dir, batch_size=32, feature='',
                                    out_dir_to_skip=output_dir)
    data_module.setup()

    # 'scat1d':
    extractor = Scat1DExtractor(output_dir, data_module)

    extractor.run()
    extractor.stats()

if __name__ == "__main__":
    process_SOLdataset()
