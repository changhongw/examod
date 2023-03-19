# Explainable Audio Models (ExAMod)

Code for the paper: Changhong Wang, Vincent Lostanlen, and Mathieu Lagrange. **Explainable Audio Classification of Playing Techniques with Layer-wise Relevance Propoagation**, submitted to [IEEE International Conference on Acoustics, Speech and Signal Processing](https://2023.ieeeicassp.org/) (ICASSP), 2023. 

## How to run
### Get code
`git clone https://github.com/changhongw/ExAMod.git`

### Install dependencies
`conda env create -f environment.yml`<br>
`conda activate examod`<br>

### Data
This work uses the [Studio On Line](https://forum.ircam.fr/collections/detail/sol-instrumental-sounds-datasets/) (SOL) dataset (version 0.9HQ). Please contact the dataset creators for version 0.9HQ if you would like to reproduce all the results in our paper. 

After downloading the dataset, the `SOL-PMT` subset and the meta data can be automatically generated by running
- `0_data_meta.ipynb`

The data split and the corresponding file IDs used in our paper is shown in the `SOL-0.9HQ-PMT_meta.csv` file.

### Carrier-modulation feature map extraction
`python 1_preprocess_feature.py`<br>
(feature extraction takes around 0.5 h on one GPU)

### Playing technique classification
- train and test network
`python 2_classification.py`<br>
(classification with the full scattering feature takes around 15 mins on one GPU)

- check classification result
`3_classification_results.ipynb`

### Explanation maps
- Local evidence map
`4_local_relevance_maps.ipynb`

- Class-wise aggregation
`5_classwise_aggregation.ipynb`

Please feel free to open an issue if you find any bug.