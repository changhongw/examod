# Explainable Audio Models (ExAMod)

Code for the paper: Changhong Wang, Vincent Lostanlen, and Mathieu Lagrange. **Explainable Audio Classification of Playing Techniques with Layer-wise Relevance Propoagation**, submitted to [IEEE International Conference on Acoustics, Speech and Signal Processing](https://2023.ieeeicassp.org/) (ICASSP), 2023. 

Accompany website showing explanation visualizations and audio examples: [changhongw.github.io/examod](https://changhongw.github.io/publications/examod.html).

## How to run
### Get code
`git clone https://github.com/changhongw/ExAMod.git`

### Install dependencies
`conda create -n myenv python=3.9.7`<br>
`conda install --file requirements.txt`

### Data
This work uses the [Studio On Line](https://forum.ircam.fr/collections/detail/sol-instrumental-sounds-datasets/) (SOL) dataset (version 0.9HQ). The meta data used are generated by `0_data_meta.ipynb`, including the training, validation, and test split.

### Carrier-modulation feature map extraction
`python 1_preprocess_feature.py`

### Playing technique classification
- train and test network
`2_classification.py`

- check classification result
`3_classification_results.ipynb`

### Explanation maps
- Local evidence map
`4_local_relevance_maps.ipynb`

- Class-wise aggregation
`5_classwise_aggregation.ipynb`
