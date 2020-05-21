# Sum-Product Attend-Infer-Repeat

This is the official implementation of "Sum-Product Attend-Infer-Repeat" (SuPAIR) as presented in
the paper "[Faster Attend-Infer-Repeat with Tractable Probabilistic Models](https://ml-research.github.io/papers/stelzner2019icml_SuPAIR.pdf)" by Karl
Stelzner, Robert Peharz, and Kristian Kersting (ICML 2019).

SuPAIR learns to decompose scenes into objects and background in an unsupervised manner via a 
structured probabilistic modelling approach. By employing tractable sum-product networks as
appearance models for objects and background, SuPAIR learns faster and more robustly than AIR,
and works well even on images with noisy backgrounds.

<p align="center">
   <img src="https://raw.githubusercontent.com/stelzner/supair/master/images/count-accs.png" alt="SuPAIR performance" height="300">
   <img src="https://raw.githubusercontent.com/stelzner/supair/master/images/noise-results.png" alt="SuPAIR noise results" height="300">
</p>

## Dependencies
We ran our experiments using Python 3.6 and CUDA 9.0, making use of the following Python packages:

 * tensorflow-gpu 1.7
 * numpy 1.15
 * scipy 1.1
 * scikit-image 0.14.1
 * visdom 0.1.8
 * observations 0.1.4
 * matplotlib 3.0
 * pillow 5.2

These may be installed via `pip install -r requirements.txt`. Other versions might also work but
were not tested.

## Project Structure

 * `model.py` contains the specification of the SuPAIR model and its inference network which are
   the main contributions of the paper
 * `main.py` contains the training loop and routines for running the reported experiments
 * `config.py` holds important options and hyperparameters
 * `region_graph.py` and `RAT_SPN.py` contain our implementation of (random) sum-product-networks
 * `datasets.py` handles the generation and loading of the various datasets
 * `visualize.py` contains routines for drawing visualizations
 * `make_plots.py` generates plots given performance data

## Run
Simply run `python src/main.py`. Adjust the configuration object created at the bottom of the file as
needed, or call one of the predefined functions to reproduce the experiments in the paper.

## Citation
If you find this repository, or the ideas presented within useful in your research, please consider citing our paper:
```
@inproceedings{stelzner2019supair,
  title={Faster Attend-Infer-Repeat with Tractable Probabilistic Models},
  author={Stelzner, Karl and Peharz, Robert and Kersting, Kristian},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  pages={5966--5975},
  year={2019},
  pdf = 	 {http://proceedings.mlr.press/v97/stelzner19a/stelzner19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/stelzner19a.html}
}
```

## Acknowledgement
This project received funding from the European Union's Horizon 2020 research
and innovation programme under the Marie Sklodowska-Curie Grant Agreement No.
797223 (HYBSPN).

