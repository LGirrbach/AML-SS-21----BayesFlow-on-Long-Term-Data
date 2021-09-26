# AML-SS21

This project contains a reimplementation and extension of [1] using [PyTorch](https://pytorch.org/) and [FrEIA](https://github.com/VLL-HD/FrEIA/).
Concretely, we apply the BayesFlow model architecture to estimate parameters of an epidemiological model for simulating COVID-19.

## Requirements

 1. Setup a new virtual environment (we recommend using [Anaconda](https://www.anaconda.com/)
 2. Install FrEIA using `pip`: `pip install git+https://github.com/VLL-HD/FrEIA.git` or check out the [documentation](https://github.com/VLL-HD/FrEIA/).
    If you want to use GPU, make sure your PyTorch installation is properly set up for CUDA.
 3. Install the other requirements in `requirements.txt`
 
## Usage
For training a new model, simply run `python main.py`. For an overview over parameters, run `python main.py --help`.

## Results
Our results are in the provided jupyter notebooks. You can have a look at them. The code for evaluation is taken from
https://github.com/stefanradev93/AIAgainstCorona and modified for our purposes.

## References


[1] Radev, S.T., Graw, F., Chen, S., Mutters, N.T., Eichel, V.M., Bärnighausen, T., Köthe, U.: Model-based bayesian inference of disease outbreak dynamics with invertible neural networks. arXiv preprint arXiv:2010.00300 (2020)
