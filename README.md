[bruno]: https://github.com/BrunoXBSantos
[bruno-pic]: https://github.com/BrunoXBSantos.png?size=120
[nelson]: https://github.com/nelsonmestevao
[nelson-pic]: https://github.com/nelsonmestevao.png?size=120
[rui]: https://github.com/Syrayse
[rui-pic]: https://github.com/Syrayse.png?size=120
[susana]: https://github.com/SusanaMarques
[susana-pic]: https://github.com/SusanaMarques.png?size=120

<div align="center">

# Generative Adversarial Networks for Molecule Generation

[Geeting Started](#rocket-getting-started)
|
[Development](#hammer-development)
|
[Tools](#hammer_and_wrench-tools)
|
[Team](#busts_in_silhouette-team)

</div>

Deep learning methods, and in particular deep generative models, such as
Autoencoders (AEs) and Generative Adversarial Networks (GANs), have been
gaining popularity in drug discovery, aiming to accelerate the discovery of new
therapeutical molecules while reducing costs. The project aims to, given a
training set of molecules, implement various forms of Generative Adversarial Networks,
such as: Deep Convolution GAN, LatentGAN; in order to generate new, unseen, molecules.

Currently the project encompasses the following experimentations:

- [Vanilla GAN](https://gitlab.com/mieiuminho/ds/aa2/tp/-/blob/main/notebooks/SimpleGAN.ipynb) : Experimentation with a vanilla GAN in order to generate new images of molecule representations. The results were far from expected and we've found that the Vanilla GAN was inadequate for this purpose.
- [Deep Convolutional GAN](https://gitlab.com/mieiuminho/ds/aa2/tp/-/blob/main/notebooks/DCGAN.ipynb) : As a result, we experimented on using a Deep Convolution GAN in order to take advantage of the neighbouring power of kernels. However, results were inadequate and the team agreed that image representations of molecules are inadequate for molecular generation.
- [Latent GAN](https://gitlab.com/mieiuminho/ds/aa2/tp/-/blob/main/notebooks/LatentGAN.ipynb) : After drifting from our original goal of the generation of molecular graphic representations, we intended to use each molecules specific fingerprint in order to map it into a pre-trained latent space. In quality, the results surpassed the previous experimentations and allowed us to better understand the inner-workings of such methods.

Future work:

- [Objective-Reinforced GANs](https://gitlab.com/mieiuminho/ds/aa2/tp/-/blob/main/notebooks/MolClustering.ipynb) : In order to do an analysis of more autonomous GANs, we propose conducing experimentations with GANs mixed with Reinforcement Learning. With such experiments, we should be able to understand how hyperparameter optimization can be crucial in generating the maximum amount of valid molecules.

See also:

- [Molecule Clustering]() : Through the use of various molecular properties, we apply clustering algorithms for the analysis of molecular sets with similar properties. Currently, we've built a data set containing for than a million properties for molecules. Next, we'll experiment with density-based clustering and fixed-size clustering, such as DBSCAN and KMEANS.

## Project Organization

`bin/`: Utility script for most common tasks.

`data/`: Datasets used for training the models.

`notebooks/`: Jupyter Notebooks with analysis, data visualization and model training.

`└── moses.ipynb`:  Sample exploratory analysis of moses dataset.

`└── PropriedadesMoleculas.ipynb`: Generation of a new dataset with molecular properties from the original dataset with only the SMILES representation of a molecule.

`└── DCGan.ipynb`: A Deep Convolutional GAN.

`└── SimpleGAN.ipynb`

`scripts/`: Utility helpers.

`src`: Python source code.

`requirements-dev.txt`: Packages used during development.

`requirements.txt`: Required packages.

`README.md`: Project and development instructions.

[report.pdf](https://gitlab.com/mieiuminho/ds/aa2/tp/-/blob/main/report.pdf): The project report.

## :rocket: Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

Start by running the setup script that installs all required packages as long
as you have all met all the [prerequisites](#inbox_tray-prerequisites).

```bash
bin/setup
```

### :inbox_tray: Prerequisites

The following software is required to be installed on your system:

- [Python 3.9+](https://www.python.org/)

### :hammer: Development

Start a Jupyter Lab environment.

```
bin/jupyter lab
```

Start a Jupyter Console.

```
bin/jupyter console
```

Format the code accordingly to common guide lines.

```
bin/format
```

Lint your Python code.

```
bin/lint
```

### :hammer_and_wrench: Tools

The recommended Development Environment is [Google
Colab](https://colab.research.google.com).

### :link: References

- [The RDKit Documentation](https://rdkit.org/docs/index.html)
- [Deep learning for molecular design — a review of the state of the art](https://pubs.rsc.org/en/content/articlelanding/2019/me/c9me00039a)
- [Mol-CycleGAN: a generative model for molecular optimization](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0404-1)
- [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/pdf/1805.11973.pdf)
- [Reinforced Adversarial Neural Computer for de Novo Molecular Design](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00690)
- [A De Novo Molecular Generation Method Using Latent Vector Based Generative Adversarial Network](https://chemrxiv.org/articles/preprint/A_De_Novo_Molecular_Generation_Method_Using_Latent_Vector_Based_Generative_Adversarial_Network/8299544)
- [Direct steering of de novo molecular generation with descriptor conditional recurrent neural networks](https://www.nature.com/articles/s42256-020-0174-5)
- [Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models](https://arxiv.org/abs/1811.12823)


## :busts_in_silhouette: Team

| [![Bruno][bruno-pic]][bruno] | [![Nelson][nelson-pic]][nelson] | [![Rui][rui-pic]][rui] | [![Susana][susana-pic]][susana] |
| :--------------------------: | :-----------------------------: | :--------------------: | :-----------------------------: |
|    [Bruno Santos][bruno]     |    [Nelson Estevão][nelson]     |    [Rui Reis][rui]     |    [Susana Marques][susana]     |

