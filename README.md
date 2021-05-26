<h1 align="center" style="margin-top: 0px;"> <b>Generalization and Exploration via Randomized Value Functions</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=arXiv:1402.0635&color=b31b1b)](https://arxiv.org/abs/1402.0635)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=JAX&color=27A59A)](https://quantumai.google/cirq)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/randomized-value-iteration/blob/main/notebooks/chain.ipynb)
</div>

## **Description**
This repository contains an <ins>unofficial</ins> implementation of the <ins>Generalization and Exploration via Randomized Value Functions</ins> and its application to the <ins>Chain</ins> environment as in :

- Paper : **Generalization and Exploration via Randomized Value Functions**
- Authors : **I. Osband, B. Van Roy and Z. Wen**
- Date : **2016**

## **Details**
- Environment : **Chain environment** *(Paper, Figure 1)* using `bsuite`
- Features : **Random Coherent basis** *(Paper, Algorithm 6)*
- Evaluation method : **Randomized Least Squares Value Iteration** *(Paper, Algorithm 1)* using `JAX`
- Agent : **RLSVI with greedy action** *(Paper, Algorithm 2)*
## **Usage**
To run the experiments :

- Option 1 : Open in [Colab](https://colab.research.google.com/github/qdevpsi3/randomized-value-iteration/blob/main/notebooks/chain.ipynb). 
- Option 2 : Run on local machine. First, you need to clone this repository and execute the following commands to install the required packages :
```
$ cd randomized-value-iteration
$ pip install -r requirements.txt
```
You can run an experiment using the following command :
```
$ cd src
$ python chain.py
```