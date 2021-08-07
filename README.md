# AdaGPR
AdaGPR is a adaptive graph convolution model which extends GCNII with generalized Pageranks at each layer. The coefficients of genaralized Pageranks are learned in an end-to-end manner to make convolution at each layer adaptive. 

# Requirements
+ Python 
+ Pytorch 1.3.1 
+ CUDA 10.1

# Usage
To run semi-supervised node-classification experiments run the follwoing script.
```sh
sh semi.sh
```
To run fully-supervised node-classification experiments run the follwoing script.
```sh
sh full.sh
```
