# Machine Learning Certainty and Competence Framework

This is a Python implementation, primarily with heavy use of [PyTorch](https://pytorch.org) implementing the Certainty and Competence Framework, and for providing an interface for using this framework to describe machine learning (ML) model uncertainty. The Certainty component of the framework was first discussed in the paper [Measuring Classification Decision Certainty and Doubt](https://arxiv.org/abs/2303.14568), while the first expression of the Competence part of the framework appeared in the paper [Uncertainty-Quantified, Robust Deep Learning for Network Intrusion Detection](https://ieeexplore.ieee.org/abstract/document/10407559).  Other papers implementing the framework are forthcoming.

The key component of the framework's efficacy is the Competence Hierarchy. The main theorem of the competence hierarchy, intended for ML models which are approximating partition functions of the sample space, identifies levels of competence ranging from incompetence, relatively amateurish, relatively competent, expert, and prescient. Particularly, ML models that achieve relative competence or higher have distribution-independent guarantees which entail that True Positives (TPs) stochastically dominate False Positives (FPs) in terms of the certainty scores. The competence hierarchy applies both to the global TP versus FP distributions, independent of a predicted label, as well as to the TP and FPs within a predicted label. Relatedly, there is the omicron statistic, which is the averaged norm distance between points drawn from two samples of certainties. In particular, we furnish several omicron statistic based methods for out-of-distribution and FP detection testing, among them, a multivariate logistic regression that takes the omicron for a predicted category, the empirical competence of the model for a predicted category, and their interaction as inputs. This method has comparable to better performance over other methods, such as Energy-Based out-of-distribution detection methods.

This Python library consists of the following folders: cancd, certainty_stats, loss, model_uq, oodd, and utils. 

The candc folder consists of the certainty and competence folders which are used for computing certainty and competence respectively. 

The certainty_stats folder contains functions for the construction of empirical probability distribution functions for certainty scores.

The loss folder, presently under development, contains methods for training with respect to certainty or competence.

The model_uq folder provides the framework with an interface between several classes, described in greater detail below, that are to be used for uncertainty quantification for a given ML model within the framework.

The oodd folder contains various functions for Out-of-Distribution Detection (OODD) within the framework.

The utils folder contains various functions and classes that arise in the course of conduction UQ and OODD tests within the framework, as well as interchanging the objects that appear in the model_uq folder for objects outside of the framework.


Although framework expects SoftMax adjusted inputs which are output from either Pytorch or TensorFlow, in principle, certainty scores can also be computed with respect to logits. However, direct comparison between ML models cannot be performed on logit certainty scores.


* Perform certainty computation on user-input probability objects, either as torch.Tensor objects capturing either one model sample as a matrix, or sampled distributions as a 3-tensor, or otherwise as a numpy array corresponding to probabilities of label assignment.


## How to install

```
pip install git+https://github.com/Army-Cyber-Institute/CandC_Framework
```
### How to set-up

We have provided two scripts, one python and one shell script for UNIX(-like) systems: setup.py and tutorial_setup.sh

The tutorial script can be run to set-up the folders for the tutorial. This consists of a data and tutorial folder. Running the tutorial_setup.sh script is optional. The EMNIST dataset will need to be downloaded due to a deprecated script within the torchvision library that prevents it from being accessed by ordinary means within the tutorial.

This data can be found at the following addresses:
* **[NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)**
* **[BIOMETRICS](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)**
* **[Online Guide](https://marvinschmitt.com/blog/emnist-manual-loading/)**

## Running the code
### Dependencies
* torch>=1.6.0
* numpy
* pickle
* statsmodels
* plotly
* keras>=2.0
* pytorch_ood
* tqdm
* gc
* math
* pandas
* hamiltorch
* itertools
* datetime
* sklearn
* typing

  
### Running the Framework

C&C_Framework is a library of functions for computing certainty and collecting statistics useful for inference. Each function within the package describes how it is to be used. 

A sample work flow might be as follows:

1. Generate predictions with a neural network model from underlying data set D.
2. Run get_certainty on predictions, returning predicted_label, cert, and cert_score
3. If we wish to perform further analysis, we can store the Model_Data and either spin out individual model_uq objects, or spin up a model_uq interface instance to compute a suite of statistics.
If one wants to perform out of distribution detection using the certainty scores from a Bayesian neural network, run out_of_distribution_detection with the baseline certainty score dataframe as the first argument, and a dictionary of params, each key corresponding the sample one is performing out of distribution detection on, using the distribution of certainty scores. 

The out-of-distribution functions and tests are contained in the subfolder, oodd. Presently one can test tranches of sample data drawn from deterministic or Bayesian ensemble models by default. If one wants to test Bayesian samples from a specific input, specify the function Bayesian. This will run the Bayesian variant of the out-of-distribution detection test.

A full walkthrough of using the model_uq interface is provided in the Model_UQ_Tutorial.ipynb notebook.

Roughly, the workflow breaks down as follows:

1. Create a torch DataLoader object with your desired data and a corresponding model
2. Set-up the model_uq_parameters, which requires providing a name, a device, a model, addresses for where the data is stored, address where the model_uq interface will be stored, a target tpr_threshold for oodd test calibration, m and the number of known classes (n_class) for the labeled data. Additional parameters can be provided for, and are named in the model_uq/base.py file.
3. Create a model_uq instance via CandC.model_uq.Model_UQ(**model_uq_params)
4. Set-up the Model_Data as a dictionary, default name data, for the model_uq interface by providing a parameters naming the data, providing an input_dataloader, a boolean indicating if the data is labeled or not, and a dictionary `classification_categories` whose key value pairs correspond between the label index and the label name
5. Generate model data via CandC.model_uq.Model_Data(model=model,data=input_data)
6. Set-up the parameters for the model_uq.fill_uq() method. The full scope is accounted for inside the model_uq/base.py file, but this requires a model_data object or address and optional name to load in the desired model data.
7. The fill_uq command creates and independently saves several class objects implementing the CandC Framework. Those have been discueed above,but include the scores object, as well as certainties, the certainty score distribution, and the omicrons of a given set of input data.
8. Additionally, out of distribution detection tests can be run, as described in the final section of the tutorial.

## How to cite?
```
@misc{BerenbeimBastian,
      title={Machine Learning Certainty and Competence Framework}, 
      author={Alexander M. Berenbeim and Nathaniel D. Bastian},
      year={2024}
}
```

## Who developed UCQ?
[Intelligent Cyber-Systems and Analytics Research Laboratory, Army Cyber Institute, United States Military Academy](https://cyber.army.mil/Research/Research-Labs/ICSARL/)
