import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from ..candc import *
from tqdm import tqdm
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc
from pytorch_ood.detector import Entropy


class Deep_Ensemble():
    f""" Optional OODD architecture drawn from []() implementing an instance of the Deep Ensemble architecture
    """
    def __init__(self,
                 model_list,
                 epsilon,
                 learning_rate,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 batch_size,
                 test_size,
                 tpr_threshold,
                 num_iter,
                 proper_scoring_rules=nn.CrossEntropyLoss()):
        self.models = model_list
        self.scoring_rules = proper_scoring_rules if type(proper_scoring_rules)==list else [proper_scoring_rules for _ in enumerate(self.models)]
        self.eps = epsilon
        self.optimizers = [torch.optim.Adam(model.parameters(),lr=learning_rate) for model in self.models]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_iter = num_iter
        self.tpr_threshold = tpr_threshold
        self.entropy_cutoff = 0.0
        self.cert_score_thresholds = dict()
        
    def making_batch(self,train=True):
        """ Method to prepare batch
        """
        if train:
            sample_size = self.batch_size
            random_indices = torch.randperm(len(self.x_train))[:sample_size]
            batch_x = self.x_train[random_indices,:]
            batch_y = self.y_train[random_indices].long()
        else:
            sample_size = self.test_size
            random_indices = torch.randperm(len(self.x_test))[:sample_size]
            batch_x = self.x_test[random_indices,:]
            batch_y = self.y_test[random_indices].long()
        return batch_x, batch_y   

    def train_model(self,i,model):
        """ method for training the selected model according to training rule i

        Parameters
        ----------------------
        :i: index for scoring rule to use
        :model: model to use

        """
        batch_x, batch_y = self.making_batch(train=True)
        batch_y = batch_y.long()
        model.train(True)
        #Forward Pass
        optimizer = self.optimizers[i]
        try:
            for _ in (range(10)):
                optimizer.zero_grad()
                output = model(batch_x)
                loss_train = self.scoring_rules[i](output,batch_y)
                loss_train.backward()
                optimizer.step()
        except RuntimeError:
            output_dim = model(batch_x).shape[-1]
            batch_y = F.one_hot(batch_y,output_dim)
            for _ in (range(10)):
                optimizer.zero_grad()
                output = model(batch_x)
                loss_train = self.scoring_rules[i](output,batch_y.float())
                loss_train.backward()
                optimizer.step()
        
    def test_model(self,i,model,batch_x_test,batch_y_test):
        """ method for testing the Deep Ensemble model

        Parameters
        ----------------------
        :i: index for scoring rule to use
        :model: model to use
        :batch_x_test: x batch from test/validation data
        :batcy_y_test y batch from test/validationd data
        """
        model.eval()
        torch.no_grad()
        output = model(batch_x_test)
        try:
            loss_test = self.scoring_rules[i](output,batch_y_test.long())
        except RuntimeError:
            output_dim = output.shape[-1]
            batch_y_test = F.one_hot(batch_y_test,output_dim)
            loss_test = self.scoring_rules[i](output,batch_y_test.float())
        # TO DO: Build out and improve; for now ignore and focus on training-- we don't care about Brier score per se for the OODD test
    
    def train_ensemble(self):
        """ method for training the Deep Ensemble
        """
        for iter in range(self.num_iter):
            batch_x_test, batch_y_test = self.making_batch(train=False)

            for i,model in enumerate(self.models):
                # create the training batch
                self.train_model(i,model)
                self.test_model(i,model,batch_x_test,batch_y_test)

    def apply(self,novel_input):
        """Method to apply DeepEnsemble Model
        """
        outputs=[]
        for model in self.models:
            outputs.append(model(novel_input))
        output=torch.stack(outputs)
        output=output.mean(0)
        return output

    def oodd_test(self,novel_input,labels):
        """Method to perform OODD_test relative to the Entropy of the DeepEnsemble model
        """
        output = F.softmax(self.apply(novel_input),dim=-1)
        print("Output:")
        print(output)
        print("Labels:")
        print(labels)
        labels=labels.detach().long().flatten()
        test_results =dict({'EntropyBased':self.entropy_based_detection(outputs=output,labels=labels),
                           })
        return test_results

    def get_tpr_thresholds(self):
        """Using the model outputs drawn from the training or validation data, find the in-distribution tpr_threshold cutoff item for
        entropy, certainty_score, and the omicron within category value"""
        ensemble_model_outputs = self.apply(self.x_test)
        print("The shape of the ensemble_model_outputs is {}".format(ensemble_model_outputs.shape))
        ensemble_model_probs = F.softmax(ensemble_model_outputs,dim=-1)
        print("The shape of the ensemble_model_probs is {}".format(ensemble_model_probs.shape))
        
    def entropy_based_detection(self,outputs,labels,in_dist_threshold=.95):
        """Runs an Entropy Based Detection Detector test
        """
        entropy_test = dict()
        print("Labeling unknown as 1")
        labels = (labels<0).detach().long()
        print("The shape of the final labels is {}. The bincount is {}".format(labels.shape,torch.bincount(labels)))
        scores = (outputs*torch.log(outputs)).sum(-1)
        scores = torch.nan_to_num(scores, nan= -1e30,neginf=-1e30,posinf=1e30)
        print("The entropies are {} with shape {}".format(scores,scores.shape))
        print("The labels are {} with shape {}".format(labels,labels.shape))
        entropy_test.__setitem__('scores',  scores)
        entropy_test.update(self._internal_test_performance(scores=scores,
                                                            labels=labels,
                                                            tpr_threshold=in_dist_threshold))
        return entropy_test

    def certainty_score_based_detection(self,outputs:torch.Tensor,labels:torch.Tensor):
        """ Method to perform certainty score based detection using the output of the DeepEnsemble model
        """
        _,certainty_scores, predictions = get_certainty(outputs)
        certainty_scores = certainty_scores.reshape(-1)
        print("The certainty scores for the Ensemble Model are {} with shape {}".format(certainty_scores,certainty_scores.shape))
        local_certainty_score_test_dict = dict()
        certainty_score_test_dict = dict()
        N_cats = outputs.shape[1]
        print("The deep ensemble model identifies that there are {} underlying categories".format(N_cats))
        for cat in range(N_cats):
            local_cat_certainty_score_test_dict = dict()
            local_cat_certainty_score_test_dict.__setitem__('scores', certainty_scores[predictions == cat])
            indices = (predictions.long() == int(cat))
            try:
                local_cat_certainty_score_test_dict.update(
                    self._internal_test_performance(scores=certainty_scores[indices],
                                                    labels=labels[indices],
                                                    tpr_threshold=self.tpr_threshold))
            except IndexError:
                print(IndexError)
                print("The indices are {}\n There are {} predicted samples of category {}".format(indices,indices.float().sum(),cat))
        scores = [local_cat_certainty_score_test_dict['scores'] for cat in range(N_cats)]
        scores = torch.cat(scores)
        certainty_score_test_dict.__setitem__('scores',scores)
        try:
            certainty_score_test_dict.update(self._internal_test_performance(scores=scores,labels=labels,tpr_threshold=self.tpr_threshold))
        except Exception:
            print(Exception)
        return certainty_score_test_dict,local_certainty_score_test_dict

    def fpr_at_tpr(self,pred, target, tpr_rate=0.95):
        """
        Calculate the False Positive Rate at a certain True Positive Rate

        :param pred: outlier scores
        :param target: target label
        :param tpr_rate: cutoff value
        :return:
        """
        # results will be sorted in reverse order
        fpr, tpr, _ = binary_roc(pred, target)
        idx = torch.searchsorted(tpr, tpr_rate)
        if idx == fpr.shape[0]:
            return fpr[idx - 1]

        return fpr[idx]
    
    def _internal_test_performance(self,scores,labels,tpr_threshold):
        """
        """
        print("BEFORE SORTING:The scores for the internal DE test are {} with shape{}.\n The labels are {} with shape{}\n The bin count with 1 for ood is {}".format(scores,scores.shape,labels,labels.shape,torch.bincount(labels.flatten().long())))
        scores =scores.flatten()
        total = scores.shape[0]
        labels = labels.flatten()
        n_scores, scores_idx = torch.sort(scores, stable=True,descending=True)
        print("The scores idx is")
        print(scores_idx)
        print(torch.unique(scores))
        if len(torch.unique(scores))>1:        
            if total == n_scores.shape[0]:
                scores = n_scores
                print("labels {}".format(labels))
                labels = labels[scores_idx]
                print("After {}".format(labels))
            else:
                print("Let's do unstable sorting. Let's sort without forced stability")
                scores, scores_idx = torch.sort(scores,stable=False,descending=True)
                labels = labels[scores_idx]
        print("AFTER SORTING: The scores for the internal DE test are {}.\n The labels are {}\n The bin count with 1 for ood is {}".format(scores,labels,torch.bincount(labels.flatten().long())))
        auroc = binary_auroc(scores, labels)

        # num_classes=None for binary
        p, r, t = binary_precision_recall_curve(scores, labels)
        aupr_in = auc(r, p)

        p, r, t = binary_precision_recall_curve(-scores, 1-labels)
        aupr_out = auc(r, p)

        fpr = self.fpr_at_tpr(pred=scores, target=labels,tpr_rate=tpr_threshold)
    
        output= {"AUROC": auroc.cpu(),
                "AUPR-IN": aupr_in.cpu(),
                "AUPR-OUT": aupr_out.cpu(),
                "FPR95TPR": fpr.cpu(),}
        print("DeepEnsemble Test results\n {}".format(output))
        return output
