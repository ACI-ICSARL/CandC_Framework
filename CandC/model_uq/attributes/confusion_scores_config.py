# confusion_scores.py

class Confusion_Scores_Thesaurus():
    thesaurus = dict()
    thesaurus.__setitem__("condition Positive","P")
    thesaurus.__setitem__("condition Negative","N")
    thesaurus.__setitem__("true Postive","TP")
    thesaurus.__setitem__("true Negative","TN")
    thesaurus.__setitem__("misclassified Positive","MP")
    thesaurus.__setitem__("false Positive","FP")
    thesaurus.__setitem__("false Negative","FN")
    thesaurus.__setitem__("sensitivity/true Positive rate","TPR")
    thesaurus.__setitem__("specificity/true Negative rate","TNR")
    thesaurus.__setitem__("misclassified Positive rate","MPR")
    thesaurus.__setitem__("precision","PPV")
    thesaurus.__setitem__("negative predictive value","NPV")
    thesaurus.__setitem__("miss rate","FNR")
    thesaurus.__setitem__("fall out","FPR")
    thesaurus.__setitem__("false discovery rate","FDR")
    thesaurus.__setitem__("false omission rate","FOR")
    thesaurus.__setitem__("Error Rate","ER")
    thesaurus.__setitem__("Catastrophic Error Ratio","CER")
    thesaurus.__setitem__("Positive likelihood ratio","LR+")  
    thesaurus.__setitem__("Negative likelihood ratio","LR-")
    thesaurus.__setitem__("Threat Score","TS")
    thesaurus.__setitem__("Prevalence","Prevalence")
    thesaurus.__setitem__("Multi Class Accuracy","MCA")
    thesaurus.__setitem__("Binary Class Accuracy","BCA")
    thesaurus.__setitem__("Balanced Accuracy","BAL")
    thesaurus.__setitem__("F1 Score","F1")
    thesaurus.__setitem__("Binary F1 Score","BF1")
    thesaurus.__setitem__("Matthews Correlation Coefficient","MCC")
    thesaurus.__setitem__("Fowlkes-Mallows Index","FM")
    thesaurus.__setitem__("Informedness", "BM")
    thesaurus.__setitem__("Markedness","MK")
    thesaurus.__setitem__("Diagnostic Odds Ratio","DOR")
    thesaurus.__setitem__("ROC-AUC Score", "RAS")
    thesaurus.__setitem__("Cohen's Kappa", "CK")
    thesaurus.__setitem__("Variation Ratio","VR") 
    thesaurus.__setitem__("Variation Ratio Original","VRO")
    thesaurus.__setitem__("Predictive Entropy","PE")
    thesaurus.__setitem__("Mutual Information Prediction","MIP")
    thesaurus.__setitem__("Average Precision","APS")
    thesaurus.__setitem__("Component Competence","CC")
    thesaurus.__setitem__("Empirical Compenence","EC")

class Confusion_Scores():
    pass