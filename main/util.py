import pandas as pd
import random
import numpy as np
import copy
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import LinkNeighborLoader

def get_data(dataset):
    alldata=np.load("../data/"+dataset+"/data_sep/train_test.npy",allow_pickle=True).item()
    mirna_id=pd.read_csv("../data/"+dataset+"/feature/mirna_id.tsv",sep="\t")
    disease_id=pd.read_csv("../data/"+dataset+"/feature/disease_id.tsv",sep="\t")
    graph_without_a= torch.load("../data/"+dataset+"/graph/data.pt")
    return (alldata,mirna_id,disease_id,graph_without_a)

def get_edge_index(ids,mirna_id,disease_id):
    mirna_disease_df = pd.DataFrame({
        'mirna': [tmp_id[0] for tmp_id in ids],
        'disease': [tmp_id[1] for tmp_id in ids]
    })
    mid_mirna_id = pd.merge(mirna_disease_df['mirna'], mirna_id, left_on='mirna', right_on='mirna_id', how='left')
    mid_mirna_id = mid_mirna_id['mapped_id'].values
    mid_disease_id = pd.merge(mirna_disease_df['disease'], disease_id, left_on='disease', right_on='disease_id', how='left')
    mid_disease_id = mid_disease_id['mapped_id'].values
    return np.stack([mid_mirna_id, mid_disease_id], axis=0)

def get_loader(pairs,mirna_id,disease_id,graph,batch,shuffle):

    edge_label_index = torch.tensor(get_edge_index(pairs,mirna_id,disease_id))
    edge_label=torch.tensor([1]*int(len(pairs)/2)+[0]*int(len(pairs)/2))

    train_loader = LinkNeighborLoader(
        data=graph,  
        num_neighbors=[-1,-1,-1,-1,-1,-1],  # 
        # neg_sampling_ratio=0.0, 
        edge_label_index=(("mirna", "rela", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch,
        shuffle=shuffle,
    )
    return train_loader


def getloss(this_loss,pred,label):
    if this_loss=='BCE' :
        si=torch.nn.Sigmoid()
        pred=si(pred)
        criterion = torch.nn.BCELoss()#sigmoid
        loss = criterion(pred, label)
    elif this_loss=='CE':
        criterion = torch.nn.CrossEntropyLoss()#softmax
        loss = criterion(pred, label)

    return loss


def getpred(this_loss,pred):
    if this_loss=='BCE' :
        si=torch.nn.Sigmoid()
        pred=si(pred)
    elif this_loss=='CE':
        so=torch.nn.Softmax(dim=1)
        pred=so(pred)
        
    return pred


def get_metrics(labels, preds, which_threshold):
    AUC = roc_auc_score(labels, preds)
    fpr, tpr, _ =  roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    AUPR = auc(recall, precision)

    if which_threshold == 0.5:
        threshold = 0.5
    else:
        threshold = np.percentile(preds, 100 - which_threshold)

    top_n_preds = np.array([1 if p > threshold else 0 for p in preds])
    
    ACC = accuracy_score(labels, top_n_preds)
    P = precision_score(labels, top_n_preds)
    R = recall_score(labels, top_n_preds)
    F1 = f1_score(labels, top_n_preds)

    # print(round(AUC, 3), round(AUPR, 3), round(ACC, 3), round(P, 3), round(R, 3), round(F1, 3))
    return ([AUC, AUPR, ACC, P, R, F1, threshold],{"auc":AUC,"aupr":AUPR,"acc":ACC,"p":P,"r":R,"f1":F1,"fpr":fpr,"tpr":tpr,"precision":precision,"recall":recall})
