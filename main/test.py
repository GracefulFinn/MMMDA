import pandas as pd
import numpy as np
import copy
import torch
import csv
import datetime
from model import Model
from util import *
import torch_geometric.transforms as T
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


settings_3={
    "emb_dim":[128,128],
    "kmer":3,
    "hidden_channels":128,
    "heads":8,
    "num_layers":2,
    "lr":0.00003,
    "T_max":25,
    "bs":96,
    "set_index":18
}
settings_4={
    "emb_dim":[128,128],
    "kmer":3,
    "hidden_channels":128,
    "heads":8,
    "num_layers":2,
    "lr":0.00002,
    "T_max":35,
    "bs":128,
    "set_index":7
}
settings_rd={
    "emb_dim":[128,128],
    "kmer":3,
    "hidden_channels":128,
    "heads":16,
    "num_layers":2,
    "lr":0.0001,
    "T_max":25,
    "bs":128,
    "set_index":6
}

settings=settings_3
dataset='data_hmdd3.2'
alldata,mirna_id,disease_id,graph_without_asso=get_data(dataset)

def test_model( val_loader, model):
    val_r=()
    preds = []
    ground_truths = []
    total_loss = total_examples = 0
    
    loader = val_loader
    model.eval()
    for sampled_data in loader:
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred= model(sampled_data).flatten()####
            preds.append(getpred('BCE',pred).detach())
            
            ground_truths.append(sampled_data["mirna", "rela", "disease"].edge_label)

            loss=getloss('BCE',pred, sampled_data["mirna", "rela", "disease"].edge_label.float())
        
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    pred_ = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    
    val_r,draw=get_metrics(ground_truth,pred_,0.5)
    
    print(f"Epoch: --, Loss: {total_loss / total_examples}")
    print(f"AUC:{val_r[0]},AUPR:{val_r[1]}")

    return (val_r,draw)

def test(model_5):
    all_fold_val=[]
    all_fold_draw=[]
    settings_=model_5["settings"]
    print(settings_)
    model_5_weight=model_5["weights"]
    for i in range(5):
        data=alldata[str(i)]
        print("\n* * * * * * * * FOLD %d * * * * * * * *\n" %(i+1))
        graph=copy.deepcopy(graph_without_asso)
        graph["mirna", "rela", "disease"].edge_index = torch.tensor(get_edge_index(data["train"]["posi"],mirna_id,disease_id))
        graph = T.ToUndirected()(graph)

        val_pairs=np.concatenate([data["test"]["posi"],data["test"]["nega"]])
        val_loader = get_loader(val_pairs,mirna_id,disease_id,graph,settings_["bs"],shuffle=False)
        
        model = Model(
            len=[graph_without_asso["mirna"].f.shape[1],graph_without_asso["disease"].f.shape[1],graph_without_asso["mirna"].kmer[settings_["kmer"]].shape[1]],
            kmer=settings_["kmer"],
            emb_dim=settings_["emb_dim"],
            hidden_channels=settings_["hidden_channels"],
            heads=settings_["heads"],
            num_layers=settings_["num_layers"]).to(device)
        model.load_state_dict(model_5_weight[i])
        val_result,draw=test_model( val_loader,model)

        all_fold_val.append(np.array(val_result))
        all_fold_draw.append(draw)

    print(np.array(all_fold_val).shape)
    print(np.array(all_fold_draw).shape)
    all_fold_val_mean=np.mean(all_fold_val,axis=0)

    return all_fold_val_mean,all_fold_draw


model=torch.load("../model/"+dataset+'.pth',map_location="cpu")
res,draw=test(model)
print(res)
