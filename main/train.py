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


def train_model(train_loader, model, optimizer,num_epochs):
    train_result=[]
    train_models=[]
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["T_max"])
    for epoch in range(0, num_epochs+1):
        train_r=()
        preds = []
        ground_truths = []
        total_loss = total_examples = 0

        loader = train_loader
        model.train()
        for sampled_data in loader:
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                sampled_data = sampled_data.to(device)
                pred= model(sampled_data).flatten()####
                preds.append(getpred('BCE',pred).detach())         
                ground_truths.append(sampled_data["mirna", "rela", "disease"].edge_label)

                loss=getloss('BCE',pred, sampled_data["mirna", "rela", "disease"].edge_label.float())
                loss.backward()
                optimizer.step()
            
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

        pred_ = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        scheduler.step()
        train_r,_=get_metrics(ground_truth,pred_,0.5)
        print(f"{str(datetime.datetime.now())[11:19]},Epoch: {epoch:03d}, Loss: {total_loss / total_examples}")
        print(f"AUC:{train_r[0]},AUPR:{train_r[1]}")

        train_models.append(copy.deepcopy(model))
        train_result.append(train_r)

    return (train_result,train_models)


def result(set_index):

    all_train_model=[]
    for i in range(5):
        data=alldata[str(i)]
        print("\n* * * * * * * * FOLD %d * * * * * * * *\n" %(i+1))
        graph=copy.deepcopy(graph_without_asso)
        graph["mirna", "rela", "disease"].edge_index = torch.tensor(get_edge_index(data["train"]["posi"],mirna_id,disease_id))
        graph = T.ToUndirected()(graph)

        train_pairs=np.concatenate([data["train"]["posi"],data["train"]["nega"]])
        train_loader = get_loader(train_pairs,mirna_id,disease_id,graph,settings["bs"],shuffle=True)

        model = Model(
            len=[graph_without_asso["mirna"].f.shape[1],graph_without_asso["disease"].f.shape[1],graph_without_asso["mirna"].kmer[settings["kmer"]].shape[1]],
            kmer=settings["kmer"],
            emb_dim=settings["emb_dim"],
            hidden_channels=settings["hidden_channels"],
            heads=settings["heads"],
            num_layers=settings["num_layers"])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])

        train_result,train_models=train_model(train_loader,model, optimizer,set_index)
        all_train_model.append(np.array(train_models))

    model=np.array(all_train_model)[:,set_index]
    model_st=[i.state_dict() for i in model]
    torch.save({"settings":settings,"weights":model_st}, "../model/"+dataset+'.pth')


result(settings["set_index"])


