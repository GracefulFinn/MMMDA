import torch
from torch_geometric.data import HeteroData
from torch import Tensor
from MLA import HANConv

class GNN(torch.nn.Module):
    def __init__(self,  hidden_channels,  heads,num_layers):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HANConv(-1, hidden_channels, metadata=(['mirna', 'disease'], [('mirna', 'rela', 'disease'),('disease', 'rev_rela', 'mirna'),('mirna', 'family', 'mirna'),('disease', 'dag', 'disease')]), heads=heads)
            self.convs.append(conv)

        self.conv3 = HANConv(-1, hidden_channels, metadata=(['mirna', 'disease'], [('mirna', 'rela', 'disease'),('disease', 'rev_rela', 'mirna')]), heads=heads)

    def forward(self, x, edge_index_dict):
        edge_only_asso={
            ('mirna', 'rela', 'disease'):edge_index_dict[('mirna', 'rela', 'disease')],
            ('disease', 'rev_rela', 'mirna'):edge_index_dict[('disease', 'rev_rela', 'mirna')]
        }
        for conv in self.convs:
            x = conv(x, edge_index_dict)
        # x = self.conv1(x, edge_index_dict)
        x = self.conv3(x, edge_only_asso)
        return x


class Classifier(torch.nn.Module):
    def __init__(self,hidden_channels,pre_type):
        super().__init__()
        self.lin1=torch.nn.Linear(hidden_channels*2, 64)
        self.lin2=torch.nn.Linear(64, pre_type)
    def forward(self, x_mirna: Tensor, x_disease: Tensor, edge_label_index: Tensor) -> Tensor:
        mirna = x_mirna[edge_label_index[0]]
        disease = x_disease[edge_label_index[1]]
        x = self.lin1(torch.concat((mirna,disease),1)).relu()
        x = self.lin2(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self,xdim,len,kmer,winlen):
        super().__init__()
        self.kmer=kmer

        self.mirna_kmer_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=( winlen,4**kmer+1))
        self.mirna_kmer_linear = torch.nn.Linear(len[2] - winlen + 1, xdim[0])
        
        self.dis_lin =torch.nn.Sequential(
            torch.nn.Linear(len[1], 1024),
            torch.nn.ReLU(), 
            # nn.Dropout(p=e_drpt),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.15),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, xdim[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.BatchNorm1d(xdim[1])
        )

    def forward(self, data: HeteroData) -> Tensor:
        convkmer = self.mirna_kmer_conv(data["mirna"].kmer[self.kmer].unsqueeze(1).float()) #n*1*l[2]*(4**kmer+1)
        convkmer = convkmer.squeeze(-1).squeeze(1) 
        convkmer = self.mirna_kmer_linear(convkmer)
        x_dict = {
            "mirna": convkmer,
            "disease": self.dis_lin(data["disease"].f),
        } 
        return x_dict

class Model(torch.nn.Module):
    def __init__(self, len,kmer,emb_dim,hidden_channels,heads,num_layers,winlen=10,pre_type=1):
        super().__init__()

        self.MLP=MLP(emb_dim,len,kmer,winlen)
        self.gnn = GNN(hidden_channels, heads=heads,num_layers=num_layers)
        self.classifier = Classifier(hidden_channels,pre_type)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict=self.MLP(data)
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["mirna"],
            x_dict["disease"],
            data["mirna", "rela", "disease"].edge_label_index,
        )
        return pred