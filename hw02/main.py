## 生成 main.py 时请勾选此 cell
from utils import DGraphFin
from utils.evaluator import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import os


# 模型定义
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, dropout: float = 0.05):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_c, h_c)
        self.conv2 = SAGEConv(h_c, int(h_c / 2))
        self.conv3 = SAGEConv(int(h_c / 2), out_c)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, adj, **kwargs):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, adj)
        return F.log_softmax(x, dim=-1)


# 模型载入
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
model = GraphSAGE(
    in_c=20,
    out_c=2,
    h_c= 32,
    dropout= 0.0,
)
model_save_path = f'results/model.pt'
# 这里可以加载你的模型
model.load_state_dict(torch.load(model_save_path, map_location=device))


# 全局变量
global predictions
predictions = None

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    global predictions

    if predictions is None:
        with torch.no_grad():
            model.eval()
            # 在GPU上预测所有节点
            predictions = model(data.x, data.adj_t)
        
    out_node = predictions[node_id]
    y_pred = out_node.exp()  # (num_classes)

    return y_pred
