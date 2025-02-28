import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from  dig.xgraph.dataset import MoleculeDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

class train_GCN(torch.nn.Module):
    def __init__(self, nfeat,hidden_channels,nclass):
        super(train_GCN, self).__init__()
        # torch.manual_seed(12345)
        # self.weight=Parameter(torch.Tensor(nfeat, hidden_channels),requires_grad=True)
        self.conv1 = GCNConv(nfeat, hidden_channels,add_self_loops=True,normalize=True,bias=False)
        self.conv2 = GCNConv(hidden_channels, nclass,add_self_loops=True,normalize=True,bias=False)
        # self.embedding=nn.Embedding(40, hidden_channels)
        # self.lin = Linear(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        # x=[i for i in range(x.shape[0])]
        #
        # x=self.embedding(torch.tensor(x))


        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        # print(batch)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print('x',x.shape)

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        return x

    def back(self, x, edge_index_1, edge_index_2):

        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1, x)
class GCN(torch.nn.Module):
    def __init__(self, nfeat,hidden_channels,nclass):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, hidden_channels,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(hidden_channels, nclass,add_self_loops=False,normalize=False,bias=False)
    def forward(self, x, edge_index,edge_weight):

        x = self.conv1(x, edge_index,edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index,edge_weight=edge_weight)

        # 2. Readout layer
        # print(batch)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x

    def back(self, x, edge_index, edge_weight):
        x_0 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index, edge_weight=edge_weight)
        return x_0, x_1

    def pre_forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def verify_layeredge(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2):

        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index1, edge_weight=edge_weight1))
        x = self.conv2(x, edge_index2, edge_weight=edge_weight2)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]


        return x






def train(model,optimizer,train_number,criterion,dataset):
    model.train()
    optimizer.zero_grad()
    loss=0
    for i in range(train_number):
        data=dataset[i]
        batch=torch.tensor([0]*(data.x.shape[0]))
        # print('data.edge_index',data.edge_index)
        # print('data.x', data.x)
        data.x=data.x.to(torch.float32)
        out = model(data.x, data.edge_index,batch)
        # print('out',out)
        # print('label',data.y)
        loss += criterion(out, data.y)
    loss.backward()
    print('loss',loss)
    optimizer.step()


def acc_val(model,train_number,dataset):
    model.eval()

    correct = 0

    for i in range(train_number,len(dataset)):
        data=dataset[i]
        batch = torch.tensor([0] * (data.x.shape[0]))
        data.x=data.x.to(torch.float32)
        out = model(data.x, data.edge_index, batch)  # 一次前向传播

        pred = out.argmax(dim=1)  # 使用概率最高的类别

        correct += int((pred == data.y).sum())

        # label = data.y.argmax(dim=1)
        # if pred == label:
        #     correct += 1  # 检查真实标签
        # # correct += int((pred == data.y).sum())  # 检查真实标签
    return correct/(len(dataset)-train_number)

def acc_train(model,train_number,dataset):
    model.eval()

    correct = 0

    for i in range(train_number):
        data=dataset[i]
        batch = torch.tensor([0] * (data.x.shape[0]))
        data.x=data.x.to(torch.float32)
        out = model(data.x, data.edge_index, batch)  # 一次前向传播
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct/train_number


    # for data in loader:  # 批遍历测试集数据集。
    #     out = model(data.x, data.edge_index, data.batch)  # 一次前向传播
    #     pred = out.argmax(dim=1)  # 使用概率最高的类别
    #     correct += int((pred == data.y).sum())  # 检查真实标签
    # return correct / len(loader.dataset)

def is_vector_all_false(vector):
    return not any(vector)

def is_one_hot(vector):
    # 计数向量中值为1的元素数量
    ones_count = np.count_nonzero(vector == 1)  # 计算数组中值为1的元素数量
    zeros_count = np.count_nonzero(vector == 0)  # 计算数组中值为0的元素数量

    return ones_count == 1 and zeros_count == (len(vector) - 1)
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
def initializeNodes(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

def train_all(args):
    if args.dataset=='mutag':
        model_name = 'mutag'
        dataset = TUDataset('data', name='MUTAG', use_node_attr='True')
    elif args.dataset=='REDDIT-BINARY' or  args.dataset=='IMDB-BINARY':
        model_name =args.dataset
        dataset = TUDataset('data', name=args.dataset, use_node_attr='True')
        initializeNodes(dataset)
    elif args.dataset=='ClinTox':
        model_name = 'ClinTox'
        dataset = MoleculeDataset('data', model_name)

    dataset = dataset.shuffle()
    train_ratio=0.5

    train_number = int(len(dataset)*train_ratio)

    train_dataset = dataset[:train_number]

    print('train_dataset', train_dataset)
    test_dataset = dataset[train_number:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = train_GCN(nfeat=dataset.num_features, hidden_channels=args.hidden, nclass=dataset.num_classes)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs):
        train(model,optimizer,train_number,criterion,dataset)
        train_acc = acc_train(model,train_number,dataset)
        test_acc = acc_val(model,train_number,dataset)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    torch.save(model.state_dict(), f'data/{model_name}/GCN_model.pth')

    # model.eval()
    # model_path = '../data/' + 'TUdataset/' + 'GCN_model' + '.pth'

    model.load_state_dict(torch.load('data/' + f'{model_name}/' + 'GCN_model' + '.pth'))

    # model.load_state_dict(torch.load(model_path))

    # print(model.state_dict())
    test_acc = acc_val(model,train_number,dataset)
    print(test_acc)