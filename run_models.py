import os
import time
import torch
import argparse
import pandas as pd 
import torchmetrics
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_fair_dataset, set_seed, sparse_2_edge_index, torch_save, torch_load, fair_metric, fair_metric_new,partition_patch,\
    adjacency_positional_encoding, edge_index_2_sparse_mx, laplacian_positional_encoding, torch_load, torch_save
from sklearn.metrics import f1_score, roc_auc_score
from models import GCN,GAT,FairGP
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['TORCH_USE_CUDA_DSA']='1'

parser = argparse.ArgumentParser()

# nba, german, facebook, income, bail, credit  ||| pokec_z, pokec_n
parser.add_argument('--dataset', type=str, default='credit', help='Random seed.') 
parser.add_argument('--datapath', type=str, default='./data/', help='datapath.') # pokec_z
parser.add_argument('--sens_attr', type=str, default='region', help='sens_attr')  # region(all) gender(pokec_z, pokec_n) 

parser.add_argument('--have_sens', type=bool, default=True, help='FFN layer size')
parser.add_argument('--feat_norm', type=str, default='row',choices=['none','row','column'], help="type of optimizer") # sgd adam adamw adadelta adagrad
parser.add_argument('--label_number', type=int, default=1000, help='label_number for train set.') # 20 22 23 25

parser.add_argument('--model', type=str, default='FairGP', help='Random seed.') # FUGNN_conv
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=20, help='Random seed.') # 20 22 23 25
parser.add_argument('--self_loop', type=bool, default=True, help='FFN layer size')
parser.add_argument('--sens_idex', type=bool, default=False, help='FFN layer size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units of classifier.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gpu', type=int, default=1, help='gpu id.') # 
parser.add_argument('--epochs', type=int, default=300, help='global epoch.') # 
parser.add_argument('--nlayer', type=int, default=2, help='layer of model.') # 20 22 23 25
parser.add_argument('--patience', type=int, default=50, help='early stopping threshold')
parser.add_argument('--nheads', type=int, default=1, help='layer of model.') # 1 2 4 8
parser.add_argument('--metric', type=int, default=4, help='select metric in train') # acc loss -sp-eo acc-sp-eo

# Ours ------------------------------------------------
parser.add_argument('--pe_method', type=str, default='adj', choices=['adj','lap','none'], help='spectral position encoding method') # 2
parser.add_argument('--pe_dim', type=int, default=2, help='spectral num') # 2
parser.add_argument('--norm', type=str, default='layer', help="type of normalization") # layer none
parser.add_argument('--num_hops', type=int, default=1, help='spectral num') # 1
parser.add_argument('--n_patch', type=int, default=100, help='n_patch') # 1
parser.add_argument('--patch_method', type=str, default='metis', choices=['metis', 'louvain', 'random', 'leiden'])


args = parser.parse_args()
# args = parser.parse_args([])


# %%
# load data
sens_idex = False
# sens_idex = True

self_loop = False
# label_number=1000

# device = args.device
device=torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

set_seed(args.seed)

# adj, feature, labels, sens, idx_train, idx_val, idx_test, sens_index = load_dataset(args)
adj, feature, labels, sens, idx_train, idx_val, idx_test, sens_index = load_fair_dataset(args)

edge_index = sparse_2_edge_index(adj) # [e,2]

print("args.dataset:", args.dataset)
print("sens_index:", sens_index)
print("args.metric:", args.metric)
print("args.lr:", args.lr)
print("args.model:", args.model)
print("args.epochs:", args.epochs)
print("args.num_hidden:", args.num_hidden)
print("args.weight_decay:", args.weight_decay)
print('sens_num:',sens.unique(return_counts=True)) 

num_nodes = feature.shape[0]
nfeat = feature.shape[1]
nhidden = args.num_hidden
nclass = labels.max().item()+1
dropout = args.dropout
nlayer = args.nlayer
nheads = args.nheads

patch = None
eignvalue, eignvector = None, None
pe=None

# %%
def get_model(args):
    model = None
    global nfeat, nhidden, nclass, nlayer, dropout, nheads
    
    if args.model == 'GCN':
        nlayer=2
        model = GCN(nfeat, nhidden, nclass, dropout, nlayer, args=args)
    
    elif args.model == 'GAT':
        # nheads = 2
        model = GAT(nfeat, nhidden, nclass, dropout, nheads)
    elif args.model =='FairGP':
        if args.pe_method in ['adj','lap']:
            nfeat = nfeat+args.pe_dim
        model = FairGP(num_nodes=num_nodes, 
                       in_channels=nfeat, 
                       hidden_channels=nhidden, 
                       out_channels=nclass, 
                       activation=F.relu, 
                       layers=nlayer,
                       n_head=nheads)

    return model


# %%

# data preprocessing
print('='*10,"preprocessing before model creating",'='*10)

if args.model in ['FairGP']:
    # position encoding
    file_path = './pe_files/'+args.dataset+'/'
    file_name = args.dataset+'_'+args.pe_method+'_'+str(args.pe_dim)+'_'+str(args.seed)+'_'+str(args.self_loop)+'_pe.pt'

    if args.pe_method=="adj":
        print('adjancency position encoding!')
        # eignvalue, eignvector = laplacian_positional_encoding_spec(g, lm=args.e_dim)
        if os.path.exists(file_path+file_name):
            print('file exist:',file_name, 'load data')
            load_data = torch_load(file_path,file_name)
            eignvalue, eignvector=load_data
        else:
            sp_adj = edge_index_2_sparse_mx(edge_index)
            eignvalue, eignvector = adjacency_positional_encoding(sp_adj, args.pe_dim)
            torch_save(file_path,file_name,[eignvalue, eignvector])
        feature = torch.cat((feature, eignvector), dim=1)
    elif args.pe_method=="lap":
        print('laplacian position encoding!')
        if os.path.exists(file_path+file_name):
            print('file exist:',file_name, 'load data')
            load_data = torch_load(file_path,file_name)
            eignvalue, eignvector=load_data
        else:
            sp_adj = edge_index_2_sparse_mx(edge_index)
            eignvalue, eignvector = laplacian_positional_encoding(sp_adj, args.pe_dim)
            torch_save(file_path,file_name,[eignvalue, eignvector])
        feature = torch.cat((feature, eignvector), dim=1)
    else:
        print('no position encoding!')
        
    n_patch = args.n_patch

    method=args.patch_method
    patch, feature, labels, num_nodes = partition_patch(feature, edge_index, labels, n_patch, num_nodes, method=method, seed=123)



# %%
model = get_model(args)
# features = torch.FloatTensor(np.array(feature))
features = torch.FloatTensor(feature)
labels = torch.LongTensor(labels)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
sens = torch.FloatTensor(sens)

print('='*10,"preprocessing after model creating",'='*10)

if args.model in ['FairGP']:
    patch = patch.to(device)

# =========================global dataset to device====================
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
sens = sens.to(device)
edge_index = edge_index.to(device)

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model = model.to(device)
print(model)

# %%

val_loss_list, val_acc_list, val_auc_list, val_f1_list, val_sp_list, val_eo_list = [],[],[],[],[],[]
test_loss_list, test_acc_list, test_auc_list, test_f1_list, test_sp_list, test_eo_list = [],[],[],[],[],[]

patience = args.patience

# 
res = []

# best_val
rec_val=None
best_metric_val = -999998.0
metric_val = -999999.0
# best_test
rec_test=None
best_metric_test = -999998.0
metric_test = -999999.0
print('dataset: ',args.dataset,' model: ',args.model)
if args.metric==1: # acc
    print('metric: acc')
elif args.metric==2: # loss
    print('metric: loss')
elif args.metric==3: # -sp-eo
    print('metric: -sp-eo')
elif args.metric==4: # val_acc-val_parity-val_equality
    print('metric: acc-sp-eo')
elif args.metric==5: # val_f1-val_parity-val_equality
    print('metric: f1-sp-eo')
elif args.metric==6: # val_auc-val_parity-val_equality
    print('metric: auc-sp-eo')
elif args.metric==7: # val_acc-val_parity
    print('metric: acc-sp')
elif args.metric==8: # val_acc-val_equality
    print('metric: acc-eo')

counter = 0
train_start = time.time()
print('='*10,"Start Training: ",args.dataset,'='*10)

accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)


def general_train_step(model, features, edge_index, labels, optimizer, idx_train):
    model.train()
    optimizer.zero_grad()
    logits = model(features, edge_index)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return logits

def fairgp_train_step(model, features, edge_index, labels, patch, optimizer, idx_train):
    model.train()
    optimizer.zero_grad()
    logits = model(features, patch, edge_index)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return logits


# acc,auc,f1
def validation(logits, labels, idx):
    # val_loss = F.cross_entropy(logits[idx], labels[idx]).item()
    acc = accuracy(logits[idx].cpu(), labels[idx].cpu()).item()
    auc = roc_auc_score(labels[idx].cpu().numpy(), F.softmax(logits,dim=1)[idx,1].detach().cpu().numpy())
    f1 = f1_score(labels[idx].cpu().numpy(),logits[idx].detach().cpu().argmax(dim=1))
    # val_auc = roc_auc_score(labels[idx].cpu().numpy(), F.softmax(logits, dim=1)[idx, 1].detach().cpu().numpy())
    
    return acc, auc, f1


logits = None
val_sp, val_eo = None, None
test_sp, test_eo = None, None

val_acc,val_auc_roc,val_f1 = None, None, None
val_speo,test_speo = None,None


for epoch in range(args.epochs):

    
    if args.model=='FairGP':
        logits = fairgp_train_step(model, features, edge_index, labels, patch, optimizer, idx_train)
    else:
        logits = general_train_step(model, features, edge_index, labels, optimizer, idx_train)

    # model.eval()
    with torch.no_grad():
        model.eval()
        val_loss = F.cross_entropy(logits[idx_val], labels[idx_val]).item()
        val_acc,val_auc_roc,val_f1 = validation(logits, labels, idx_val)
        if args.have_sens: 
            if args.dataset in ['aminer_s', 'aminer_l']:
                val_sp, val_eo = fair_metric_new(labels, sens, torch.argmax(logits, dim=1), idx_val)
            else:
                val_sp, val_eo = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_val)
            val_speo = val_sp + val_eo
        
        test_loss = F.cross_entropy(logits[idx_test], labels[idx_test]).item()
        test_acc,test_auc_roc,test_f1 = validation(logits, labels, idx_test)
        if args.have_sens:  
            if args.dataset in ['aminer_s', 'aminer_l']:
                test_sp, test_eo = fair_metric_new(labels, sens, torch.argmax(logits, dim=1), idx_test)
            else:
                test_sp, test_eo = fair_metric(labels, sens, torch.argmax(logits, dim=1), idx_test)
            test_speo = test_sp + test_eo
            
            res.append([100 * test_acc, 100 * test_sp, 100 * test_eo, 100 * test_f1, 100 * test_auc_roc, (epoch+1)])
        else:
            res.append([100 * test_acc, 100 * test_f1, 100 * test_auc_roc, (epoch+1)])
        
    if args.metric==1: # acc
        metric_val = val_acc
        metric_test = test_acc
    elif args.metric==2: # loss
        metric_val = -val_loss
        metric_test = -test_loss
    elif args.metric==3: # -sp-eo
        metric_val = (-val_sp-val_eo)
        metric_test = (-test_sp-test_eo)
    elif args.metric==4: # val_acc-val_parity-val_equality
        metric_val = (val_acc-val_sp-val_eo)
        metric_test = (test_acc-test_sp-test_eo)
    elif args.metric==5: # val_f1-val_parity-val_equality
        metric_val = (val_f1-val_sp-val_eo)
        metric_test = (test_f1-test_sp-test_eo)
    elif args.metric==6: # val_auc-val_parity-val_equality
        metric_val = (val_auc_roc-val_sp-val_eo)
        metric_test = (test_auc_roc-test_sp-test_eo)
    elif args.metric==7: # val_acc-val_parity
        metric_val = (val_acc-val_sp)
        metric_test = (test_acc-test_sp)
    elif args.metric==8: # val_acc-val_equality
        metric_val = (val_acc-val_eo)
        metric_test = (test_acc-test_eo)
        

    if metric_val > best_metric_val and epoch>5 and val_speo>0:
        if args.have_sens:
            # if val_sp*val_eo>0:
            best_metric_val = metric_val
            rec_val = res[-1]
            counter = 0
            
        else:
            best_metric_val = metric_val
            rec_val = res[-1]
            counter = 0
    else:
        counter += 1   

    if metric_test > best_metric_test and epoch>5 and test_speo>0:
        if args.have_sens:
            # if test_sp*test_eo>0:
            best_metric_test = metric_test
            rec_test = res[-1]
        else:
            best_metric_test = metric_test
            rec_test = res[-1] # list
        
    if (epoch+1)%2==0:
        if args.have_sens:
            print('epoch:{:05d}, val_loss:{:.4f}, test_acc:{:.4f}, parity:{:.4f}, equality:{:.4f}, f1:{:.4f}, auc:{:.4f}'.format(epoch+1, val_loss, 100 * test_acc, 100 * test_sp, 100 * test_eo, 100 * test_f1, 100 * test_auc_roc ))
        else:
            print('epoch:{:05d}, val_loss:{:.4f}, test_acc:{:.4f}, f1:{:.4f}, auc:{:.4f}'.format(epoch+1, val_loss, 100 * test_acc, 100 * test_f1, 100 * test_auc_roc ))
    
    # if counter>args.patience:
    if counter>20 and epoch>200:
        print('patience touch break! epoch:',epoch)
        break
    
train_end = time.time()
train_time = (train_end-train_start)
print('success train data, time is:{:.3f}'.format(train_time))

max_memory_cached = torch.cuda.max_memory_cached(device=device) / 1024 ** 2 
max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
print("Max memory cached:", max_memory_cached, "MB")
print("Max memory allocated:", max_memory_allocated, "MB")

print('best val -- acc: {:.4f}, parity: {:.4f}, equality: {:.4f}, f1: {:.4f}, auc: {:.4f}, epoch: {:04d}'.format(rec_val[0],rec_val[1],rec_val[2],rec_val[3],rec_val[4],rec_val[5]))
print('best test -- acc: {:.4f}, parity: {:.4f}, equality: {:.4f}, f1: {:.4f}, auc: {:.4f}, epoch: {:04d}'.format(rec_test[0],rec_test[1],rec_test[2],rec_test[3],rec_test[4],rec_test[5]))

    



# %%
log_data = {
    'method': [args.model],
    'dataset': [args.dataset],
    'label_number':[args.label_number],
    'feat_norm':[args.feat_norm],
    'sens': [args.sens_attr],
    'is_self_loop': [args.self_loop],
    'pe_method': [args.pe_method],
    'pe_dim': [args.pe_dim],
    'patch_method': [args.patch_method],
    'n_patch': [args.n_patch],
    'num_hops': [args.num_hops],
    'nlayer': [args.nlayer],
    'nheads': [args.nheads],
    'epochs': [args.epochs],
    'num_hidden': [args.num_hidden],
    'dropout': [args.dropout],
    'lr': [args.lr],
    'seed': [args.seed],
    'metric': [args.metric],
    'weight_decay': [args.weight_decay],
    'train_time(s)': [train_time],
    
    'val_best_epoch': [rec_val[-1]],
    'val_best_acc': [rec_val[0]],
    'val_best_sp': [rec_val[1]],
    'val_best_eo': [rec_val[2]],
    'val_best_f1': [rec_val[3]],
    'val_best_auc': [rec_val[4]],
    
    'test_best_epoch': [rec_test[-1]],
    'test_best_acc': [rec_test[0]],
    'test_best_sp': [rec_test[1]],
    'test_best_eo': [rec_test[2]],
    'test_best_f1': [rec_test[3]],
    'test_best_auc': [rec_test[4]],
    
    'args':[args]
}

train_logs = pd.DataFrame(log_data)

import os
logs_path = './logs/'
train_log_save_file=logs_path+'fairgt'+'_gpu_'+str(args.gpu)+'_train_log.csv'


if os.path.exists(train_log_save_file): # add
    train_logs.to_csv(train_log_save_file, mode='a', index=False, header=0)
else: # create
    train_logs.to_csv(train_log_save_file, index=False)

print('='*10,"End Log",'='*10)

# %%



