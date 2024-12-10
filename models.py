import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nlayer=2, args=None):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.nlayer = nlayer
        self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels))
        for layer in range(self.nlayer-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x



class GAT(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout, heads=1, args=None):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden, heads=self.heads, dropout=dropout)
        self.conv2 = GATConv(hidden * self.heads, out_channels, dropout=dropout)

    def forward(self, x, edge_index, return_attn=False):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_alpha = self.conv1(x, edge_index,return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        if return_attn:
            return x , edge_alpha[1]
        return logits




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None): # (B, H, L_q, d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # self.label_same_matrix = self.label_same_matrix.to(attn.device)
        # attn = attn * self.label_same_matrix * 2 + attn * (1-self.label_same_matrix)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn # (B, H, L_q, d_v)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        # q, k, v [n_patch, subnode, d]
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0) # batch/patch size
        N_q = q.size(1) # subnode
        B_k = k.size(0) # batch/patch size
        N_k = k.size(1) # subnode
        B_v = v.size(0) # batch/patch size
        N_v = v.size(1) # subnode

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q) # n_head x dv = channels
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B x heads x subnode x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask) # B x heads x subnode x dv

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1) # B x h x N x dv    :B x subnode x heads x dv --> B x subnode x channels
        q = self.fc(q) # B x subnode x channels
        q = q + residual

        return q, attn


class FFN1(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN1, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class FFN2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(FFN2, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.lin1(x))
        return x


class ICALayer(nn.Module):
    def __init__(self, n_head, channels, dropout=0.1):
        super(ICALayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN1(channels, dropout)
        self.patch_ffn = FFN1(channels, dropout)
        self.fuse_lin = nn.Linear(2 * channels, channels)


    def forward(self, x, patch, attn_mask=None): # in: N x hidden --> middle: B x sub_N x hidden --> out: N x hidden
        x = self.node_norm(x)
        patch_x = x[patch] # x[[n_patch,n_sub_nodes]]-->[n_patch, n_sub_nodes, hidden_dim]
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)  # B x sub_N x (h*dv) 子图内部的节点attention
        patch_x = self.node_ffn(patch_x)

        x[patch] = patch_x # B x sub_N x (hidden)--> N x hidden

        return x


class ICABlock(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, dropout1=0.5, dropout2=0.1):
        super(ICABlock, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN2(in_channels, hidden_channels)
        self.ICALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.ICALayers.append(
                ICALayer(n_head, hidden_channels, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, patch):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1) # only true node id [n_patch, subgraph_nodes]
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int() # only true node pairs shape = [n_patch,n_patch]

        x = self.attribute_encoder(x) # FFN
        for i in range(0, self.layers):
            x = self.ICALayers[i](x, patch, attn_mask) # B x sub_N x (hidden)
        x = self.dropout(x)
        x = self.classifier(x) # B x sub_N x (hidden)
        return x


class FairGP(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, layers: int, n_head: int, dropout1=0.5, dropout2=0.1):
        super(FairGP, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = nn.Dropout(dropout1)
        
        self.ica = ICABlock(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head, dropout1, dropout2)


    def forward(self, x, patch, edge_index):
        z = self.ica(x, patch)
        return z
    