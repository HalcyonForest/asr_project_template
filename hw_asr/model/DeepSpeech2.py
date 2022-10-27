from torch import nn
from torch.nn import Sequential
import torch
from hw_asr.base import BaseModel


    
    
class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(n_feats)
        
    def forward(self, spectrogram):
        spectrogram = spectrogram.transpose(2,3)
        spectrogram = self.layernorm(spectrogram)
        return spectrogram.transpose(2,3)
        
        
class ResCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResCNNBlock, self).__init__()
        
        self.ln1 = CNNLayerNorm(n_feats)
        self.gelu1 = nn.GELU()
        self.dp1 = nn.Dropout(dropout)
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        
        self.ln2 = CNNLayerNorm(n_feats)
        self.gelu2 = nn.GELU()
        self.dp2 = nn.Dropout(dropout)
        self.cn2 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        
    def forward(self, s):
        res_s = s
        s = self.ln1(s)
        s = self.gelu1(s)
        s = self.dp1(s)
        s = self.cn1(s)
        s = self.ln2(s)
        s = self.gelu2(s)
        s = self.dp2(s)
        s = self.cn2(s)
        s += res_s
        return s
    
    
class GRUplus(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first=True):
        super(GRUplus, self).__init__()
        
        self.gru = nn.GRU(rnn_dim, hidden_size, 1, batch_first=batch_first, bidirectional=True)
        self.layernorm = nn.LayerNorm(rnn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spectrogram):
        spectrogram = self.layernorm(spectrogram)
        spectrogram = self.gelu(spectrogram)
        spectrogram, _ = self.gru(spectrogram)
        spectrogram = self.dropout(spectrogram)
        return spectrogram
    
class MyModel(nn.Module):
    def __init__(self, n_cnn, n_rnn, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, **batch):
        super(MyModel, self).__init__()
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)
        self.rescnnblocks = Sequential(*[ResCNNBlock(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) for _ in range(n_cnn)])
        self.linear = nn.Linear(n_feats * 32, rnn_dim)
        self.grus = Sequential(*[GRUplus(rnn_dim, rnn_dim, dropout)] + [GRUplus(rnn_dim*2, rnn_dim, dropout, batch_first=False) for i in range(1, n_rnn)])
        self.net = nn.Sequential(
                nn.Linear(rnn_dim*2, rnn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(rnn_dim, n_class)
            )

    def forward(self, spectrogram, **batch):
            s = self.cnn(spectrogram.unsqueeze(1))
            s = self.rescnnblocks(s)
            sizes = s.size()
            s = s.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
            s = s.transpose(1, 2) # (batch, time, feature)
            s = self.linear(s)
            s = self.grus(s)
            s = self.net(s)
            return {"logits": s}
        
    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
