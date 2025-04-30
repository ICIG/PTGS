import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

#动态门控

class CrossAttentionGate(nn.Module):
    def __init__(self, vis_channels, audio_channels, hidden_dim=32):
        super().__init__()
        # 视觉分支（对应空间特征）
        self.vis_proj = nn.Sequential(
            nn.Conv1d(vis_channels, hidden_dim, 1),
            nn.ReLU()
        )
        # 音频分支
        self.aud_proj = nn.Sequential(
            nn.Conv1d(audio_channels, hidden_dim, 1),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        # 注意力权重生成
        self.attn = nn.Sequential(
            nn.Conv1d(hidden_dim*2, 1, 1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Sigmoid()
        )

    def forward(self, vis_feat, aud_feat):
        # vis_feat: [N, vis_channels] -> [1, vis_channels, N]
        # aud_feat: [1, audio_channels] -> [1, audio_channels, 1]
        vis = self.vis_proj(vis_feat.unsqueeze(0).permute(0,2,1)) # [1, hidden, N]
        aud = self.aud_proj(aud_feat.unsqueeze(-1)) # [1, hidden, 1]
        
        # 广播相加
        combined = torch.cat([vis, aud.expand_as(vis)], dim=1) # [1, 2*hidden, N]
        attn = self.attn(combined).permute(0,2,1) # [1, N, 1]
        return attn.squeeze(0) # [N, 1]

'''
# 动态门控--线性层版
class CrossAttentionGate(nn.Module):
    def __init__(self, vis_channels, audio_channels, hidden_dim=32):
        super().__init__()
        # 视觉分支（线性层替代）
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_channels, hidden_dim),  # 替换Conv1d
            nn.ReLU()
        )
        # 音频分支（线性层替代）
        self.aud_proj = nn.Sequential(
            nn.Linear(audio_channels, hidden_dim),  # 替换Conv1d
            nn.ReLU()
        )
        # 注意力权重生成（线性层替代）
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),  # 替换Conv1d
            nn.Sigmoid()
        )

    def forward(self, vis_feat, aud_feat):
        """
        输入维度说明：
        vis_feat: [N, vis_channels] (例如[N,36]空间特征)
        aud_feat: [B, audio_channels] (例如[1,32]音频特征)
        """
        # 投影视觉特征
        vis = self.vis_proj(vis_feat)  # [N, hidden]
        
        # 投影音频特征并扩展维度
        B, C = aud_feat.shape
        aud = self.aud_proj(aud_feat)        # [B, hidden]
        aud = aud.unsqueeze(1)               # [B, 1, hidden]
        aud = aud.expand(-1, vis.shape[0], -1) # [B, N, hidden]
        aud = aud.reshape(-1, aud.shape[-1])  # [B*N, hidden]
        
        # 重复视觉特征以匹配音频维度
        vis = vis.repeat(B, 1)                # [B*N, hidden]
        
        # 拼接特征
        combined = torch.cat([vis, aud], dim=1)  # [B*N, 2*hidden]
        
        # 生成注意力权重
        attn = self.attn(combined)            # [B*N, 1]
        attn = attn.reshape(B, -1)            # [B, N]
        return attn.transpose(0, 1).squeeze(-1)  # [N, B] → [N] (当B=1时)
'''

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        # 动态门控融合
        self.cross_attn_gate = CrossAttentionGate(
            vis_channels=self.in_dim,  # 空间编码维度(36)
            audio_channels=self.audio_dim, # 音频维度(32)
            hidden_dim=self.in_dim
        )

        # 新增音频-眼部融合门控
        self.audio_eye_gate = CrossAttentionGate(
            vis_channels=self.eye_dim,  # 眼部特征维度6
            audio_channels=self.audio_dim,  # 音频维度32
            hidden_dim=self.eye_dim
        )
        
        # 新增空间融合门控（来自MouthMotionNetwork）
        self.space_gate = CrossAttentionGate(
            vis_channels=self.in_dim,  # 空间编码36
            audio_channels=self.audio_dim + self.eye_dim,  # 融合后维度
            hidden_dim=32
        )

        # 添加投影层
        self.audio_proj = nn.Linear(32, 36)  # 32 → 36
        self.audio_eye_proj = nn.Linear(32, 6)
        self.vis_proj = nn.Linear(36, 36)    # 保持空间编码维度


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)
        #print(f"enc_x: {enc_x.shape}")
        enc_a = self.encode_audio(a)
        enc_a_e = enc_a
        #print(f"enc_a: {enc_a.shape}")
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        #print(f"enc_a-r: {enc_a.shape}")
        aud_ch_att = self.aud_ch_att_net(enc_x)
        #print(f"enc_ch_att: {aud_ch_att.shape}")
        enc_w = enc_a * aud_ch_att
        #print(f"enc_w: {enc_w.shape}")
        
        eye_att = torch.relu(self.eye_att_net(enc_x))
        #print(f"eye_att: {eye_att.shape}")

        eye_att_n = eye_att.norm(dim=-1, keepdim=True)
        #print(f"eye_att_n: {eye_att_n.shape}")

        enc_e = self.exp_encode_net(e[:-1])
        #print(f"enc_e: {enc_e.shape}")
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        #print(f"enc_e-c: {enc_e.shape}")
        
        #------------------------------------------------------
        #                   动态门控融合
        #------------------------------------------------------

        enc_e = enc_e.unsqueeze(0)          # [6] -> [1,6]
        enc_e = enc_e.repeat(enc_x.shape[0], 1) # [1,6] => [N,6]
        
        #音频-眨眼融合
        eye_audio_weight = self.audio_eye_gate(enc_e, enc_a_e)  # [N,1]
        #print(f"eye_audio_weight: {eye_audio_weight.shape}")
        enc_a_expanded = enc_a
        #fused_eye_audio = eye_audio_weight * self.audio_eye_proj(enc_a_expanded) + (1 - eye_audio_weight) * enc_e

        fused_eye_audio = eye_audio_weight * self.audio_eye_proj(enc_a_expanded) + enc_e

        #print(f"fused_eye_audio: {fused_eye_audio.shape}")
        
        eye_att = eye_audio_weight

        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            #print(f"c: {c.shape}")
            h = torch.cat([enc_x, enc_w, fused_eye_audio, c], dim=-1)
            #print(f"h-c: {h.shape}")
        else:
            h = torch.cat([enc_x, enc_w, fused_eye_audio], dim=-1)
            #print(f"h-e: {h.shape}")

        #print(f"h-in: {h.shape}")
        h = self.sigma_net(h)
        #print(f"h-out: {h.shape}")
        






        #音频空间融合特征
        # 生成空间自适应的音频权重
        #attn_weight = self.cross_attn_gate(enc_x, enc_a) # [N,1]
        #print(f"attn_weight shape: {attn_weight.shape}")
         # 广播音频特征并与空间编码融合
        #enc_a_expanded = enc_a.repeat(enc_x.shape[0], 1) # [N,32]
        #print(f"enc_a_expanded shape: {enc_a_expanded.shape}")
        #enc_a_expanded = self.audio_proj(enc_a_expanded)  # [N,36]
        #print(f"enc_a_expanded-p shape: {enc_a_expanded.shape}")
        # 门控融合（替换原有拼接）
        #fused_feat = attn_weight * enc_a_expanded + (1 - attn_weight) * enc_x
        #print(f"fused_feat shape: {fused_feat.shape}")
        #残差连接
        #fused_feat = enc_x + attn_weight * enc_a_expanded
        #print(f"fused_feat-c shape: {fused_feat.shape}")
        

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        #self.sigma_net = MLP(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        # 动态门控融合
        self.cross_attn_gate = CrossAttentionGate(
            vis_channels=self.in_dim,  # 空间编码维度(36)
            audio_channels=self.audio_dim, # 音频维度(32)
            hidden_dim=32
        )

        # 添加投影层
        self.audio_proj = nn.Linear(32, 36)  # 32 → 36
        self.vis_proj = nn.Linear(36, 36)    # 保持空间编码维度
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a):
        
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)
        #print(f"enc_x shape: {enc_x.shape}")
        enc_a = self.encode_audio(a)
        #print(f"enc_a shape: {enc_a.shape}")
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        #-------------------------------------------------------
        #              动态门控融合（音频和空间）
        #-------------------------------------------------------
        
        # 生成空间自适应的音频权重
        attn_weight = self.cross_attn_gate(enc_x, enc_a) # [N,1]
        #print(f"attn_weight shape: {attn_weight.shape}")
         # 广播音频特征并与空间编码融合
        enc_a_expanded = enc_a.repeat(enc_x.shape[0], 1) # [N,32]
        #print(f"enc_a_expanded shape: {enc_a_expanded.shape}")
        enc_a_expanded = self.audio_proj(enc_a_expanded)  # [N,36]
        #print(f"enc_a_expanded-p shape: {enc_a_expanded.shape}")
        # 门控融合（替换原有拼接）
        fused_feat = attn_weight * enc_a_expanded + (1 - attn_weight) * enc_x
        #print(f"fused_feat shape: {fused_feat.shape}")
        #残差连接
        fused_feat = enc_x + attn_weight * enc_a_expanded
        #print(f"fused_feat-c shape: {fused_feat.shape}")
        
        h = torch.cat([fused_feat, enc_w], dim=-1)
        h = self.sigma_net(h)
        #print(f"h-s shape: {h.shape}")
        d_xyz = h * 1e-2
        #print(f"d_xyz shape: {d_xyz.shape}")
        

        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params
