import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from inspect import isfunction
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

from utils.misc import exists


# 自注意力机制，侧重于挖掘序列内部的依赖关系
class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        # dropout层用于进行正则化，以一定的概率随机将神经元的输出置零（即丢弃
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size() # 此处的x就是emb，B是batch size，T是tokens数量，C是token对应的特征维度
        # nh是头数，hs = d_model / head_num，相当于有head_num个注意力矩阵，每个value矩阵是序列长度 * 编码后维度
        # 这里的编码后的维度有dq = dk= dv = d_model / head_num
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 这里的y就是按照公式得到的注意力的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

# 交叉注意力机制，侧重于两个序列之间的信息交互
class CrossAttention(nn.Module):
    def __init__(self,
                 condition_seq_len,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=False, #  causal mask to ensure that attention is only applied to the left in the input sequence
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # 此处的encoder_output是condition
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

# 正弦编码，嵌入时间步t
class SinusoidalPosEmb(nn.Module):
    def __init__(self, diffusion_step, n_embd, rescale_steps=4000):
        '''
        :param diffusion_step: 最大时间步，对应执行扩散步骤的总数
        :param n_embd: 目标嵌入向量的维度
        :param rescale_steps: 用于调整位置编码的频率
        '''
        super().__init__()
        self.n_embd = n_embd
        self.diffusion_step = float(diffusion_step)
        self.rescale_steps = float(rescale_steps)

    def forward(self, timestep):
        '''
        :param x: 当前时间步
        :return: 时间步x对应的嵌入
        '''
        timestep = timestep / self.diffusion_step * self.rescale_steps
        device = timestep.device
        half_dim = self.n_embd // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 实现自适应层归一化，其中归一化的规模（scale）和偏移（shift）参数是基于输入的时间步动态生成的。
# AdaLayerNorm 在需要考虑时间动态的模型中特别有用，如扩散模型。
# 在这些模型中，每个时间步可能需要不同的处理方式，而自适应层归一化允许模型根据当前的时间步调整其行为。
# 这可以增加模型处理时序数据的灵活性和效率。
# 更适合于处理那些特征间相互依赖性较强的任务。
class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        '''
        :param n_embd: 嵌入的维度
        :param diffusion_step: 最大时间步，对应执行扩散步骤的总数
        :param emb_type: 使用的嵌入类型
        '''
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        '''
        :param x: 输入x
        :param t
        imestep: 当前时间步
        :return:
        '''

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        # emb shape = (batch, 1, n_embd*2)
        scale, shift = torch.chunk(emb, 2, dim=2)
                
        '''
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(2)
        scale, shift = torch.chunk(emb, 2, dim=1)
        '''
        # 两种方式得到的sacle和shift维度相同，即现在对每个参数都生成了相同的scale和shift

        # 此处的乘法是元素对元素相乘，因此要求scale与x具有相同的shape,如果不同，需要进行广播
        x = self.layernorm(x) * (1 + scale) + shift
        return x

# 实现自适应实例归一化，其中归一化的规模（scale）和偏移（shift）参数是基于输入的时间步动态生成的。
# 使得模型在处理每个样本时可以根据额外的信息（如时间步骤）来动态调整其归一化策略，从而提高其适应不同数据特征的能力。
# 更适合处理高度个体化的数据
class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        '''
        :param n_embd: 嵌入的维度
        :param diffusion_step: 最大时间步，对应执行扩散步骤的总数
        :param emb_type: 使用的嵌入类型
        '''
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x

# 通用transformer块
class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 class_type='adalayernorm',    # 影响归一化方式
                 class_number=1000,            # 仅类生图 selfcondition使用
                 condition_seq_len=77,         # attention模块参数,未使用
                 n_embd=1024,                  # attention模块参数,the embed dim, 对应输入的特征维度
                 condition_dim=1024,           # attention模块参数，条件对应的特征维度
                 n_head=16,                    # attention模块参数,the number of heads         
                 seq_len=256,                  # attention模块参数,the max length of sequence
                 attn_pdrop=0.1,               # attention模块参数,attention dropout prob
                 resid_pdrop=0.1,              # attention模块参数,residual attention dropout prob
                 mlp_hidden_times=4,           # mlp网络结构参数，升维倍数
                 activate='GELU',              # 激活函数类型
                 attn_type='full',             # 注意力机制类型
                 diffusion_step=100,           # 最大时间步，对应执行扩散步骤的总数
                 timestep_type='adalayernorm', # 归一化方式
                 mlp_type = 'fc',              # mlp网络结构类型
                 use_biomarks = False,         # 是否使用AP相关的生物标志物，即手工特征
                 biomarks_embd = 32,           # 生物标志物维度
                 level1_biomarks = 0,          # 0：提前使用mlp投影生物标志物；2：不投影
                 level2_biomarks = 0,          # 0: 特征融合；1：分别使用两个独立的交叉注意力机制引入
                 level3_biomarks = 0,          # 0: 拼接+再次投影； 1：拼接(相加)；2：仅针对分别引入，使用串行交叉注意力机制 
                 ):
        super().__init__()
        self.attn_type = attn_type

        # 文生图 selfcross， 类生图 selfcondition， 无条件生成self
        # 根据任务类型，选取归一化方式(self.ln1 & self.ln2)，两个根据时间步t的自适应 & 默认的LayerNorm
        if attn_type in ['selfcross', 'selfcondition', 'self']: 
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            self.ln1 = nn.LayerNorm(n_embd)
        
        self.ln2 = nn.LayerNorm(n_embd)
        # 根据任务类型，选取注意力机制
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if attn_type == 'selfcondition':
                if 'adalayernorm' in class_type:
                    self.ln2 = AdaLayerNorm(n_embd, class_number, class_type)
                else:
                    self.ln2 = AdaInsNorm(n_embd, class_number, class_type)
        # 目前采用的注入条件信息的注意力机制
        elif attn_type == 'selfcross':
            self.attn1 = FullAttention(
                    n_embd=n_embd,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop, 
                    resid_pdrop=resid_pdrop,
                    )
            self.attn2 = CrossAttention(
                    condition_seq_len,
                    n_embd=n_embd,
                    condition_embd=condition_dim,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
            # 如何引用手工特征
            self.use_biomarks = use_biomarks
            self.level2_biomarks = level2_biomarks
            self.level3_biomarks = level3_biomarks
            if use_biomarks and level2_biomarks == 1:
                self.attn3 = CrossAttention(
                    condition_seq_len,
                    n_embd=n_embd,
                    condition_embd=biomarks_embd,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    )
                if level3_biomarks == 0:
                   self.parallel_proj = nn.Linear(n_embd + n_embd, n_embd)
                if 'adalayernorm' in timestep_type:
                    self.ln1_2 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
                else:
                    print("timestep_type wrong")
        else:
            print("attn_type error")

        # 选取激活函数
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        # 确定后续mlp，提升非线性关系建模能力
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    
    def forward(self, x, encoder_output, biomarks, timestep, mask=None):
        '''
        :param x: 输入x
        :param encoder_output: 条件编码器输出，作为条件
        :param timestep: 当前时间步
        :param mask: 掩码
        :return:
        '''
        if self.attn_type == "selfcross":
            a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            a2, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
            # 如果使用生物标志物并选择分别使用交叉注意力机制引入
            if self.use_biomarks and self.level2_biomarks == 1:
                if self.level3_biomarks == 0:
                    a3, att = self.attn3(self.ln1_2(x, timestep), biomarks, mask=mask)
                    a_proj = self.parallel_proj(torch.cat([a2, a3], dim=-1)) 
                    x = x + a_proj
                elif self.level3_biomarks == 1:
                    a3, att = self.attn3(self.ln1_2(x, timestep), biomarks, mask=mask)
                    a_paral = a2 + a3 
                    x = x + a_paral
                elif self.level3_biomarks == 2:
                    x = a2 + x
                    a3, att = self.attn3(self.ln1_2(x, timestep), biomarks, mask=mask)
                    x = a3 + x
                else:
                    print("level3_biomarks wrong") 
            else:
                x = x + a2
        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            x = x + self.mlp(self.ln2(x, encoder_output.long()))   # only one really use encoder_output
            return x, att
        else:  # 'self'
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a 
        
        x = x + self.mlp(self.ln2(x))
        return x, att

class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n =  x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)


def sinusoidal_positional_encoding(tokens_num, d_model):
    """
    生成基于正弦和余弦的位置嵌入矩阵

    参数:
    d_model (int): 嵌入维度
    tokens_num (int): tokens数量

    返回:
    np.ndarray: 形状为 (tokens_num, d_model) 的位置嵌入矩阵
    """
    prev_d_model = d_model
    if d_model % 2 == 1:
        d_model = d_model + 1
    pe = np.zeros((tokens_num, d_model))
    position = np.arange(0, tokens_num).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe[:,:prev_d_model]
    
# AP模型校准任务
class AP2ParaTransformer(nn.Module):
    def __init__(
            self,
            condition_seq_len=77,  # attention模块参数，未使用
            n_layer=4,             # Block数量 
            n_embd=1024,           # attention模块参数，the embed dim, 对应输入的空间维度
            condition_dim=512,     # attention模块参数，条件对应的空间维度
            n_head=16,             # attention模块参数，the number of heads
            content_seq_len=1024,  # attention模块参数，the max length of sequence
            attn_pdrop=0,          # attention模块参数，attention dropout prob 
            resid_pdrop=0,         # attention模块参数，residual attention dropout prob
            mlp_hidden_times=4,    # mlp网络结构参数，升维倍数
            block_activate=None,   # 激活函数类型
            attn_type='selfcross', # 注意力机制类型
            diffusion_step=1000,   # 最大时间步，对应执行扩散步骤的总数
            timestep_type='adalayernorm',  # 归一化方式
            mlp_type='fc',         # mlp网络结构类型
            use_biomarks = False,  # 是否使用AP相关的生物标志物，即手工特征
            biomarks_embd = 32,    # 生物标志物维度
            level1_biomarks = 0,   # 0：提前使用mlp投影生物标志物；2：不投影
            level2_biomarks = 0,   # 0: 特征融合；1：分别使用两个独立的交叉注意力机制引入
            level3_biomarks = 0,   # 0: 拼接+再次投影； 1：拼接(相加)；2：仅针对分别引入，使用串行交叉注意力机制
            use_content_emb = False, # 是否使用一层全连接层得到embedding对应的输入
            content_times = 1, # Embedding层升维倍数
            use_pos_emb = False, # 是否使用位置编码
            pos_emb_type = 0, # 0：正弦编码，1：可学习位置编码
            use_tokens = False, # 是否将\theta认为是tokens_num
            tokens_nums = 1, # tokens 数量
            checkpoint=False,
    ): 
        super().__init__()
        self.use_checkpoint = checkpoint

        # 记录tokens_num
        self.use_tokens = use_tokens
        self.tokens_nums = tokens_nums
        if use_tokens:
            self.tokens_nums = n_embd
            n_embd = 1
                    
        # Initial embedding layer
        self.use_content_emb = use_content_emb
        if use_content_emb:
            self.content_emb = nn.Sequential(
                nn.Linear(n_embd, n_embd * content_times),
            )
            n_embd = n_embd * content_times
        
        if not use_tokens and tokens_nums > 1:
            assert tokens_nums == 9
            n_embd = n_embd//tokens_nums
            
        # 使用位置embedding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            pos_emb = sinusoidal_positional_encoding(tokens_nums, n_embd)
            if pos_emb_type == 1:
                # Will use learnbale sin-cos embedding:
                self.pos_embed = nn.Parameter(torch.zeros(1, tokens_nums, n_embd), requires_grad=True)
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, tokens_nums, n_embd), requires_grad=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_emb).float().unsqueeze(0))
        
        # 手工特征使用阶段
        self.use_biomarks = use_biomarks
        self.level2_biomarks =level2_biomarks
        if self.use_biomarks == True:
            if level1_biomarks == 0:
                self.level1_proj = nn.Linear(biomarks_embd, n_embd)
                biomarks_embd = n_embd
            else:    
                self.level1_proj = None
            if level2_biomarks == 0:
                # 修改条件的维度
                condition_dim = condition_dim + biomarks_embd
                if level3_biomarks == 0:
                    self.level3_proj = nn.Linear(condition_dim, condition_dim)
                else:
                    self.level3_proj = None
        # transformer
        assert attn_type == 'selfcross'
        all_attn_type = [attn_type] * n_layer

        self.blocks = nn.Sequential(*[Block(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=content_seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            attn_type=all_attn_type[n],
            condition_dim=condition_dim,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
            mlp_type=mlp_type,
            use_biomarks = use_biomarks,  # 是否使用AP相关的生物标志物，即手工特征
            biomarks_embd = biomarks_embd,    # 生物标志物维度
            level1_biomarks = level1_biomarks,   # 0：提前使用mlp投影生物标志物；2：不投影
            level2_biomarks = level2_biomarks,   # 0: 特征融合；1：分别使用两个独立的交叉注意力机制引入
            level3_biomarks = level3_biomarks,   # 0: 拼接+再次投影； 1：拼接(相加)；2：仅针对分别引入，使用串行交叉注意力机制 
        ) for n in range(n_layer)])

        # final prediction head
        if use_content_emb:
            self.to_logits = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.Linear(n_embd, n_embd // content_times),
            )
        else:
            self.to_logits = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.Linear(n_embd, n_embd),
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}  # if p.requires_grad}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups
    #  cond if use Cost: batch x T(2000) x input_dim(1)  -> batch x 1 x repr_dims(64)
    def forward(
            self,
            input,
            t,
            cond,
            biomarks,
            ):
        ## 使用生物标志物作为补充信息
        if self.use_biomarks:
            assert biomarks is not None
            if biomarks.dim() == 2:  # Check if the biomarks is 2D
                biomarks = biomarks.unsqueeze(1)  # Add a sequence length dimension
            if exists(self.level1_proj):
                bio_cond = self.level1_proj(biomarks)
            else:
                bio_cond = biomarks
            if self.level2_biomarks == 0:
                cond = torch.cat([cond, bio_cond], dim=2)
                # print(f"the shape of bio_cond = {bio_cond.shape} in forward()")
                # print(f"the shape of cond = {cond.shape} in forward()")
                if exists(self.level3_proj):
                    cond = self.level3_proj(cond) 
        else:
            bio_cond = biomarks
        # print(f"the shape of input = {input.shape} in forward()")
        # print(f"the shape of bio_cond = {bio_cond.shape} in forward()")
        # print(f"the shape of cond = {cond.shape} in forward()")
        
        emb = input
        ## from batch x n_embd -> batch x 1 x n_embd
        if emb.dim() == 2:  # Check if the input is 2D
            ## batch x 1 x 9(embd)
            emb = emb.unsqueeze(1)  # Add a sequence length dimension
            if self.use_tokens:
                ## 将9个参数视为token_num, 将其变为batch x 9(embd) x 1 
                emb = emb.transpose(1, 2)  # Add a sequence length dimension
        # print(f"the shape of emb = {emb.shape} in forward()")
        # print(f"the shape of cond = {cond.shape} in forward()")
            
        ## 使用全连接层获得输入对应的embedding
        if self.use_content_emb:
            emb = self.content_emb(emb)
            # print(f"the shape of content_emb = {emb.shape}")
        if not self.use_tokens and self.tokens_nums > 1:
            B, _, C = emb.shape
            emb = emb.reshape(B, self.tokens_nums, -1)
            # print(f"the shape of emb = {emb.shape} in forward()")
        ## 使用位置编码
        if self.use_pos_emb:
            emb = emb + self.pos_embed
            # print(f"the shape of pos_emb = {emb.shape} in forward()")    
            
        for block_idx in range(len(self.blocks)):
            if self.use_checkpoint == False:
                emb, att_weight = self.blocks[block_idx](emb, cond, bio_cond,
                                                         t.cuda())  # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            else:
                emb, att_weight = checkpoint(self.blocks[block_idx], emb, cond, t.cuda())
        # print(f"the shape of block emb = {emb.shape} in forward()")
        if self.use_content_emb:
            logits = self.to_logits(emb) # B x (Ld+Lt) x n
            out = logits
        else:
            out = emb
        # print(f"the shape of logits = {logits.shape} in forward()")
        if out.dim() == 3:  # Check if the input is 2D
            B, _, C = out.shape
            out = out.reshape(B, -1)
            # print(f"the shape of out = {out.shape} in forward()")
        return out