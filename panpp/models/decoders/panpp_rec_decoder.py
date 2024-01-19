# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.decoders.base import BaseDecoder
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample


@MODELS.register_module()
class PANPPRecDecoder(BaseDecoder):
    """Decoder for PANPP.

    Args:
        in_channels (int): Number of input channels.
        dropout_prob (float): Probability of dropout. Default to 0.5.
        teach_prob (float): Probability of teacher forcing. Defaults to 0.5.
        dictionary (dict or :obj:`Dictionary`): The config for `Dictionary` or
            the instance of `Dictionary`.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int, optional): Max sequence length. Defaults to 30.
        init_cfg (dict or list[dict], optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 128,
                 dictionary: Union[Dictionary, Dict] = None,
                 module_loss: Dict = None,
                 postprocessor: Dict = None,
                 max_seq_len: int = 30,
                 lstm_num: int = 2,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
            max_seq_len=max_seq_len)
        self.in_channels = in_channels
        # for encoder
        self.en_embedding = nn.Embedding(self.dictionary.num_classes, in_channels)
        self.en_attn = MultiHeadAttentionLayer(in_channels, 8)

        # for decoder
        self.max_seq_len = max_seq_len
        self.lstm_num = lstm_num
        self.de_lstms = nn.ModuleList()
        for i in range(self.lstm_num):
            self.de_lstms.append(nn.LSTMCell(in_channels, in_channels))
        self.de_embedding = nn.Embedding(self.dictionary.num_classes, in_channels)
        self.de_attn = MultiHeadAttentionLayer(in_channels, 8)
        self.de_cls = nn.Linear(in_channels + in_channels, self.dictionary.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward_train(
            self,
            feat: torch.Tensor,
            out_enc: Optional[torch.Tensor] = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        # encoder
        feat_holistic = self._encoder(feat)

        # decoder
        batch_size, feature_dim, H, W = feat.size()
        feat_flatten = feat.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        lstm_h = []
        for i in range(self.lstm_num):
            lstm_h.append((feat.new_zeros((batch_size, self.in_channels), dtype=torch.float32),
                           feat.new_zeros((batch_size, self.in_channels), dtype=torch.float32)))

        outputs = feat.new_zeros((batch_size, self.max_seq_len + 1, self.dictionary.num_classes),
                             dtype=torch.float32)

        trg_seq = []
        for target in data_samples:
            trg_seq.append(target.gt_text.padded_indexes.to(feat.device))
        trg_seq = torch.stack(trg_seq, dim=0)

        for t in range(self.max_seq_len + 1):
            if t == 0:
                feat_t = feat_holistic
            else:
                characters = trg_seq[:, t - 1]
                feat_t = self.de_embedding(characters)

            # lstms
            for i in range(self.lstm_num):
                inp = feat_t if i == 0 else lstm_h[i - 1][0]
                lstm_h[i] = self.de_lstms[i](inp, lstm_h[i])
            lstm_ht = lstm_h[-1][0]
            if t == 0:
                continue

            # attention
            out_t, _ = self.de_attn(lstm_ht, feat_flatten, feat_flatten)

            # character classification
            out_t = torch.cat((out_t, lstm_ht), dim=1)
            out_t = self.de_cls(out_t)

            # softmax would make loss not decline
            # out_t = self.softmax(out_t)
            outputs[:, t, :] = out_t
            # score, pre_seq = torch.max(out_t, dim=1)
        return outputs[:, 1:, :]

    def forward_test(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: Optional[torch.Tensor] = None,
            data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, C, 1, W)`.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing ``gt_text`` information.
                Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        # encoder
        feat_holistic = self._encoder(feat)

        # decoder
        batch_size, feature_dim, H, W = feat.size()
        feat_flatten = feat.view(batch_size, feature_dim, H * W).permute(0, 2, 1)

        lstm_h = feat.new_zeros(self.lstm_num, 2, batch_size, self.in_channels)
        outputs = feat.new_zeros((batch_size, self.max_seq_len + 1, self.dictionary.num_classes),
                             dtype=torch.float32)

        end = feat.new_ones((batch_size,), dtype=torch.uint8)
        pre_seq = feat.new_full((batch_size,),
                         self.dictionary.start_idx,
                         dtype=torch.long)
        for t in range(self.max_seq_len + 1):
            if t == 0:
                feat_t = feat_holistic
            else:
                feat_t = self.de_embedding(pre_seq)

            # lstms
            for i in range(self.lstm_num):
                inp = feat_t if i == 0 else lstm_h[i - 1, 0]
                lstm_h[i, 0], lstm_h[i, 1] = self.de_lstms[i](inp, (lstm_h[i, 0], lstm_h[i, 1]))
            lstm_ht = lstm_h[-1, 0]
            if t == 0:
                continue

            # attention
            out_t, _ = self.de_attn(lstm_ht, feat_flatten, feat_flatten)

            # character classification
            out_t = torch.cat((out_t, lstm_ht), dim=1)
            out_t = self.de_cls(out_t)

            out_t = self.softmax(out_t)
            outputs[:, t, :] = out_t
            score, pre_seq = torch.max(out_t, dim=1)

            end = end & (pre_seq != self.dictionary.end_idx)
            if torch.sum(end) == 0:
                break

        return outputs[:, 1:, :]

    def _encoder(
            self,
            feat: Optional[torch.Tensor] = None,
    ):
        batch_size, feature_dim, H, W = feat.size()
        feat_flatten = feat.view(batch_size, feature_dim, H * W).permute(0, 2, 1)
        start_tokens = feat.new_full((batch_size,), self.dictionary.start_idx, dtype=torch.long)
        start_emb = self.en_embedding(start_tokens)
        feat_holistic, _ = self.en_attn(start_emb, feat_flatten, feat_flatten)
        return feat_holistic


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.layer_norm(q)

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        att = torch.matmul(q / self.scale, k.permute(0, 1, 3, 2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)

        out = torch.matmul(self.dropout(att), v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, self.hidden_dim)

        out = self.dropout(self.fc_o(out))

        return out, att
