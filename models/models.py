import numpy as np
import torch
import torch.nn as nn

from modules.attention import ClsAttention, Attention
from modules.base_cmn import BaseCMN
from modules.gcn import GCN
from modules.ram import RAM
from modules.visual_extractor import VisualExtractor
from torch.nn.parameter import Parameter
from modules.contrastive_loss import SupConLoss

class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer, num_classes, fw_adj, bw_adj, feat_size=2048, embed_size=256, hidden_size=512): #, num_classes, fw_adj, bw_adj, feat_size=1024, embed_size=256, hidden_size=512
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.fw_adj = fw_adj
        self.bw_adj = bw_adj
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer,)
        self.cls_atten = ClsAttention(feat_size, num_classes)
        # criterionContrastive = contrastive_loss(batch_size=6)
        # self.criterionContrastive = criterionContrastive
        self.gcn = GCN(feat_size, feat_size // 4)
        self.atten = Attention(hidden_size, feat_size)
        self.ram = RAM(reduction=2)

        self.ziji = torch.nn.Linear(62, 98) # (62,98)

        fw_D = torch.diag_embed(fw_adj.sum(dim=1))
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))
        inv_sqrt_fw_D = fw_D.pow(-0.5)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0

        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_fw_D)
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_bw_D)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        fw_A = self.fw_A.repeat(8, 1, 1)
        bw_A = self.bw_A.repeat(8, 1, 1)
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)

        att_feats_0 = self.ram(att_feats_0)
        att_feats_1 = self.ram(att_feats_1)

        att_feats_0 = self.cls_atten(att_feats_0)
        att_feats_1 = self.cls_atten(att_feats_1)
        # att_feats_0 = self.gcn(att_feats_0, fw_A, bw_A)
        # att_feats_1 = self.gcn(att_feats_1, fw_A, bw_A)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        att_feats = self.ziji(att_feats.transpose(1, 2)).transpose(1, 2)
        # att_feats = np.pad(att_feats,pad_width=((0.0),(18,18),(0.0)))
        # 空间注意机制初始化
        # fc_feats = self.gcn(fc_feats, fw_A, bw_A)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            # c_loss = self.criterionContrastive(fc_feats_0, fc_feats_1)
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
