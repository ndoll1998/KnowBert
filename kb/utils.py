import torch
import torch.nn as nn
import torch.nn.functional as F
# import transformers
from transformers.modeling_bert import (
    BertEncoder, BertLayer, BertAttention, 
    BertSelfAttention, BertSelfOutput, BertOutput, 
    BertIntermediate, BertLayerNorm
)

def match_shape_2d(ls:list, shape_2d:tuple, fill_value=0):
    # only support two dimensional target tensors
    assert len(shape_2d) == 2
    dim1, dim2 = shape_2d
    # pad all existing elements to match dim2
    tensor = [l[:dim2] + [fill_value] * (dim2 - min(dim2, len(l))) for l in ls[:dim1]]
    tensor = torch.tensor(tensor)
    # pad to match dim1
    return torch.cat((
        tensor, torch.zeros(dim1 - min(dim1, len(ls)), dim2).type(tensor.dtype)
    ), dim=0)

def bert_extended_attention_mask(mask):
    # mask = (batch_size, timesteps)
    # returns an attention_mask useable with BERT
    # see: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L696
    extended_attention_mask = mask.unsqueeze(1).unsqueeze(2).float()
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

def pseudo_inverse(A):
    """ compute pseudo inverse based on QR-Decomposition """
    # see: https://discuss.pytorch.org/t/torch-pinverse-seems-to-be-inaccurate/33616
    rows,cols = A.size()
    if rows >= cols:
        Q,R = torch.qr(A)
        return R.inverse().mm(Q.t())
    else:
        Q,R = torch.qr(A.t())
        return R.inverse().mm(Q.t()).t()

def init_bert_weights(module, initializer_range, extra_modules_without_weights=()):
    # these modules don't have any weights, other then ones in submodules,
    # so don't have to worry about init
    modules_without_weights = (
        BertEncoder, torch.nn.ModuleList, torch.nn.Dropout, BertLayer,
        BertAttention, BertSelfAttention, BertSelfOutput,
        BertOutput, BertIntermediate
    ) + extra_modules_without_weights

    # modified from transformers
    def _do_init(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            m.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(m, BertLayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        elif isinstance(m, modules_without_weights):
            pass
        else:
            raise ValueError(str(m))

        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    for mm in module.modules():
        _do_init(mm)