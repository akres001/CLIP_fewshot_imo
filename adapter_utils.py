import torch
from torch import nn
import math
from torchvision import transforms as T
import torch.nn.functional as F
import copy
    
class CLIP_Adapter(nn.Module):
    def __init__(self, clip_model, **kwargs):
        super().__init__()
        
        self.image_encoder = copy.deepcopy(clip_model.visual)        
        self.hid_dim = 512 if kwargs['backbone'] == 'ViT' else 1024

        out_class = kwargs['out_class']
        self.predictor = nn.Linear(self.hid_dim, out_class)

    def initialize_parameters(self):
        nn.init.normal_(self.predictor.weight, std=self.hid_dim** -0.5)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def encode_image(self, image, apply_adapter=False):
        return self.image_encoder(image.type(self.dtype), apply_adapter=apply_adapter)
    
    def forward(self, x):
        representation = self.encode_image(x, apply_adapter=True)
        out = self.predictor(representation)
        return out
    

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.down_proj.apply(self.init_bert_weights)
            self.up_proj.apply(self.init_bert_weights)
            # if self.use_gating:
                # self.gate.apply(self.init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    
    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
