import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import timm
from timm.models.layers import trunc_normal_, DropPath
from .vit import VisionTransformer,  VisionTransformer2, VisionTransformerClip
import numpy as np
import copy
from .hypermodel.mlp import MLP as Mlp
from .hypermodel.module_wrappers import CLHyperNetInterface
from .hypermodel.hyper_utils import build_hyper_mlps, flatten_dictionary, assign_weights, collect_target_shapes
from .hypermodel.chunked_hyper_model import ChunkedHyperNetworkHandler
from .flow import build_flow

class MainAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self._param_shapes = {}
        self.scale = qk_scale or head_dim ** -0.5
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = Mlp(n_in=dim, n_out=dim*3, hidden_layers=[], use_bias=qkv_bias, activation_fn=None, no_weights=True, dropout_rate=-1)
        self._param_shapes['qkv'] = self.qkv.param_shapes
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = Mlp(n_in=dim, n_out=dim, hidden_layers=[], activation_fn=None, no_weights=True, dropout_rate=-1)
        self._param_shapes['proj'] = self.proj.param_shapes
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    @property
    def param_shapes(self):
        return self._param_shapes
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, weights, register_hook=False, prompt=None):
        B, N, C = x.shape
        qkv_weights = weights['qkv']
        proj_weights = weights['proj']
        qkv = self.qkv(x, weights=qkv_weights).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, weights=proj_weights)
        x = self.proj_drop(x)
        return x
    
class MainBlock(nn.Module):

    def __init__(self, dim, num_heads, num_classes, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self._param_shapes = {}
        # self.norm1 = norm_layer(dim)
        self.norm1 = ("layernorm", dim)
        self.attn = MainAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self._param_shapes['attn'] = self.attn.param_shapes
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = ("layernorm", dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(n_in=dim, n_out=dim, hidden_layers=[mlp_hidden_dim], no_weights=True, dropout_rate=-1 if drop==0. else drop)
        self._param_shapes['mlp'] = self.mlp.param_shapes
        # classifier
        # self.last = nn.Linear(768, num_classes) 
        self.last = Mlp(n_in=768, n_out=num_classes, hidden_layers=[], activation_fn=None, no_weights=True, dropout_rate=-1)
        
        self._param_shapes['last'] = self.last.param_shapes
        self._externalnorm_layers = {'main': [self.norm1, self.norm2]}
    @property
    def param_shapes(self):
        return self._param_shapes
    
    @property
    def externalnorms(self):
        return self._externalnorm_layers

    def forward(self, x, weights, extnorms, register_hook=False, prompt=None):
        attn_weights = weights['attn']
        norm1, norm2 = extnorms['main'][0], extnorms['main'][1]
        x = x + self.drop_path(self.attn(norm1(x), weights=attn_weights, register_hook=register_hook, prompt=prompt))
        mlp_weights = weights['mlp']
        x = x + self.drop_path(self.mlp(norm2(x), weights=mlp_weights))
        last_weights = weights['last']

        out = x[:,0,:]
        out = out.view(out.size(0), -1)
        out = self.last(out, weights=last_weights)
        return out

class HyperBlock(nn.Module, CLHyperNetInterface):
    def __init__(self, target_shapes, target_norms, num_domains, chunk_dims,
                layers=[50, 100], te_dim=8, activation_fn=torch.nn.ReLU(),
                use_bias=True, no_weights=False, ce_dim=None,
                init_weights=None, dropout_rate=-1, noise_dim=-1,
                temb_std=-1):
        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)
        self.num_domains = num_domains
        self.shape_indices = {}
        # shared domain embeddings
        self._domain_embs = nn.ParameterList()
        for _ in range(num_domains):
            self._domain_embs.append(nn.Parameter(data=torch.Tensor(te_dim//2),
                                                requires_grad=True))
            torch.nn.init.normal_(self._domain_embs[-1], mean=0., std=1.)
        # shared domain norms
        self._domain_norms_name = target_norms
        self._domain_norms = nn.ModuleList()
        for _ in range(num_domains):
            self._domain_norms.append(nn.ModuleDict())
            for name, norms in target_norms.items():
                normlist = []
                for normtype, hidden_dim in norms:
                    if normtype == 'batchnorm':
                        normlist.append(nn.BatchNorm1d(hidden_dim))
                    elif normtype == 'layernorm':
                        normlist.append(nn.LayerNorm(hidden_dim))
                    else:
                        raise ValueError('Unknown normalization type: %s' % normtype)
                self._domain_norms[-1][name] = nn.ModuleList(normlist)
        self.flattened_target_shapes, self.index_mapping = flatten_dictionary(target_shapes)

        self._thetas, self._theta_shapes = [], []
        self.hyper_layers = nn.ModuleDict()
        for layer_name in target_shapes.keys():
            layer_shapes, self.shape_indices[layer_name] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, layer_name)
            self.hyper_layers[layer_name] = ChunkedHyperNetworkHandler(layer_shapes, num_domains, no_te_embs=True, chunk_dim=chunk_dims[layer_name], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std, verbose=True)
            self._thetas.extend(self.hyper_layers[layer_name].theta)
            self._theta_shapes.extend(self.hyper_layers[layer_name].theta_shapes)

    def create_new_domain(self):
        self._domain_embs.append(nn.Parameter(data=torch.Tensor(self.te_dim//2),
                                                requires_grad=True))
        torch.nn.init.normal_(self._domain_embs[-1], mean=0., std=1.)

        self._domain_norms.append(nn.ModuleDict())
        for name, norms in self._domain_norms_name.items():
            normlist = []
            for normtype, hidden_dim in norms:
                if normtype == 'batchnorm':
                    normlist.append(nn.BatchNorm1d(hidden_dim))
                elif normtype == 'layernorm':
                    normlist.append(nn.LayerNorm(hidden_dim))
                else:
                    raise ValueError('Unknown normalization type: %s' % normtype)
            self._domain_norms[-1][name] = nn.ModuleList(normlist)

    def get_domain_emb(self, domain_id):
        return self._domain_embs[domain_id]
    
    def get_domain_norm(self, domain_id):
        return self._domain_norms[domain_id]
    
    def get_domain_targets(self, domain_id):
        hnet_mode = self.training
        self.eval()
        ret = None

        def collect_W(sub_dict):
            Ws = []
            for k, v in sub_dict.items():
                if isinstance(v, list):
                    Ws.extend(v)
                elif isinstance(v, dict):
                    Ws.extend(collect_W(v))
            return Ws
        with torch.no_grad():
            W = self.forward(domain_id=domain_id)
            W = collect_W(W)
            ret = [d.detach().clone() for d in W]

        self.train(mode=hnet_mode)

        return ret
    
    @property
    def has_theta(self):
        """Getter for read-only attribute ``has_theta``."""
        return True
    
    @property
    def theta(self):
        return self._thetas

    @property
    def theta_shapes(self):
        return self._theta_shapes
    
    def forward(self, domain_id=None, theta=None, dTheta=None, domain_emb=None, ext_inputs=None, squeeze=True):
        weight_dicts = {}
        if domain_emb is None:
            domain_emb = self.get_domain_emb(domain_id)
        last_index = 0
        for layer_name in self.hyper_layers.keys():
            if theta is not None:
                layer_theta = theta[layer_name]
            else:
                layer_theta = None
            if dTheta is not None:
                layer_dTheta = dTheta[last_index:last_index+len(self.hyper_layers[layer_name].theta_shapes)]
            else:
                layer_dTheta = None
            if ext_inputs is not None:
                layer_ext_inputs = ext_inputs[layer_name]
            else:
                layer_ext_inputs = None
            last_index += len(self.hyper_layers[layer_name].theta_shapes)
            weights_flatten = self.hyper_layers[layer_name](domain_id, theta=layer_theta, dTheta=layer_dTheta, task_emb=domain_emb, ext_inputs=layer_ext_inputs)
            for i, w in zip(self.shape_indices[layer_name], weights_flatten):
                weight_dicts[i] = w
        weights = assign_weights(self.index_mapping, weight_dicts)
        return weights
    
class ViThyper(nn.Module):
    def __init__(self, num_classes=10, hyper_param=None):
        super(ViThyper, self).__init__()

        # get last layer
        # self.last = nn.Linear(512, num_classes)
        # self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        zoo_model = VisionTransformer2(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                    num_heads=12, ckpt_layer=0,
                                    drop_path_rate=0
                                    )
        from timm.models import vit_base_patch16_224
        load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del load_dict['head.weight']; del load_dict['head.bias']
        zoo_model.load_state_dict(load_dict, strict=False)

        # timm_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
        # sd = timm_model.state_dict()
        ####################
        # zoo_model = VisionTransformerClip(img_size=224, patch_size=16, embed_dim=768, depth=12,
        #                             num_heads=12, ckpt_layer=0,
        #                             drop_path_rate=0
        #                             )
        # zoo_model.load_from_timm('vit_base_patch16_clip_224.openai', pretrained=True, copy_clip_proj=True)
        ########################
        # load into your implementation (keys match the usual ViT layout)
        # missing, unexpected = zoo_model.load_state_dict(sd, strict=False)
        # print('missing:', missing)
        # print('unexpected:', unexpected)


        # let us try the first hyper network architecture which is to introduce a new attention block
        self.main_block = MainBlock(
                dim=768, num_heads=12, num_classes=num_classes, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0
                )
        self.hyper_block = HyperBlock(
            target_shapes = self.main_block.param_shapes,
            target_norms = self.main_block.externalnorms,
            num_domains = hyper_param[0], #'num_tasks',
            te_dim =  hyper_param[1], #'te_dim',
            ce_dim = hyper_param[2], #'ce_dim',
            chunk_dims = hyper_param[3], #'chunk_dims',
            layers = hyper_param[4], #'layers',
        )
        # add a normalizing flow module here 
        self.detectors = build_flow(dim=768, depth=8, num_domains = hyper_param[0])

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        self.prompt = None
    
    def forward_detector(self, x, train=False):
        self.feat.eval() 
        self.main_block.eval()
        with torch.no_grad():
            # q, _ = self.feat(x)
            ret = self.feat(x)
            cls = ret['cls_proj']
        domain_detector = self.detectors[self.task_id]
        bpd = domain_detector._get_likelihood(cls)
        return bpd

    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):

        # if self.prompt is not None:
        #     with torch.no_grad():
        #         q, _ = self.feat(x)
        #         q = q[:,0,:]
        #     out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
        #     out = out[:,0,:]
        # else:
        #     out, _ = self.feat(x)
        #     out = out[:,0,:]

        self.feat.eval() 
        self.main_block.eval()
        with torch.no_grad():
            # q, _ = self.feat(x)
            ret = self.feat(x)
            q = ret['tokens']
            cls = ret['cls_proj']
            # q = q[:,0,:]
        # add the flow-based detectors here
        if train:
            # domain_detector = self.detectors[self.task_id]
            # bpd = domain_detector._get_likelihood(q[:,0,:])
            domain_id = self.task_id
            pred_targets = self.hyper_block(domain_id=domain_id)
            extnorms = self.hyper_block.get_domain_norm(domain_id)
            out = self.main_block(q, weights=pred_targets, extnorms=extnorms)
            return out#, bpd
        
        else:
            with torch.no_grad():
                scores = []
                outs = []
                for i in range(self.task_id + 1):
                    score = self.detectors[i].predict_ood_score(cls.clone())
                    scores.append(score.unsqueeze(1))
                    pred_targets = self.hyper_block(domain_id=i)
                    extnorms = self.hyper_block.get_domain_norm(i)
                    out = self.main_block(q.clone(), weights=pred_targets, extnorms=extnorms)
                    outs.append(out.unsqueeze(1))
                scores = torch.cat(scores, dim=1)  # (B, num_domains)
                outs = torch.cat(outs, dim=1)      # (B, num_domains, num_classes)
                
                best_scores, best_indices = torch.min(scores, dim=1)  # (B,), (B,)
                out = outs[torch.arange(outs.size(0)), best_indices]  # (B, num_classes)
                return out
            
        # out = out.view(out.size(0), -1)
        # if not pen:
        #     out = self.last(out)
        # if self.prompt is not None and train:
        #     return out, prompt_loss
        # else:
        #     return out
        # return out, bpd
    
def vit_hyper_imnet(out_dim, hyper_param=None):
    num_tasks = hyper_param[0]
    te_dim, ce_dim = hyper_param[1]
    chunk_dims = {
        'attn': 2048 * 8,
        'mlp': 2048 * 8,
        'last': 1280,
    }
    layers = [256, 256]
    return ViThyper(num_classes=out_dim, hyper_param=[num_tasks, te_dim, ce_dim, chunk_dims, layers])