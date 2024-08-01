
import copy
import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from abc import ABC, abstractmethod
from blocks import MBConv
from timm.models.registry import register_model
from MFCA_module import MFCA



def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class CTRL_F(nn.Module , ABC):
    """
    Args:
        conv_settings (dictionary(list)): Number of filters and Number of blocks for each convolution stage
        mfca_features_idx (list(int)): Feature maps indicies that is fed to the MFCA module. Default: [1,3] (zero-based)
        image_size (list(int)): Input resolution (H,W). Default: [224,224]
        patch_size_large (int): Patch size for the large/early feature map. Default: 8
        patch_size_small (int): Patch size for the small/late feature map. Default: 2
        small_embedding_dim (int): Number of linear projection output dimension for the small patches.
        large_embedding_dim (int): Number of linear projection output dimension for the large patches.
        depth (list(int)): Number of transformer blocks for each branch in the MFCA module. Default: [3,3]
        num_heads: (list(int)): Number of attention heads in different layers. Default: [6,6]
        head_dim: (list(int)): Dimension for each head for the small and large branch respectively. Default: [16,64]
        mlp_ratio: (list(int)): Ratio of mlp hidden dim to embedding dim for the small and large branch respectively. Default: [12,4]
        cross_attn_depth (int): Number of cross-attention layers. Default: 3
    """
    def __init__(self, conv_settings, mfca_features_idx, image_size, patch_size_large,
                 patch_size_small,small_embedding_dim,large_embedding_dim , depth, num_heads, head_dim, mlp_ratio, cross_attn_depth,
                 num_classes=None,**kwargs):
        super().__init__()

        conv_filters = conv_settings['num_filters'][:]
        conv_num_blocks = conv_settings['num_blocks'][:]

        self.stem = conv_nxn_bn(3, conv_filters[0], stride=1)
        self.mv2_lst = nn.ModuleList([])

        self.mv2_lst.append(self._create_block(conv_filters[0],conv_filters[1],conv_num_blocks[0],kernel_size=3))
        self.mv2_lst.append(self._create_block(conv_filters[1],conv_filters[2],conv_num_blocks[1],kernel_size=3))
        self.mv2_lst.append(self._create_block(conv_filters[2],conv_filters[3],conv_num_blocks[2],kernel_size=3))
        self.mv2_lst.append(self._create_block(conv_filters[3],conv_filters[4],conv_num_blocks[3],kernel_size=3))

        img_h , img_w = image_size

        self.feat1_idx , self.feat2_idx = mfca_features_idx[0] , mfca_features_idx[1]

        FM1_size= (img_h // (2 ** (self.feat2_idx + 1)) , img_w // (2 ** (self.feat2_idx + 1))) #small feature map size
        FM2_size= (img_h // (2 ** (self.feat1_idx + 1)) , img_w // (2 ** (self.feat1_idx + 1))) #large feature map size

        assert FM1_size[0] % patch_size_small ==0 and FM1_size[1] % patch_size_small == 0, 'Feature map dimensions must be divisible by the patch size.'
        assert FM2_size[0] % patch_size_large ==0 and FM2_size[1] % patch_size_large == 0, 'Feature map dimensions must be divisible by the patch size.'

        num_patches_small = (FM1_size[0]//patch_size_small) * (FM1_size[1]//patch_size_small)
        num_patches_large = (FM2_size[0]//patch_size_large) * (FM2_size[1]//patch_size_large)

        patch_small_dim = conv_filters[4] * (patch_size_small ** 2)
        patch_large_dim = conv_filters[2] * (patch_size_large ** 2)

        self.rearrange_small = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_small, p2 = patch_size_small),

        )
        self.rearrange_large = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_large, p2 = patch_size_large),

        )
        self.to_patch_embedding_small_feature = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_small, p2 = patch_size_small),
            nn.Linear(patch_small_dim, small_embedding_dim),
        )

        self.to_patch_embedding_large_feature = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_large, p2 = patch_size_large),
            nn.Linear(patch_large_dim, large_embedding_dim),
        )

        self.pos_embedding_small = nn.Parameter(torch.randn(1,num_patches_small+1,small_embedding_dim))
        self.pos_embedding_large = nn.Parameter(torch.randn(1,num_patches_large+1,large_embedding_dim))

        self.cls_token_small = nn.Parameter(torch.randn(1,1,small_embedding_dim))
        self.cls_token_large = nn.Parameter(torch.randn(1,1,large_embedding_dim))


        self.cross_transformer = MFCA(small_embedding_dim,depth[0],num_heads[0],head_dim[0],mlp_ratio[0],large_embedding_dim,depth[1],
                                      num_heads[1],head_dim[1],mlp_ratio[1],cross_attn_depth) 

        self.global_average_pooling=nn.AdaptiveAvgPool2d((1))



    def _create_block(self,inp,oup,depth,expand_ratio=4,kernel_size=3):
        layers=nn.ModuleList([])
        for i in range(depth):
            if i==0:
                layers.append(MBConv(inp,oup,kernel_size,2,expand_ratio))
            else:
                layers.append(MBConv(oup,oup,kernel_size,1,expand_ratio))

        return nn.Sequential(*layers)

    @abstractmethod
    def forward(self, *args):
        pass



class CTRL_F_AKF(CTRL_F):
    def __init__(self, conv_settings, mfca_features_idx, image_size, patch_size_large,
                 patch_size_small,small_embedding_dim,large_embedding_dim , depth, num_heads, head_dim, mlp_ratio, cross_attn_depth,
                 num_classes=None, scaling_factor=None, **kwargs):

        super().__init__(conv_settings, mfca_features_idx, image_size, patch_size_large,
                 patch_size_small,small_embedding_dim,large_embedding_dim , depth, num_heads, head_dim, mlp_ratio, cross_attn_depth,
                        num_classes, **kwargs)



        self.scaling_factor=scaling_factor

        # heads for small and large branches in MFCA module
        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_embedding_dim),
            nn.Linear(small_embedding_dim, num_classes)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_embedding_dim),
            nn.Linear(large_embedding_dim, num_classes)
        )

        # Fully connected layers for convolution path
        self.fc1=nn.Linear(conv_settings['num_filters'][4], 128)
        self.fc2=nn.Linear(128, num_classes)

    def forward(self, x,alpha):
        batch_size,_,_,_ = x.shape

        x = self.stem(x)
        MFCA_features = {}
        for i, layer in enumerate(self.mv2_lst):
            x = layer(x)
            if i in {self.feat1_idx, self.feat2_idx}:
                MFCA_features[i] = x

        FM1 = MFCA_features.get(self.feat1_idx)
        FM2 = MFCA_features.get(self.feat2_idx)

        x_cnn=self.global_average_pooling(x) #batch_sizexdimx1x1

        x_cnn=torch.flatten(x_cnn,1) #batch_sizexdim (can use squeeze instead)

        x_cnn=self.fc1(x_cnn)
        x_cnn=self.fc2(x_cnn)


        xl_size = self.rearrange_large(FM1)
        xs_size = self.rearrange_small(FM2)

        xl_tokens = self.to_patch_embedding_large_feature(FM1)
        xs_tokens = self.to_patch_embedding_small_feature(FM2)


        cls_token_small=repeat(self.cls_token_small,'() n d -> b n d',b=batch_size)
        xs_tokens=torch.cat((cls_token_small,xs_tokens),dim=1)

        _,n,_=xs_tokens.shape
        xs_tokens+=self.pos_embedding_small[:,:(n)]

        cls_token_large=repeat(self.cls_token_large,'() n d -> b n d',b=batch_size)
        xl_tokens=torch.cat((cls_token_large,xl_tokens),dim=1)
        xl_tokens+=self.pos_embedding_large

        xs,xl = self.cross_transformer(xs_tokens,xl_tokens)

        xs_cls =  xs[:, 0]
        xl_cls =  xl[:, 0]

        xs_cls = self.mlp_head_small(xs_cls)
        xl_cls = self.mlp_head_large(xl_cls)
        x_trans = xs_cls + xl_cls

        x_cnn = nn.functional.normalize(x_cnn,p=1,dim=1) * self.scaling_factor
        x_trans = nn.functional.normalize(x_trans,p=1,dim=1) * self.scaling_factor

        cnn_head_prob = 1-alpha
        vit_head_prob = alpha

        x_cls_combined = (cnn_head_prob * x_cnn) + (vit_head_prob * x_trans) 

        return x_cls_combined


class CTRL_F_CKF(CTRL_F):
    def __init__(self, conv_settings,mfca_features_idx , image_size, patch_size_large,
                 patch_size_small,small_embedding_dim,large_embedding_dim , depth, num_heads, head_dim, mlp_ratio, cross_attn_depth,
                 num_classes=None, drop=0.0, **kwargs):

        super().__init__(conv_settings, mfca_features_idx, image_size, patch_size_large,
                 patch_size_small,small_embedding_dim,large_embedding_dim , depth, num_heads, head_dim, mlp_ratio, cross_attn_depth,
                         num_classes, **kwargs)


        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_embedding_dim),
            nn.Linear(small_embedding_dim, 128)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_embedding_dim),
            nn.Linear(large_embedding_dim, 128)
        )

        self.fc=nn.Linear(conv_settings['num_filters'][4],128)
        self.out=nn.Linear(128 * 2,num_classes)
        self.dropout=nn.Dropout(drop)

    def forward(self, x):

        batch_size,_,_,_=x.shape

        x = self.stem(x)
        mfca_features = {}
        for i, layer in enumerate(self.mv2_lst):
            x = layer(x)
            if i in {self.feat1_idx, self.feat2_idx}:
                mfca_features[i] = x

        FM1 = mfca_features.get(self.feat1_idx)
        FM2 = mfca_features.get(self.feat2_idx)

        x_cnn=self.global_average_pooling(x)
        x_cnn=torch.flatten(x_cnn,1)
        x_cnn=self.fc(x_cnn)


        xl_size = self.rearrange_large(FM1)
        xs_size = self.rearrange_small(FM2)

        xl_tokens = self.to_patch_embedding_large_feature(FM1)
        xs_tokens = self.to_patch_embedding_small_feature(FM2)


        cls_token_small=repeat(self.cls_token_small,'() n d -> b n d',b=batch_size)
        xs_tokens=torch.cat((cls_token_small,xs_tokens),dim=1)

        _,n,_=xs_tokens.shape
        xs_tokens+=self.pos_embedding_small

        cls_token_large=repeat(self.cls_token_large,'() n d -> b n d',b=batch_size)
        xl_tokens=torch.cat((cls_token_large,xl_tokens),dim=1)
        xl_tokens+=self.pos_embedding_large

        xs,xl = self.cross_transformer(xs_tokens,xl_tokens)


        xs_cls =  xs[:, 0]
        xl_cls =  xl[:, 0]

        xs_cls = self.mlp_head_small(xs_cls)
        xl_cls = self.mlp_head_large(xl_cls)

        x_trans = xl_cls + xs_cls

        x_cls_combined=torch.cat((x_cnn,x_trans),-1)

        x_cls_combined=self.dropout(x_cls_combined)

        x_cls_combined=self.out(x_cls_combined)

        return x_cls_combined



@register_model
def CTRLF_S_AKF(pretrained=False, **kwargs):

    conv_settings = {'num_filters': [16,32,64,128,256]
                     , 'num_blocks': [2,2,3,5]}

    model = CTRL_F_AKF(conv_settings, mfca_features_idx=[1,3], image_size=[224,224],patch_size_large=8,patch_size_small=2,
                   small_embedding_dim=128,large_embedding_dim=256,
                   depth=[3,3], num_heads=[6,6], head_dim=[16,64], mlp_ratio=[12,4], cross_attn_depth=2, **kwargs
                   )
    return model


@register_model
def CTRLF_S_CKF(pretrained=False, **kwargs):

    conv_settings = {'num_filters': [16,32,64,128,256]
                     , 'num_blocks': [2,2,3,5]}

    model = CTRL_F_CKF(conv_settings, mfca_features_idx=[1,3], image_size=[224,224],patch_size_large=8,patch_size_small=2,
                   small_embedding_dim=128,large_embedding_dim=256,
                   depth=[3,3], num_heads=[6,6], head_dim=[16,64], mlp_ratio=[12,4], cross_attn_depth=2, **kwargs
                   )
    return model

@register_model
def CTRLF_B_AKF(pretrained=False, **kwargs):

    conv_settings = {'num_filters': [16,64,92,196,256]
                     , 'num_blocks': [2,2,4,8]}

    model = CTRL_F_AKF(conv_settings, mfca_features_idx=[1,3], image_size=[224,224],patch_size_large=8,patch_size_small=2,
                   small_embedding_dim=192,large_embedding_dim=384,
                   depth=[3,3], num_heads=[6,6], head_dim=[16,64], mlp_ratio=[12,4], cross_attn_depth=4, **kwargs
                   )
    return model


@register_model
def CTRLF_B_CKF(pretrained=False, **kwargs):

    conv_settings = {'num_filters': [16,64,92,196,256]
                     , 'num_blocks': [2,2,4,8]}

    model = CTRL_F_CKF(conv_settings, mfca_features_idx=[1,3], image_size=[224,224],patch_size_large=8,patch_size_small=2,
                   small_embedding_dim=192,large_embedding_dim=384,
                   depth=[3,3], num_heads=[6,6], head_dim=[16,64], mlp_ratio=[12,4], cross_attn_depth=4, **kwargs
                   )
    return model
