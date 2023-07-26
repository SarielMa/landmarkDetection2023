import torch.nn as nn
import torch
from utils.ShapeSelfAttention import ShapeSelfAttentionLayer
from utils.ImageSelfAttention import ImageSelfAttention
from utils.ShapeToImageAttention import ShapeToImageAttentionLayer
from utils.ShapeEmbedding import ObjectShapeEmbedding
from utils.SpatialContrastNorm import SpatialContrastNorm
#%%
class CNNBlock(nn.Module):
    def __init__(self, C_in, C_out, H_in, W_in, kernel_size, stride, padding):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        hid_dim = 2 * max([C_in, C_out])
        self.norm = SpatialContrastNorm(kernel_size=(H_in//8, W_in//8))
        self.conv = nn.Sequential(nn.Conv2d(C_in, hid_dim, kernel_size, stride, padding),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, hid_dim, kernel_size=1, stride=1, padding=0),
                                  nn.GroupNorm(1, hid_dim, affine=False),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(hid_dim, C_out, kernel_size=3, stride=1, padding=1))
        self.res_path = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
        
    def forward(self, x):
        #x.shape (B, C, H, W), C=in_channels        
        x = self.norm(x)
        y = self.res_path(x) + self.conv(x) #(B, out_channels, H_new, W_new)
        return y
#%%
class ImageFeatureLayer(nn.Module):    
    def __init__(self, ):
        super().__init__()
        self.layer0 = CNNBlock(C_in=3,  C_out=32, H_in=512, W_in=512, kernel_size=5, stride=1, padding=2)
        self.layer1 = CNNBlock(C_in=32, C_out=128, H_in=128, W_in=128, kernel_size=5, stride=4, padding=2)
    def forward(self, x):
        #x.shape (B,1,512,512)
        x0 = self.layer0(x)  #x0.shape (B,32,512,512)
        x1 = self.layer1(x0) #x1.shape (B,128,128,128)
        return x0, x1
#%%
class ShapeDeformer(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg=arg
        n_layers = arg['n_layers']
        n_heads = arg['n_heads']
        embed_dim = arg['embed_dim']
        alpha = arg['alpha']
        attn_drop = arg['attn_drop']
        patch_size = arg['patch_size']
        C_in = arg['C_in']
        W_in = arg['W_in']
        H_in = arg['H_in']
        
        self.image_feature = ImageFeatureLayer()
        
        self.shape_embedding = ObjectShapeEmbedding(C=C_in,
                                                    embed_dim=embed_dim,
                                                    pos_dim=2,
                                                    alpha=alpha)
        
        self.shape_self_attn = nn.ModuleList()
        for n in range(0, n_layers+2):
            self.shape_self_attn.append(ShapeSelfAttentionLayer(n_heads=n_heads, 
                                                                embed_dim=embed_dim, 
                                                                pos_dim=2,
                                                                alpha=alpha,
                                                                attn_drop=attn_drop))
        self.image_self_attn = ImageSelfAttention(H_in=H_in, 
                                                  W_in=W_in, 
                                                  C_in=C_in, 
                                                  patch_size=patch_size,
                                                  embed_dim=embed_dim,
                                                  n_heads=n_heads,
                                                  attn_drop=attn_drop,
                                                  n_layers=n_layers, 
                                                  alpha=alpha)
        
        self.shape_to_image_attn = nn.ModuleList()
        for n in range(0, n_layers):
            self.shape_to_image_attn.append(ShapeToImageAttentionLayer(n_heads=n_heads, 
                                                                       embed_dim=embed_dim, 
                                                                       pos_dim=2,
                                                                       alpha=alpha,
                                                                       attn_drop=attn_drop))                
                       
        self.out = nn.Sequential(nn.LayerNorm(embed_dim),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.Softplus(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.Softplus(),
                                 nn.Linear(embed_dim, 2))
        
    def forward(self, obj_init, template, img):
        #obj_init: (B, N, 2) in the 512x512 image space
        #template: (B, N, 2) in the 512x512 image space
        #img: (B,1,512,512) image
        #-----------------------------
        feature512, feature128 = self.image_feature(img)
        if self.arg['W_in'] == 512:
            feature = feature512
        elif self.arg['W_in'] == 128:
            feature = feature128
        #-----------------
        template = -1 + template / (511/2) #range: -1 to 1
        #-----------------        
        x0 = self.shape_embedding(obj_init, template, feature, self.arg['beta'])#(B,N,E)
        #-----------------        
        xp = -1 + obj_init / (511/2) #range: -1 to 1
        #-----------------
        x1 = x0
        if self.arg['beta'] != 0:
            x1, w1 = self.shape_self_attn[self.arg['n_layers'] + 1](x1, template)
            x1, w1 = self.shape_self_attn[self.arg['n_layers']](x1, template) # difference from ShapeDeformNet4A
        #-----------------        
        y_list, w_y_list, yp = self.image_self_attn(feature)
        #-----------------
        x2 = x1
        for n in range(self.arg['n_layers']-1, 0, -1):
            x2, w2 = self.shape_to_image_attn[n](x2, xp, y_list[n], yp)
            x2, w2 = self.shape_self_attn[n](x2, template) #use template, not xp
            # x2 (B,N,E)
        #------------------ difference from ShapeDeformNet4A and ShapeDeformNet4A1 and  ShapeDeformNet4A1 
        u_pred = self.out(x2) #(B,N,2)      
        #------------------
        obj_pred = obj_init+u_pred
        return obj_pred
#%%    
if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    arg = {}
    arg['n_layers'] = 4
    arg['n_heads'] = 16
    arg['embed_dim'] = 512
    arg['alpha'] = 100
    arg['attn_drop'] = 0
    arg['patch_size'] = [16,16]
    arg['C_in'] = 32
    arg['H_in'] = 512
    arg['W_in'] = 512
    arg['beta'] = 1
    
    B = 4
    N = 38
    device = torch.device('cuda:0')
    obj = torch.rand(B, N, 2).to(device)
    template = torch.rand(B, N, 2).to(device)
    img = torch.rand(B, 3, 512, 512).to(device)

    model = ShapeDeformer(arg).to(device)
    pred = model(obj, template, img)          