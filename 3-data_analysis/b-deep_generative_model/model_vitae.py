from email.iterators import walk
import torch
from torch import nn
from math import gamma
import torch
from torch import device
from torch import diag
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTAE_Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim_latent, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        self.head_to_latent = nn.Sequential(
            Rearrange('b ... -> b (...)'),
            nn.LayerNorm(num_patches*dim),
            nn.Linear(num_patches*dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim_latent),
        )
        # self.weight_init()

    def forward(self, img):

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = self.to_latent(x)
        x = self.head_to_latent(x)

        return x

    def weight_init(self):
        torch.nn.init.kaiming_normal_(self.to_patch_embedding[1].weight)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        torch.nn.init.kaiming_normal_(self.head_to_latent[2].weight)
        torch.nn.init.kaiming_normal_(self.head_to_latent[4].weight)

        return

class ViTAE_Decoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim_latent, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width

        self.latent_to_head = nn.Sequential(
            nn.Linear(dim_latent, dim),
            nn.GELU(),
            nn.Linear(dim, num_patches*dim),
            nn.LayerNorm(num_patches*dim),
            Rearrange('b (d1 d2) -> b d1 d2', d1 = num_patches, d2 = dim),
            )

        self.get_patch_embedding_dim = nn.Sequential(
            Rearrange('b (h w) d -> b h w d', h = image_height // patch_height, w = image_width // patch_width),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2) -> b (h p1) (w p2)', h = image_height // patch_height, w = image_width // patch_width, p1 = patch_height, p2 = patch_width),
            nn.Sigmoid(),
        )
        # self.weight_init()

    def forward(self, x):
        x = self.latent_to_head(x)

        x_temp = self.get_patch_embedding_dim(x.clone())
        pe = posemb_sincos_2d(x_temp)
        x = x + pe

        x = self.transformer(x)
        x = self.to_pixels(x)

        return x

    def weight_init(self):
        torch.nn.init.kaiming_normal_(self.to_pixels[0].weight)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        torch.nn.init.kaiming_normal_(self.latent_to_head[0].weight)
        torch.nn.init.kaiming_normal_(self.latent_to_head[2].weight)

        return

class ViTAE_RGB2Depth(nn.Module):
    """ViTAE RGB2Depth."""

    def __init__(self, image_size = None,
                        beta = 1.0,
                        orth_factor = 100.0,
                        latent_unit_range = 3.0,
                        patch_size = 32,
                        dim_latent = 16,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048, 
                        device=None):
        super(ViTAE_RGB2Depth, self).__init__()

        self.image_size = image_size
        self.beta = beta
        self.orth_factor = orth_factor
        self.patch_size = patch_size
        self.dim_latent = dim_latent
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.device = device,
        self.latent_unit_range = latent_unit_range
        
        self.encoder = ViTAE_Encoder(
            image_size = self.image_size,
            patch_size = self.patch_size,
            dim_latent = self.dim_latent*2,
            dim = self.dim,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim
        )

        self.decoder = ViTAE_Decoder(
            image_size = self.image_size,
            patch_size = self.patch_size,
            dim_latent = self.dim_latent,
            dim = self.dim,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim
        )

        # self.weight_init()

    def forward(self, x):

        self.distributions = self.encoder(x)
        mu = (torch.sigmoid(self.distributions[:, :self.dim_latent])*2.0 - 1.0)*self.latent_unit_range
        logvar = self.distributions[:, self.dim_latent:]
        self.z = self.reparametrize(mu, logvar)
        depth_recon = self.decoder(self.z)

        return depth_recon, mu, logvar

    def reconstruction_loss(self, depth_img, depth_recon):
        batch_size = depth_img.size(0)
        assert batch_size != 0

        recon_loss = F.mse_loss(depth_recon, depth_img, reduction='sum').div(batch_size)

        return recon_loss

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds_var = 0.5*(logvar.exp() - logvar - 1 + mu.pow(2))

        # corr between units
        triu_corr_mu = torch.triu(torch.corrcoef(mu.T) + 1.0, diagonal=1) 
        # triu_corr_mu = torch.triu(torch.pow(torch.corrcoef(mu.T), exponent=2.0), diagonal=1) 
        # triu_corr_mu = torch.triu(torch.corrcoef(mu.T).abs(), diagonal=1)
        corr_triu_loss = torch.flatten(triu_corr_mu).sum(0, True) * self.orth_factor 

        # klds = 0.5*(logvar.exp() - logvar - 1 + mu.pow(2))
        # klds = 0.5*(logvar.exp() - logvar - 1 + torch.nn.functional.relu(mu - 1.0).pow(2))
        
        klds_var = klds_var.sum(1).sum(0, True)

        return klds_var, corr_triu_loss

    def calc_monotonic_annealing_loss(self, rgb_images, depth_images, capacity_num_iter, capacity_stop_iter):

        self.depth_recon, self.mu, self.logvar = self.forward(rgb_images)
        self.recon_loss = self.reconstruction_loss(depth_images, torch.squeeze(self.depth_recon))
        self.klds_var, self.corr_triu_loss = self.kl_divergence(self.mu, self.logvar)

        self.total_kld = self.klds_var + self.corr_triu_loss

        monotonic_coef = capacity_stop_iter/(capacity_num_iter/2.0)
        if(monotonic_coef > 1.0): monotonic_coef = 1.0

        self.kld_loss = self.beta * monotonic_coef * self.total_kld
        loss = self.recon_loss + self.kld_loss

        return loss


    def calc_cyclical_annealing_loss(self, rgb_images, depth_images, num_cycles, capacity_num_iter, capacity_stop_iter):

        self.depth_recon, self.mu, self.logvar = self.forward(rgb_images)
        self.recon_loss = self.reconstruction_loss(depth_images, torch.squeeze(self.depth_recon))
        self.total_kld = self.kl_divergence(self.mu, self.logvar)

        single_cycle = capacity_num_iter/num_cycles
        i_stop_cycle = capacity_stop_iter%single_cycle

        cycle_coef = i_stop_cycle/(single_cycle/2.0)
        if(cycle_coef > 1.0): cycle_coef = 1.0

        self.kld_loss = self.beta * cycle_coef * self.total_kld
        loss = self.recon_loss + self.kld_loss

        return loss


    def calc_beta_vae_basic_loss(self, rgb_images, depth_images):

        self.depth_recon, self.mu, self.logvar = self.forward(rgb_images)
        self.recon_loss = self.reconstruction_loss(depth_images, torch.squeeze(self.depth_recon))
        self.total_kld = self.kl_divergence(self.mu, self.logvar)

        # loss = self.recon_loss + self.beta * self.total_kld
        self.kld_loss = self.beta * self.total_kld
        loss = self.recon_loss + self.kld_loss

        return loss

    def calc_beta_vae_capacity_control_loss(self, rgb_images, depth_images, capacity_max, capacity_num_iter, capacity_stop_iter):

        self.depth_recon, self.mu, self.logvar = self.forward(rgb_images)
        self.recon_loss = self.reconstruction_loss(depth_images, torch.squeeze(self.depth_recon))
        self.total_kld = self.kl_divergence(self.mu, self.logvar)

        capacity = torch.clamp(torch.as_tensor(capacity_stop_iter/capacity_num_iter * capacity_max, device=self.device),
                                 0.0, capacity_max)
        # loss = self.recon_loss + self.beta * torch.nn.functional.relu(self.total_kld - capacity)
        self.kld_loss = self.beta * torch.nn.functional.relu(self.total_kld - capacity)
        loss = self.recon_loss + self.kld_loss

        return loss

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Transformer)):
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.Transformer)):
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)


if __name__ == '__main__':
    pass
