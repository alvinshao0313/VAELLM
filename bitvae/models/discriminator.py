import random
import torch.nn as nn

from bitvae.modules.normalization import Normalize

class DiscriminatorPool:
    def __init__(self, pool_size):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.images.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.pool_size - 1)
                    tmp = self.images[i].clone()
                    self.images[i] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.stack(return_images)

class ImageDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.discriminator = NLayerDiscriminator(ndf=args.base_ch_disc) # by default using PatchGAN
        self.disc_pool = args.disc_pool # be default "no"
        if args.disc_pool == "yes":
            self.real_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
            self.fake_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
    
    def forward(self, x, pool_name=None):
        if pool_name and self.disc_pool == "yes":
            assert pool_name in ["real", "fake"]
            if pool_name == "real":
                x = self.real_pool.query(x)
            elif pool_name == "fake":
                x = self.fake_pool.query(x)
        # by default without pool
        return self.discriminator(x)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_type = "batch"
        use_bias = norm_type != "batch"

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                Normalize(ndf * nf_mult, norm_type=norm_type),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            Normalize(ndf * nf_mult, norm_type=norm_type),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)