# -*- coding: utf-8 -*-
import torch as th


class Encoder(th.nn.Module):
    def __init__(self, in_ch=4, ngf=64, n_bl=None):
        super().__init__()
        self.enc = [
            th.nn.Sequential(
                th.nn.Conv2d(in_ch, ngf, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(ngf),
            )
        ]
        for ic, oc in [
            (ngf * 1, ngf * 2),
            (ngf * 2, ngf * 4),
            (ngf * 4, ngf * 8),
            (ngf * 8, ngf * 16),
            (ngf * 16, ngf * 16),
            (ngf * 16, ngf * 16),
        ]:
            self.enc += [self._dblock(ic, oc)]
        if n_bl:
            self.enc = self.enc[:n_bl]
        self.enc = th.nn.Sequential(*self.enc)

    def _dblock(self, in_ch, out_ch):
        return th.nn.Sequential(
            th.nn.LeakyReLU(0.2, True),
            th.nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        o = self.enc(x)

        return o


class Decoder(th.nn.Module):
    def __init__(self, out_ch=3, exp=1, ngf=64, n_bl=None):
        super().__init__()
        self.dec = []
        for ic, oc in [
            (ngf * 16, ngf * 16),
            (ngf * 16 * exp, ngf * 16),
            (ngf * 16 * exp, ngf * 8),
            (ngf * 8 * exp, ngf * 4),
            (ngf * 4 * exp, ngf * 2),
            (ngf * 2 * exp, ngf * 1),
        ]:
            self.dec += [self._ublock(ic, oc)]
        self.dec += [
            th.nn.Sequential(
                th.nn.ReLU(True),
                th.nn.ConvTranspose2d(ngf * exp, out_ch, 4, 2, 1),
            )
        ]
        if n_bl:
            self.dec = self.dec[-n_bl:]
        self.dec = th.nn.Sequential(*self.dec)

    def forward(self, x):
        o = self.dec(x)
        o = th.sigmoid(o)

        return o

    def _ublock(self, in_ch, out_ch):
        return th.nn.Sequential(
            th.nn.ReLU(True),
            th.nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(out_ch),
        )


class AE(th.nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_bl=None):
        super().__init__()
        self.enc = Encoder(in_ch=in_ch, ngf=ngf, n_bl=n_bl)
        self.dec = Decoder(out_ch=out_ch, ngf=ngf, n_bl=n_bl)

    def forward(self, x):
        o = self.enc(x)
        o = self.dec(o)

        return o


class VAE(th.nn.Module):
    def __init__(self, in_ch, out_ch, h_dim=1024, z_dim=256):
        super().__init__()
        self.enc = Encoder(in_ch=in_ch, ngf=ngf, n_bl=n_bl)
        self.dec = Decoder(out_ch=out_ch, ngf=ngf, n_bl=n_bl)
        self.mu = th.nn.Conv1d(h_dim, z_dim, 1)
        self.std = th.nn.Conv1d(h_dim, z_dim, 1)
        self.asd = th.nn.Conv1d(z_dim, h_dim, 1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = th.randn(*mu.size())
        z = mu + std * esp

        return z

    def forward(self, x):
        h = self.enc(x)
        mu, std = self.mu(h), self.std(h)
        z = self.reparametrize(mu, std)
        o = self.asd(z)
        __import__("ipdb").set_trace()

        return o


class Unet(th.nn.Module):
    def __init__(self, in_ch=4, out_ch=1, ngf=64, n_bl=None, cat=False):
        super().__init__()
        self.n_bl = n_bl
        self.cat = cat
        self.enc = [
            th.nn.Sequential(
                th.nn.Conv2d(in_ch, ngf, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(ngf),
            )
        ]
        for ic, oc in [
            (ngf * 1, ngf * 2),
            (ngf * 2, ngf * 4),
            (ngf * 4, ngf * 8),
            (ngf * 8, ngf * 8),
            (ngf * 8, ngf * 8),
            (ngf * 8, ngf * 8),
        ]:
            self.enc += [self._dblock(ic, oc)]
        if n_bl:
            self.enc = self.enc[:n_bl]
        self.enc = th.nn.Sequential(*self.enc)

        self.dec = []
        for ic, oc in [
            (ngf * 8 * (2 if self.cat else 1), ngf * 8),
            (ngf * 8 * 2, ngf * 8),
            (ngf * 8 * 2, ngf * 8),
            (ngf * 8 * 2, ngf * 4),
            (ngf * 4 * 2, ngf * 2),
            (ngf * 2 * 2, ngf * 1),
        ]:
            self.dec += [self._ublock(ic, oc)]
        self.dec += [
            th.nn.Sequential(
                th.nn.ReLU(True),
                th.nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1),
            )
        ]
        if n_bl:
            self.dec = self.dec[-n_bl:]
            ic, oc = self.dec[0][1].weight.shape[:2]
            self.dec[0] = self._ublock(ic // 2, oc)
        self.dec = th.nn.Sequential(*self.dec)

    def _dblock(self, in_ch, out_ch):
        return th.nn.Sequential(
            th.nn.LeakyReLU(0.2, True),
            th.nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(out_ch),
        )

    def _ublock(self, in_ch, out_ch):
        return th.nn.Sequential(
            th.nn.ReLU(True),
            th.nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(out_ch),
        )

    def forward(self, x, asd=None):
        o = x
        e = []
        for b in self.enc:
            o = b(o)
            e.append(o)
        e = e[:-1][::-1]
        if asd is not None:
            o = th.cat((o, asd), 1)
        o = self.dec[0](o)
        for b, s in zip(self.dec[1:], e):
            o = b(th.cat((s, o), dim=1))

        o = th.sigmoid(o)

        return o
