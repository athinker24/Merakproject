"""Sahibin Merak'tan aldığı renkli kabartma görüntüsünü atlasın kabartma çerçevesine taşır; atlas deposuna dokunmaz.

Zincir: atlas kabartma pikseli → atlas tuvali (dunyam.json kabartma sx sy ox oy) → ana harita/4 (atlas_ana kaydı) → ana harita 16k
→ renkli görüntü (kara maskeleri IoU ile kayıt). Renkli görüntüde deniz: mavi bileşen kırmızıdan belirgin büyük.
Kullanım: python3 atlas_kabartma.py RENKLI.png ANA_16K.png [ATLAS_DEPOSU]  → atlas_ana/kabartma_kaynak.jpg, kabartma_kontrol.png
"""
import json, re, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from merak_birlestir import S

Image.MAX_IMAGE_PIXELS = None
F = 4


def main():
    renkli, ana = sys.argv[1], sys.argv[2]; ATLAS = sys.argv[3] if len(sys.argv) > 3 else "/home/user/athinker24/dunyam-atlas-mobil"
    hm = json.load(open(f"{ATLAS}/dunya/dunyam.json"))["heightmap"]; kb = hm["kabartma"]
    not_ = json.load(open("atlas_ana/heightmap_blogu.json"))["_not"]
    sx, sy, ox, oy = [float(v) for v in re.search(r"sx ([\d.]+), sy ([\d.]+)\) \+ \(ox ([-\d.]+), oy ([-\d.]+)\)", not_).groups()]
    # ana harita /4 kara oranı
    A = np.asarray(Image.open(ana)).astype(np.int32) > S
    H, W = A.shape; Hc, Wc = H // F * F, W // F * F
    ak = A[:Hc, :Wc].reshape(Hc // F, F, Wc // F, F).mean((1, 3)).astype(np.float32)
    # renkli görüntü /4
    R = Image.open(renkli).convert("RGB")
    Rw, Rh = R.size
    r4 = np.asarray(R.resize((Rw // F, Rh // F), Image.BOX)).astype(np.float32)
    rk = (r4[..., 2] - r4[..., 0]) < 8                      # kara: mavi kırmızıdan belirgin büyük değil
    L, n = ndimage.label(rk); s = ndimage.sum(rk, L, range(1, n + 1)); rk = np.isin(L, np.nonzero(s >= 30)[0] + 1)
    # kayıt: renkli/4 (y, x) = ana/4 (y, x) · (b, a) + (d, c)
    yy, xx = np.nonzero(ak > 0.5); ry, rx = np.nonzero(rk)
    a0 = (rx.max() - rx.min()) / (xx.max() - xx.min()); b0 = (ry.max() - ry.min()) / (yy.max() - yy.min())
    p0 = [a0, b0, rx.min() - xx.min() * a0, ry.min() - yy.min() * b0]
    y0, y1, x0, x1 = yy.min() - 20, yy.max() + 20, xx.min() - 20, xx.max() + 20
    Y, X = np.mgrid[y0:y1:2, x0:x1:2].astype(np.float32); akk = ak[y0:y1:2, x0:x1:2] > 0.5
    rf = rk.astype(np.float32)

    def m(p):
        a, b, c, d = p
        q = ndimage.map_coordinates(rf, [Y * b + d, X * a + c], order=1, cval=0) > 0.5
        return -(q & akk).sum() / max((q | akk).sum(), 1)
    r = optimize.minimize(m, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
    a, b, c, d = r.x; print("renkli kayıt", np.round(r.x, 4), "IoU", round(-r.fun, 4))
    # kabartma çerçevesi
    ky, kx = Image.open(f"{ATLAS}/{kb['file']}").size[::-1]
    gy, gx = np.mgrid[0:ky, 0:kx].astype(np.float64)
    tr = gy * kb["sy"] + kb["oy"]; tc = gx * kb["sx"] + kb["ox"]
    ay, ax = (tr - oy) / sy, (tc - ox) / sx                    # ana/4
    by, bx = ay * b + d, ax * a + c                            # renkli/4
    olc = max(kb["sx"] / sx * a, 1.0)                          # bir kabartma pikseli kaç renkli/4 pikseli
    rs = ndimage.gaussian_filter(r4, (0.45 * olc, 0.45 * olc, 0)) if olc > 1.2 else r4
    kab = np.stack([ndimage.map_coordinates(rs[..., k], [by, bx], order=1, mode="nearest") for k in range(3)], -1)
    Image.fromarray(np.clip(kab, 0, 255).astype(np.uint8)).save("atlas_ana/kabartma_kaynak.jpg", quality=92)
    # kontrol: kabartmanın karası ile ana haritanın karası
    kk = ndimage.map_coordinates(ak, [ay, ax], order=1, cval=0) > 0.5
    kr = (kab[..., 2] - kab[..., 0]) < 8
    rgb = np.where(kk[..., None], np.array([200, 200, 200]), np.array([20, 30, 60])).astype(np.uint8)
    rgb[kk & ~kr] = [255, 60, 60]; rgb[~kk & kr] = [60, 140, 255]
    Image.fromarray(rgb).save("atlas_ana/kabartma_kontrol.png")
    print("kabartma", kx, ky, "kara uyuşmazlığı", int((kk ^ kr).sum()), "/", int(kk.sum()))


if __name__ == "__main__":
    main()
