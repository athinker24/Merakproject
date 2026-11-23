"""Sahibin Merak'tan aldığı renkli kabartma görüntüsünü atlasın kabartma çerçevesine taşır; atlas deposuna dokunmaz.

Zincir: atlas kabartma pikseli → atlas tuvali (dunyam.json kabartma sx sy ox oy) → ana harita/4 (atlas_ana kaydı) → ana harita 16k
→ renkli görüntü (kara maskeleri IoU ile kayıt). Renkli görüntüde deniz: mavi bileşen kırmızıdan belirgin büyük.
Kullanım: python3 atlas_kabartma.py ANA_16K.png RENKLI1.png [RENKLI2.png ...]  (sonraki görüntü öncelikli; kırpılmış görüntüler birleşir)  → atlas_ana/kabartma_kaynak.jpg, kabartma_kontrol.png
"""
import json, re, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from merak_birlestir import S

Image.MAX_IMAGE_PIXELS = None
F = 4


def main():
    ana = sys.argv[1]; renkliler = sys.argv[2:]; ATLAS = "/home/user/athinker24/dunyam-atlas-mobil"
    hm = json.load(open(f"{ATLAS}/dunya/dunyam.json"))["heightmap"]; kb = hm["kabartma"]
    not_ = json.load(open("atlas_ana/heightmap_blogu.json"))["_not"]
    sx, sy, ox, oy = [float(v) for v in re.search(r"sx ([\d.]+), sy ([\d.]+)\) \+ \(ox ([-\d.]+), oy ([-\d.]+)\)", not_).groups()]
    # ana harita /4 kara oranı
    A = np.asarray(Image.open(ana)).astype(np.int32) > S
    H, W = A.shape; Hc, Wc = H // F * F, W // F * F
    ak = A[:Hc, :Wc].reshape(Hc // F, F, Wc // F, F).mean((1, 3)).astype(np.float32)
    ky, kx = Image.open(f"{ATLAS}/{kb['file']}").size[::-1]
    gy, gx = np.mgrid[0:ky, 0:kx].astype(np.float64)
    tr = gy * kb["sy"] + kb["oy"]; tc = gx * kb["sx"] + kb["ox"]
    ay, ax = (tr - oy) / sy, (tc - ox) / sx                    # ana/4
    yy, xx = np.nonzero(ak > 0.5)
    y0, y1, x0, x1 = yy.min() - 20, yy.max() + 20, xx.min() - 20, xx.max() + 20
    Y, X = np.mgrid[y0:y1:2, x0:x1:2].astype(np.float32); akk = ak[y0:y1:2, x0:x1:2] > 0.5
    kab = np.zeros((ky, kx, 3), np.float32); wtop = np.zeros((ky, kx), np.float32)
    # sonraki görüntüler yalnız kara ve kıyı çevresinde kullanılır: açık denizdeki kadraj karartması (vinyet) dikiş bırakmasın
    yakin = ndimage.gaussian_filter((ndimage.map_coordinates(ak, [ay, ax], order=1, cval=0) > 0.5).astype(np.float32), 6)
    yakin = np.clip(ndimage.gaussian_filter(ndimage.maximum_filter(yakin, 31), 8) * 3, 0, 1)
    for yol in renkliler:                                     # sonraki görüntü öncelikli (üstüne yazar)
        R = Image.open(yol).convert("RGB"); Rw, Rh = R.size
        r4 = np.asarray(R.resize((Rw // F, Rh // F), Image.BOX)).astype(np.float32)
        rk = (r4[..., 2] - r4[..., 0]) < 8
        L, n = ndimage.label(rk); s_ = ndimage.sum(rk, L, range(1, n + 1)); rk = np.isin(L, np.nonzero(s_ >= 30)[0] + 1)
        ry, rx = np.nonzero(rk)
        a0 = (rx.max() - rx.min()) / (xx.max() - xx.min())   # yatayda kıta tam görünür kabul edilir; dikeyde kırpılmış olabilir
        p0 = [a0, a0, rx.min() - xx.min() * a0, ry.min() - yy.min() * a0]
        rf = rk.astype(np.float32); hR, wR = rk.shape

        def m(p):
            a, b, c, d = p
            qy, qx = Y * b + d, X * a + c
            ic = (qy >= 0) & (qy <= hR - 1) & (qx >= 0) & (qx <= wR - 1)
            q = ndimage.map_coordinates(rf, [qy, qx], order=1, cval=0) > 0.5
            return -(q & akk & ic).sum() / max(((q | akk) & ic).sum(), 1)
        r = optimize.minimize(m, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 1000})
        a, b, c, d = r.x; print(yol, "kayıt", np.round(r.x, 4), "IoU (görünen kısım)", round(-r.fun, 4))
        by, bx = ay * b + d, ax * a + c
        olc = max(kb["sx"] / sx * a, 1.0)
        rs = ndimage.gaussian_filter(r4, (0.45 * olc, 0.45 * olc, 0)) if olc > 1.2 else r4
        g = np.stack([ndimage.map_coordinates(rs[..., k], [by, bx], order=1, mode="nearest") for k in range(3)], -1)
        pay = 250.0                                            # kadraj kenarından 250 piksel (renkli/4) içeriye kadar hiç, 500'de tam: kenardaki vinyet karartması girmesin
        w = np.clip((np.minimum.reduce([by, hR - 1 - by, bx, wR - 1 - bx]) - pay) / pay, 0, 1).astype(np.float32)
        if yol == renkliler[0]:
            kab = g                                            # ilk görüntü her yeri doldurur (kadraj dışı en yakın kenar)
        else:
            w = w * yakin
            kab = kab * (1 - w[..., None]) + g * w[..., None]
        wtop = np.maximum(wtop, w)
    if (wtop < 0.99).any():
        print("uyarı: hiçbir görüntünün kapsamadığı kabartma pikseli", int((wtop < 0.5).sum()))
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
