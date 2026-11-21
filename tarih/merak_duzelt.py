"""Merak birleşik haritasında sahibin işaretlediği yapay görünen kara alanlarını bizim yüksekliklerle değiştirir.

Bölgeler (Merak 8k dışa aktarım koordinatları): Mavi Dağlar'ın batısındaki düz şerit ve Sarı Ön Havzası ile Uzun Yarımada.
Yalnız kara pikselleri değişir; deniz pikselleri gri değeriyle aynen kalır. Kenarda 15 piksellik (8k) geçiş.
Kullanım: python3 merak_duzelt.py GIRDI_16k.png CIKTI_16k.png
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage
from matplotlib.path import Path
from merak_birlestir import coz, kodla, S

Image.MAX_IMAGE_PIXELS = None
H8, W8 = 3859, 8000


def _p(c):
    return [(3150 + x * 1.7, 800 + y * 1.7) for x, y in c]


BOLGE = [_p([(370, 40), (400, 90), (390, 170), (370, 240), (335, 300), (290, 290), (265, 230), (275, 120), (300, 60)]),
         _p([(470, 465), (520, 500), (520, 580), (500, 630), (455, 660), (425, 610), (415, 530), (435, 480)])]
GECIS = 15


def bizim(y8, x8):
    """Merak 8k koordinatlarında bizim yükseklik (tam çözünürlüklü benzer_z; dışında v2 dolgusu)."""
    o = json.load(open("merak/v2_olcek.json")); c0, r0 = o["kita_kutusu_px"][:2]
    s, sy, ox, oy = json.load(open("merak/birlesik_olcek.json"))["kayit"]
    ksy, ksx, koy, kox = o["kayit"]
    z = np.load("cikti/benzer_z.npy").astype(np.float32)
    kk = (np.asarray(Image.open("maske.png").convert("L")) > 127).astype(np.float32)
    kara5 = np.asarray(Image.open("merak/v2_kita_kara_16bit.png")).astype(np.float32) / 65535 * o["en_yuksek_m"]
    ry, rx = y8 * sy + oy - r0, x8 * s + ox - c0
    my, mx = ry * ksy + koy, rx * ksx + kox
    biz = ndimage.map_coordinates(z, [my, mx], order=1)
    bk = ndimage.map_coordinates(kk, [my, mx], order=1) > 0.5
    return np.maximum(np.where(bk, biz, ndimage.map_coordinates(kara5, [ry, rx], order=1)), 5)


def main():
    g = np.asarray(Image.open(sys.argv[1])).astype(np.int32); H, W = g.shape
    fy, fx = H8 / H, W8 / W
    # bölge ağırlığı 8k'da
    yy, xx = np.mgrid[0:H8, 0:W8]
    ic = np.zeros((H8, W8), bool)
    for b in BOLGE:
        x0, y0 = np.min(b, 0).astype(int) - 2; x1, y1 = np.max(b, 0).astype(int) + 3
        pts = np.c_[xx[y0:y1, x0:x1].ravel(), yy[y0:y1, x0:x1].ravel()]
        ic[y0:y1, x0:x1] |= Path(b).contains_points(pts).reshape(y1 - y0, x1 - x0)
    w8 = np.clip(ndimage.distance_transform_edt(ic) / GECIS, 0, 1).astype(np.float32)
    ys, xs = np.nonzero(ic); a0, a1 = int(ys.min() / fy) - 2, int(ys.max() / fy) + 3
    out = g.copy()
    Y, X = np.mgrid[a0:a1, 0:W].astype(np.float32)
    y8, x8 = (Y + 0.5) * fy - 0.5, (X + 0.5) * fx - 0.5
    w = ndimage.map_coordinates(w8, [y8, x8], order=1)
    gg = g[a0:a1]; sel = (w > 1e-3) & (gg > S)
    h = bizim(y8[sel], x8[sel])
    blok = out[a0:a1]; blok[sel] = kodla(w[sel] * h + (1 - w[sel]) * coz(gg[sel])); out[a0:a1] = blok
    Image.fromarray(out.astype(np.uint16)).save(sys.argv[2])
    print("değişen piksel", int(sel.sum()))


if __name__ == "__main__":
    main()
