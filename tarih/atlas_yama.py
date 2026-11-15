"""Atlas girdisinde bir enlem boylam kutusunu başka bir denemenin değerleriyle değiştirir (yalnız kara pikselleri).

Kullanım: python3 atlas_yama.py KAYNAK_KLASORU GUNEY KUZEY BATI DOGU [GECIS_PX]
KAYNAK_KLASORU: o denemenin dunyam_heightmap.png, kabartma_kaynak.jpg ve heightmap_blogu.json dosyaları (git geçmişinden).
Geçiş: kutunun içinde, kenara doğru GECIS_PX piksellik doğrusal karışım; kutunun dışı değişmez.
"""
import json
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def agirlik(lat, lon, g, k, b, d, gecis_der):
    u = np.minimum(np.minimum(lat - g, k - lat), np.minimum(lon - b, d - lon)) / gecis_der
    return np.clip(u, 0, 1)


def main():
    kay, g, k, b, d = sys.argv[1], *map(float, sys.argv[2:6])
    px = float(sys.argv[6]) if len(sys.argv) > 6 else 4
    b6 = json.load(open(f"{kay}/heightmap_blogu.json"))["heightmap"]; b9 = json.load(open("atlas/heightmap_blogu.json"))["heightmap"]
    for a in ("max_elev_m", "elev_curve", "sea_level_value", "elev_gamma"):
        assert b6[a] == b9[a], a
    sea = b9["sea_level_value"]
    h9 = np.asarray(Image.open("atlas/dunyam_heightmap.png")).astype(np.float64)
    h6 = np.asarray(Image.open(f"{kay}/dunyam_heightmap.png")).astype(np.float64)
    assert ((h9 >= sea) == (h6 >= sea)).all(), "kara maskeleri farklı"
    ny, nx = h9.shape
    lat = (90 - (np.arange(ny) + 0.5) * 180 / ny)[:, None]; lon = (-180 + (np.arange(nx) + 0.5) * 360 / nx)[None, :]
    W = agirlik(lat, lon, g, k, b, d, px * 180 / ny) * (h9 >= sea)
    yeni = np.round(h9 * (1 - W) + h6 * W)
    yeni = np.where(h9 >= sea, np.maximum(yeni, sea), h9)
    Image.fromarray(yeni.astype(np.uint16)).save("atlas/dunyam_heightmap.png")
    # kabartma: görsel pikseli → tuval (satır = y·sy + oy, sütun = x·sx + ox) → enlem boylam
    kb = b9["kabartma"]
    k9 = np.asarray(Image.open("atlas/kabartma_kaynak.jpg").convert("RGB")).astype(np.float64)
    k6 = np.asarray(Image.open(f"{kay}/kabartma_kaynak.jpg").convert("RGB")).astype(np.float64)
    gy, gx = np.mgrid[0:k9.shape[0], 0:k9.shape[1]]
    klat = 90 - (gy * kb["sy"] + kb["oy"]) * 180 / ny; klon = -180 + (gx * kb["sx"] + kb["ox"]) * 360 / nx
    Wk = agirlik(klat, klon, g, k, b, d, px * 180 / ny)[..., None]
    Image.fromarray(np.clip(k9 * (1 - Wk) + k6 * Wk, 0, 255).astype(np.uint8)).save("atlas/kabartma_kaynak.jpg", quality=92)
    print("yama tamam; değişen piksel", int((yeni != h9).sum()))


if __name__ == "__main__":
    main()
