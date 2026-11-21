"""Merak "Place on existing" için: v2 dünya çerçevesinde (8000 × 4000, Fill world) bölge başına heightmap.

Kodlama v2_yukseklik ile aynı (%22 gri deniz seviyesi, beyaz en yüksek). Bölge dışı saydam (RGBA) ya da siyah (16 bit gri).
Bölgenin işlenmiş komşu karayla sınırında saydamlık 15 pikselde yumuşak geçer; kıyıda keskin kalır.
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage

BOLGE = {"guneybati": [5, 6], "kuzeybati": [1], "ada": [7], "tum_kalan": [1, 5, 6, 7]}
GECIS = 15


def main():
    o = json.load(open("merak/v2_olcek.json")); c0, r0, ww, hh = o["kita_kutusu_px"]
    g = np.asarray(Image.open("merak/v2_yukseklik_16bit.png"))
    lab = np.load("merak/v2_parca.npy"); kara = lab > 0
    H, W = g.shape
    for ad, ks in BOLGE.items():
        b = np.isin(lab, ks)
        # içeriden sınıra uzaklık yalnız diğer karaya karşı; denize karşı sınır yok sayılır
        d = ndimage.distance_transform_edt(~(kara & ~b))
        a = np.where(b, np.clip(d / GECIS, 0, 1), 0)
        a = np.where(b & (d >= GECIS), 1.0, a)
        alfa = np.zeros((H, W), np.float32); alfa[r0:r0 + hh, c0:c0 + ww] = a
        # deniz kenarı: bölgenin 6 piksel çevresindeki denizi de dahil et, kıyı Merak'ın mevcut denizine otursun
        deniz = ndimage.binary_dilation(b, iterations=6) & ~kara
        alfa[r0:r0 + hh, c0:c0 + ww] = np.maximum(alfa[r0:r0 + hh, c0:c0 + ww], deniz.astype(np.float32))
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., :3] = np.round(g / 257).astype(np.uint8)[..., None]
        rgba[..., 3] = np.round(alfa * 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(f"merak/yerlestir_{ad}_saydam.png", optimize=True)
        Image.fromarray(np.where(alfa > 0.5, g, 0).astype(np.uint16)).save(f"merak/yerlestir_{ad}_16bit.png")
        print(ad, int(b.sum()))


if __name__ == "__main__":
    main()
