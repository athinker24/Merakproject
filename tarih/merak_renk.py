"""Merak'ın "Colour map (assign colours)" içe aktarması için basamaklı renk haritası.

Merak her pikseli listedeki en yakın renge bağlar ve o rengin metresini verir; ara değer yoktur. Bu yüzden yükseklik
birkaç basamağa bölünür ve her basamak birbirinden uzak, tek bir renkle boyanır (kenar yumuşatma yok).
Basamak metresi o basamaktaki piksellerin ortancasıdır. Girdi merak/dunya_yukseklik_16bit.png (merak_dunya.py çıktısı).
İç deniz batimetrisindeki aşırı derinlik (−7 km dolayı) burada en derin basamağa düşer.
"""
import json
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# (alt sınır m, üst sınır m, ad, renk)
BASAMAK = [
    (-1e9, -2500, "Derin okyanus", "0A1E5A"),
    (-2500, -600, "Açık deniz", "1F5FB4"),
    (-600, 0, "Sığ / kıyı", "7FC8F0"),
    (0, 100, "Kıyı ovası", "1E8C3C"),
    (100, 300, "Ova", "8CD25A"),
    (300, 600, "Alçak yayla", "F0F046"),
    (600, 1000, "Yayla", "F0B428"),
    (1000, 1500, "Tepeler", "D2641E"),
    (1500, 2000, "Alçak dağ", "A01E1E"),
    (2000, 2700, "Dağ", "78288C"),
    (2700, 3400, "Yüksek dağ", "B482DC"),
    (3400, 4300, "Çok yüksek dağ", "505050"),
    (4300, 1e9, "Zirveler", "FFFFFF"),
]


def main():
    olc = json.load(open("merak/dunya_olcek.json"))
    zmax, zmin, dg = olc["en_yuksek_m"], olc["en_derin_m"], olc["deniz_seviyesi_gri"]
    g = np.asarray(Image.open("merak/dunya_yukseklik_16bit.png")).astype(np.float32) / 65535
    h = np.where(g >= dg, (g - dg) / (1 - dg) * zmax, (1 - g / dg) * zmin)
    kara = np.asarray(Image.open("merak/dunya_maske.png")) > 127
    h = np.where(kara, np.maximum(h, 1), np.minimum(h, -1))
    idx = np.zeros(h.shape, np.uint8); satir = []
    for k, (a, b, ad, renk) in enumerate(BASAMAK):
        m = (h >= a) & (h < b); idx[m] = k
        orta = float(np.median(h[m])) if m.any() else (a + b) / 2
        if k == 0:
            orta = -4000.0                       # batimetri hatası yüzünden ortanca yerine sabit
        satir.append({"sira": k + 1, "ad": ad, "renk": "#" + renk, "metre": int(round(orta, -1)),
                      "aralik_m": [None if a < -1e8 else a, None if b > 1e8 else b], "pay_yuzde": round(100 * m.mean(), 3)})
    im = Image.fromarray(idx, "P")
    pal = sum(([int(r[i:i + 2], 16) for i in (0, 2, 4)] for *_, r in BASAMAK), [])
    im.putpalette(pal + [0] * (768 - len(pal)))
    im.save("merak/dunya_renk_haritasi.png", optimize=True)
    json.dump(satir, open("merak/dunya_renk_listesi.json", "w"), ensure_ascii=False, indent=1)
    for s in satir:
        print(s)


if __name__ == "__main__":
    main()
