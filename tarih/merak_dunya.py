"""Merak için tüm dünya (eşdikdörtgen, 2:1) görselleri: kıta bizim ölçeğimizle doğru enlem ve boylamlara oturtulur.

Bizim harita: 8000 km genişlik, üst kenar 50°K, alt kenar yaklaşık 5,07°G (1 görüntü pikseli = 4 km). Her enlemde boylam açıklığı
x_km / (111,32 × cos enlem) ile hesaplanır, yani kıta kuzeye doğru eşdikdörtgen projeksiyonda genişler (küre üzerinde doğru şekil).
Kıtanın ortası MERKEZ_BOYLAM'da. Merak'ta "Fill world" ile yerleştirilir.
Kodlama (Merak): siyah deniz tabanı, beyaz en yüksek, deniz seviyesi %22 gri.
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage
import olaylar as O

Image.MAX_IMAGE_PIXELS = None
GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
MERKEZ_BOYLAM = 0.0
DENIZ_GRI = 0.22


def main():
    W, H = GEN, GEN // 2
    kara = np.asarray(Image.open("maske.png").convert("L")) > 127
    z = np.load("cikti/benzer_z.npy").astype(np.float32)
    zt = np.load("cikti/son_z_tam.npy").astype(np.float32)
    my, mx = kara.shape
    km_px = O.KM_PX * O.GEN / mx                      # maske pikseli km (yaklaşık 2,1)
    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    lon = -180 + (np.arange(W) + 0.5) * 360 / W
    satir = (50 - lat) * 111.2 / km_px                 # maske satırı
    zz = np.where(kara, np.maximum(z, 1.0), np.minimum(zt, -5.0))
    yuk = np.full((H, W), np.nan, np.float32); mk = np.zeros((H, W), bool)
    for i in np.nonzero((satir >= 0) & (satir <= my - 1))[0]:
        sutun = (lon - MERKEZ_BOYLAM) * 111.32 * np.cos(np.radians(lat[i])) / km_px + mx / 2
        ok = (sutun >= 0) & (sutun <= mx - 1)
        if not ok.any():
            continue
        r = np.full(ok.sum(), satir[i]); c = sutun[ok]
        yuk[i, ok] = ndimage.map_coordinates(zz, [r, c], order=1)
        mk[i, ok] = ndimage.map_coordinates(kara.astype(np.float32), [r, c], order=1) > 0.5
    icerde = ~np.isnan(yuk)
    w = ndimage.gaussian_filter(icerde.astype(np.float32), GEN / 200)
    v = ndimage.gaussian_filter(np.where(icerde, yuk, 0), GEN / 200) / np.maximum(w, 1e-3)
    dolgu = ndimage.gaussian_filter(np.where(w > 0.02, np.minimum(v, -200), -4200.0), GEN / 140)
    h = np.where(icerde, yuk, dolgu)
    h = np.where(mk, np.maximum(h, 1.0), np.minimum(h, -5.0))
    zmax, zmin = float(h.max()), float(h.min())
    g = np.where(h >= 0, DENIZ_GRI + (1 - DENIZ_GRI) * h / zmax, DENIZ_GRI * (1 - h / zmin))
    Image.fromarray(np.round(g * 65535).astype(np.uint16)).save("merak/dunya_yukseklik_16bit.png")
    Image.fromarray(np.round(g * 255).astype(np.uint8)).save("merak/dunya_yukseklik_8bit.png")
    Image.fromarray((mk * 255).astype(np.uint8)).save("merak/dunya_maske.png")
    on = Image.fromarray(np.round(g * 255).astype(np.uint8)); on.thumbnail((2000, 1000)); on.save("merak/dunya_onizleme.jpg", quality=88)
    ys, xs = np.nonzero(mk)
    json.dump({"boyut": [W, H], "projeksiyon": "eşdikdörtgen, tüm dünya; Merak'ta Fill world",
               "kita": {"enlem": [round(float(lat[ys.max()]), 2), round(float(lat[ys.min()]), 2)],
                        "boylam": [round(float(lon[xs.min()]), 2), round(float(lon[xs.max()]), 2)], "merkez_boylam": MERKEZ_BOYLAM},
               "kodlama": "gri = 0,22 + 0,78 × m / %.0f (kara); gri = 0,22 × (1 − m / %.0f) (deniz)" % (zmax, zmin),
               "deniz_seviyesi_gri": DENIZ_GRI, "en_yuksek_m": round(zmax), "en_derin_m": round(zmin)},
              open("merak/dunya_olcek.json", "w"), ensure_ascii=False, indent=1)
    print("boyut", W, H, "kıta enlem", lat[ys.max()], lat[ys.min()], "boylam", lon[xs.min()], lon[xs.max()])


if __name__ == "__main__":
    main()
