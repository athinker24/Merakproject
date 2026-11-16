"""Merak için tüm dünya (eşdikdörtgen, 2:1) görselleri: kıta bizim ölçeğimizle doğru enlem ve boylamlara oturtulur.

Bizim harita: 8000 km genişlik, üst kenar 50°K, alt kenar yaklaşık 5,07°G (1 görüntü pikseli = 4 km). Harita bükülmeden, oranı korunarak konur
(sahibin isteği): bir maske pikseli enlemde ve boylamda aynı derece.
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
    # düz yerleşim (sahibin isteği): harita bükülmeden konur; enlem 50°K ile 5°G, boylam açıklığı oran korunarak (her satırda aynı)
    der_px = (50 - (50 - my * km_px / 111.2)) / my      # maske pikseli başına derece (enlem ve boylamda aynı)
    for i in np.nonzero((satir >= 0) & (satir <= my - 1))[0]:
        sutun = (lon - MERKEZ_BOYLAM) / der_px + mx / 2
        ok = (sutun >= 0) & (sutun <= mx - 1)
        if not ok.any():
            continue
        r = np.full(ok.sum(), satir[i]); c = sutun[ok]
        yuk[i, ok] = ndimage.map_coordinates(zz, [r, c], order=1)
        mk[i, ok] = ndimage.map_coordinates(kara.astype(np.float32), [r, c], order=1) > 0.5
    icerde = ~np.isnan(yuk)
    # haritanın dışındaki deniz: kıyıdan uzaklığa bağlı sahanlık ve yamaç (gürültülü); harita kenarına 150 km kala bizim
    # batimetriyle karıştırılır, böylece kutu kenarı düz bir çizgi olarak görünmez
    km_y = 180 / H * 111.2
    km_x = 360 / W * 111.2 * np.cos(np.radians(20))
    d = ndimage.distance_transform_edt(~mk, sampling=(km_y, km_x))
    rng = np.random.default_rng(7)
    gur = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 120 / km_y)
    gur /= gur.std() + 1e-6
    d = d * (1 + 0.25 * np.clip(gur, -1.5, 1.5))
    t = np.clip((d - 40) / 360, 0, 1); t = t * t * (3 - 2 * t)
    profil = -80 - 3920 * t
    ic = ndimage.distance_transform_edt(icerde, sampling=(km_y, km_x))
    w = np.clip((ic - 20) / 150, 0, 1)
    h = np.where(icerde, w * np.nan_to_num(yuk) + (1 - w) * profil, profil)
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
