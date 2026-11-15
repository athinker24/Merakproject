"""Merak çerçevesine (tüm gezegen, eşdikdörtgen) yerleştirilmiş yükseklik haritası ve maske.

Kıyı maskemiz Merak'ın çıktısındaki kıtaya ölçek ve kaymayla oturtulur (örtüşme en büyüklenir), sonra kara yükseklikleri ve
deniz tabanı bu çerçeveye taşınır. Kendi çerçevemizin dışındaki okyanus, kenardaki derinliklerden yumuşakça doldurulur.
Kullanım: python3 merak_cikti.py MERAK_GORSELI
Çıktı: merak/merak_yukseklik_16bit.png, merak/merak_maske.png, merak/merak_onizleme.jpg, merak/merak_olcek.json
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize

Image.MAX_IMAGE_PIXELS = None


def main():
    ref = np.asarray(Image.open(sys.argv[1]).convert("L")).astype(np.float32)
    H, W = ref.shape
    kara = np.asarray(Image.open("maske.png").convert("L")) > 127
    z = np.load("cikti/benzer_z.npy").astype(np.float64)
    zt = np.load("cikti/son_z_tam.npy").astype(np.float64)
    my, mx = kara.shape
    hedef = ndimage.gaussian_filter(ref, 3) > 70
    yy, xx = np.nonzero(hedef); ky, kx = np.nonzero(kara)
    sx = (xx.max() - xx.min()) / (kx.max() - kx.min()); sy = (yy.max() - yy.min()) / (ky.max() - ky.min())
    p0 = np.array([sx, sy, xx.min() - kx.min() * sx, yy.min() - ky.min() * sy])
    # kaba ızgarada örtüşme en büyüklenir
    f = 4
    hk = hedef[::f, ::f]; R, C = np.mgrid[0:hk.shape[0], 0:hk.shape[1]].astype(np.float64) * f
    mf = kara.astype(np.float32)
    def eksi(p):
        a, b, c, d = p
        m = ndimage.map_coordinates(mf, [(R - d) / b, (C - c) / a], order=1, cval=0) > 0.5
        return -(m & hk).sum() / max((m | hk).sum(), 1)
    son = optimize.minimize(eksi, p0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 800})
    a, b, c, d = son.x
    print("kayıt", son.x, "örtüşme", -son.fun, "başlangıç", -eksi(p0))
    R, C = np.mgrid[0:H, 0:W].astype(np.float32)
    my_, mx_ = (R - d) / b, (C - c) / a
    icerde = (my_ >= 0) & (my_ <= my - 1) & (mx_ >= 0) & (mx_ <= mx - 1)
    zz = np.where(kara, np.maximum(z, 1.0), np.minimum(zt, -5.0))
    h = ndimage.map_coordinates(zz, [my_, mx_], order=1, mode="nearest").astype(np.float32)
    km = ndimage.map_coordinates(mf, [my_, mx_], order=1, cval=0) > 0.5
    # çerçeve dışı okyanus: kenar derinliğinin yumuşatılmışı, 4000 m'ye doğru
    kenar = np.where(icerde, h, np.nan)
    dolgu = np.full_like(h, -4200.0)
    w = ndimage.gaussian_filter(icerde.astype(np.float32), 40)
    v = ndimage.gaussian_filter(np.where(icerde, h, 0), 40) / np.maximum(w, 1e-3)
    dolgu = np.where(w > 0.02, np.minimum(v, -200), -4200.0)
    dolgu = ndimage.gaussian_filter(dolgu, 60)
    h = np.where(icerde, h, dolgu)
    h = np.where(km, np.maximum(h, 1.0), np.minimum(h, -5.0))
    zmax, zmin = float(h.max()), float(h.min())
    q = np.round((h - zmin) / (zmax - zmin) * 65535).astype(np.uint16)
    deniz = int(round(-zmin / (zmax - zmin) * 65535))
    Image.fromarray(q).save("merak/merak_yukseklik_16bit.png")
    Image.fromarray((km * 255).astype(np.uint8)).save("merak/merak_maske.png")
    on = np.where(km, 60 + 195 * np.clip(h / zmax, 0, 1) ** 0.6, 20 * np.clip(1 + h / 6000, 0, 1))
    o = Image.fromarray(on.astype(np.uint8)); o.thumbnail((2000, 2000)); o.save("merak/merak_onizleme.jpg", quality=88)
    ref_m = Image.fromarray(np.stack([(hedef * 255).astype(np.uint8), (km * 255).astype(np.uint8), np.zeros_like(q, np.uint8)], -1))
    ref_m.thumbnail((2000, 2000)); ref_m.save("merak/merak_kayit_kontrol.jpg", quality=85)
    json.dump({"boyut": [W, H], "kayit": {"olcek_x": a, "olcek_y": b, "kayma_x": c, "kayma_y": d,
                "not": "Merak pikseli (satır, sütun) = (maske satırı × olcek_y + kayma_y, maske sütunu × olcek_x + kayma_x)"},
               "ortusme": -son.fun, "deniz_seviyesi_degeri": deniz, "en_derin_m": round(zmin), "en_yuksek_m": round(zmax),
               "metre": "değer / 65535 × %.0f − %.0f" % (zmax - zmin, -zmin)},
              open("merak/merak_olcek.json", "w"), ensure_ascii=False, indent=1)
    print("boyut", W, H, "deniz", deniz, "aralık", zmin, zmax)


if __name__ == "__main__":
    main()
