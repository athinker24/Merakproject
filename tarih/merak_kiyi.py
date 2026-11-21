"""Merak birleşik haritasında kıyı yüksekliklerinin düzeltilmesi: alçalan kıyı şeridi ve kıta sahanlığı. Kıyı çizgisi değişmez.

Ölçüm (birlesik_16k): kıyıdan 2,5 ile 10 km içeride kara ortancası 350 ile 430 m, iç kesimden (260 m) yüksek; deniz kıyıdan
5 ile 15 km dışarıda −4300 m'ye iniyor (sahanlık yok). Bu ikisi kıyıları "yüksek" gösteriyor.
Girdi/çıktı Merak kodlamasında (merak_birlestir.py). Kullanım: python3 merak_kiyi.py
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from merak_birlestir import coz, kodla, S, DD, DP

Image.MAX_IMAGE_PIXELS = None
KM = 2.5                     # 16k pikseli yaklaşık km
rng = np.random.default_rng(11)


def gurultu(shape, olcekler):
    t = np.zeros(shape, np.float32)
    for s, a in olcekler:
        n = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), s)
        t += a * n / (n.std() + 1e-6)
    return t


def deniz_kodla(m):
    return np.round(S * (1 - np.clip(-m / DD, 0, 1) ** DP)).clip(0, S - 1)


def main():
    g = np.asarray(Image.open("merak/birlesik_16k.png")).astype(np.int32)
    H, W = g.shape
    E = coz(g)
    ys, xs = np.nonzero(E[:, W // 3:] > 0.5)
    y0, y1 = max(ys.min() - 400, 0), min(ys.max() + 400, H)
    x0, x1 = xs.min() + W // 3 - 400, xs.max() + W // 3 + 400
    h = E[y0:y1, x0:x1].copy(); kara = h > 0; sh = h.shape
    # kıyı çizgisi ve kara parçaları olduğu gibi kalır (sahibin isteği): yalnız yükseklikler düzeltilir
    yeni = kara
    # 3) kara yükseklikleri: yeni kara en yakın eski karadan, sonra kıyı rampası
    hl = h.copy()
    dk = ndimage.distance_transform_edt(yeni) * KM                                # kıyıdan içeri km
    gen = 18 * (1 + 0.6 * np.clip(gurultu(sh, [(40, 1)]), -1.2, 1.2))            # rampa genişliği 7 ile 31 km
    t = np.clip(dk / gen, 0, 1); t = t * t * (3 - 2 * t)
    kiyi_tavan = 3 + 25 * np.clip(dk / 4, 0, 1)
    hl = np.maximum(hl * (0.12 + 0.88 * t), np.minimum(hl, kiyi_tavan)) + 4 * np.clip(gurultu(sh, [(1.5, 1)]), 0, None)
    # kıyı kenarındaki tümseği (iç kesimden yüksek şerit) bastır: rampa bölgesinde iç ortalamanın üstüne çıkma
    ic = ndimage.uniform_filter(np.where(yeni, hl, 0), 25) / np.maximum(ndimage.uniform_filter(yeni.astype(np.float32), 25), 1e-3)
    hl = np.where(t < 1, np.minimum(hl, np.maximum(ic * (0.4 + 0.6 * t) + 30, hl * t)), hl)
    hl = np.maximum(hl, 2)
    # 4) sahanlık ve yamaç
    dsk = ndimage.distance_transform_edt(~yeni) * KM
    sah = 45 * (1 + 0.7 * np.clip(gurultu(sh, [(50, 1)]), -1.1, 1.3))            # sahanlık genişliği 0 ile 100 km
    sah = np.maximum(sah, 6)
    derin_sah = -12 - 140 * np.clip(dsk / sah, 0, 1) ** 1.3
    t2 = np.clip((dsk - sah) / 90, 0, 1); t2 = t2 * t2 * (3 - 2 * t2)
    profil = derin_sah + (-4300 - derin_sah) * t2
    profil = ndimage.gaussian_filter(profil, 8)                                  # uzaklık dönüşümünün sırt çizgilerini sil
    eski_deniz = np.where(kara, -4300, h)
    hs = np.maximum(eski_deniz, profil) + 6 * gurultu(sh, [(2, 1)]) * (dsk < sah)
    hs = np.minimum(hs, -3)
    out = g.copy()
    out[y0:y1, x0:x1] = np.where(yeni, kodla(hl), deniz_kodla(hs)).astype(np.int32)
    Image.fromarray(out.astype(np.uint16)).save("merak/kiyi_16k.png")
    m16 = coz(out)
    m8 = np.asarray(Image.fromarray(m16.astype(np.float32), "F").resize((W // 2, (H + 1) // 2 if H % 2 else H // 2), Image.BOX))
    m8 = np.asarray(Image.fromarray(m16.astype(np.float32), "F").resize((8000, 3859), Image.BOX))
    k8 = np.asarray(Image.open("merak/birlesik_8k.png")).astype(np.int32) > S            # 8k kıyı çizgisi de aynen kalır
    Image.fromarray(np.where(k8, kodla(np.maximum(m8, 2)), deniz_kodla(np.minimum(m8, -3))).astype(np.uint16)).save("merak/kiyi_8k.png")
    # ölçüm
    k2 = yeni; d = ndimage.distance_transform_edt(k2)
    for a, b in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16)]:
        mm = (d > a) & (d <= b); print("kara", a, b, int(np.median(hl[mm])))
    L, n = ndimage.label(k2); s = ndimage.sum(k2, L, range(1, n + 1))
    print("kara parçası", n, "eski", int(ndimage.label(kara)[1]), "kara piksel farkı", int(k2.sum() - kara.sum()))


if __name__ == "__main__":
    main()
