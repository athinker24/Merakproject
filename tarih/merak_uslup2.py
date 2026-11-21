"""'Son' haritada bizim eklediğimiz karayı sahibin çalışma biçimine göre yeniden yapar. Yalnız 16k.

Sahibin yöntemi: genel hatlar korunur, küçük ayrıntı Merak'ın aşınma aracıyla deneme yanılma üretilir. Buna göre:
1) taban: bizim yükseklik, önceki ölçümden gelen yükseklik eşlemesiyle, 15 km ölçeğinde süzülür (genel hatlar);
2) sırtlar 50 km bandında hafifçe keskinleştirilir;
3) ince doku sahibin elle işlediği bölgelerden (2, şerit, 3) parça parça kopyalanır (yamalı doku sentezi) ve
   genliği, sahibin dokusunun yüksekliğe göre ölçülen genliğine ayarlanır. Böylece doku Merak aşınmasının kendisidir.
Kıyı çizgisi, deniz ve sahibin elle işlediği pikseller değişmez. Kullanım: python3 merak_uslup2.py
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from merak_birlestir import kodla, S
import merak_uslup as U

Image.MAX_IMAGE_PIXELS = None
P = U.P
Y0, Y1, X0, X1 = U.Y0, U.Y1, U.X0, U.X1
SIG = 6                       # doku/taban ayrımı (16k pikseli, yaklaşık 15 km)
YAMA, ADIM = 128, 64
rng = np.random.default_rng(23)
KUTU = [0, 300, 700, 1200, 2000, 3000, 9000]


def genlik_tablosu(A, g, kaynak):
    hp = A - g; orta, std = [], []
    for lo, hi in zip(KUTU[:-1], KUTU[1:]):
        m = kaynak & (g >= lo) & (g < hi)
        orta.append(np.median(g[m])); std.append(hp[m].std())
    return np.array(orta), np.array(std)


def main():
    A = np.load(P + "A16.npy"); B = np.load(P + "B16.npy"); LB = np.load(P + "LB16.npy"); W = np.load(P + "W16.npy")
    land = A > 0
    _, (iy, ix) = ndimage.distance_transform_edt(LB == 0, return_indices=True); LB = LB[iy, ix]
    kd = ndimage.distance_transform_edt(land)
    kaynak = land & (W == 0) & (kd > 25) & np.isin(LB, [2, 3, 4])
    gA = ndimage.gaussian_filter(A, SIG)
    xo, so = genlik_tablosu(A, gA, kaynak)
    # normalleştirilmiş doku kaynağı
    t = (A - gA) / np.interp(gA, xo, so)
    t = np.clip(t, -4, 4)
    # yamalar: tamamı kaynakta olan pencereler
    kk = ndimage.uniform_filter(kaynak.astype(np.float32), YAMA) > 0.97
    ys, xs = np.nonzero(kk[YAMA // 2:-YAMA // 2:8, YAMA // 2:-YAMA // 2:8])
    ys = ys * 8 + YAMA // 2 - YAMA // 2; xs = xs * 8 + YAMA // 2 - YAMA // 2
    print("aday yama", len(ys))
    # yamalı doku sentezi (Hann pencereleri yarı örtüşmeyle toplamı 1)
    H, Wd = A.shape
    han = np.hanning(YAMA + 1)[:-1]; pen = np.outer(han, han)
    doku = np.zeros((H + 2 * YAMA, Wd + 2 * YAMA), np.float32); top = np.zeros_like(doku)
    for y in range(-ADIM, H, ADIM):
        for x in range(-ADIM, Wd, ADIM):
            k = rng.integers(len(ys)); p = t[ys[k]:ys[k] + YAMA, xs[k]:xs[k] + YAMA]
            p = np.rot90(p, rng.integers(4))
            if rng.random() < 0.5:
                p = p[:, ::-1]
            yy, xx = y + ADIM, x + ADIM
            doku[yy:yy + YAMA, xx:xx + YAMA] += p * pen; top[yy:yy + YAMA, xx:xx + YAMA] += pen
    doku = (doku / np.maximum(top, 1e-3))[ADIM:ADIM + H, ADIM:ADIM + Wd]
    doku /= doku[land].std() + 1e-6
    # taban: önceki yükseklik eşlemesi + 15 km süzgeç + 50 km bandında sırt keskinleştirme
    tr = land & np.isin(LB, [2, 3, 4]) & (A != B)
    qs = np.linspace(0.5, 99.95, 80)
    na = np.maximum.accumulate(np.r_[0, np.percentile(A[tr], qs), A[tr].max() * 1.02])
    nb = np.maximum.accumulate(np.r_[0, np.percentile(B[tr], qs), B[tr].max() * 1.02]) + np.arange(82) * 1e-3
    m = np.where(land, np.interp(A, nb, na), A)
    # yalnız karada süzgeç (deniz kıyıyı aşağı çekmesin); dağlık yerde daha az süzülür, sıraların biçimi kalsın
    def kara_suz(z, s):
        return ndimage.gaussian_filter(np.where(land, z, 0), s) / np.maximum(ndimage.gaussian_filter(land.astype(np.float32), s), 1e-3)
    g6, g2 = kara_suz(m, SIG), kara_suz(m, 2)
    sd = np.sqrt(np.maximum(kara_suz(m ** 2, 20) - kara_suz(m, 20) ** 2, 0))
    k = np.clip((sd - 80) / 250, 0, 1)
    taban = (1 - k) * g6 + k * g2
    bant = taban - kara_suz(taban, 20)
    taban = taban + np.where(bant > 0, 0.25 * bant, 0.05 * bant)
    yeni = taban + doku * np.interp(taban, xo, so) * (1 - 0.6 * k)
    # kıyı: ilk 10 km'de doku azalır, kıyı düşük kalır
    yeni = np.where(kd < 4, np.minimum(yeni, A + 20), yeni)
    R = np.where(land, W * np.maximum(yeni, 2) + (1 - W) * A, A)
    g = np.asarray(Image.open("merak/son_temiz_16k.png")).astype(np.int32)
    out = g.copy(); blok = out[Y0:Y1, X0:X1]; blok[land] = kodla(np.maximum(R[land], 2)); out[Y0:Y1, X0:X1] = blok
    Image.fromarray(out.astype(np.uint16)).save("merak/uslup2_16k.png")
    olc = {"genlik_tablosu": {"yukseklik": xo.round().tolist(), "doku_std": so.round(1).tolist()},
           "sahip": U.ozet(A, kaynak), "bizim_bolge_sonra": U.ozet(R, W > 0.99),
           "kara_farki_px": int(((out > S) != (g > S)).sum()), "deniz_farki_px": int((out[g <= S] != g[g <= S]).sum())}
    json.dump(olc, open("merak/uslup2_olcum.json", "w"), ensure_ascii=False, indent=1)
    np.save(P + "R2.npy", R)
    print(olc)


if __name__ == "__main__":
    main()
