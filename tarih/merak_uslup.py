"""'Son' haritada bizim eklediğimiz (sahibin elle işlemediği) karayı sahibin üslubuna yaklaştırır. Yalnız 16k.

Üslup, sahibin elle işlediği 2, şerit ve 3'te iki sürüm karşılaştırılarak ölçüldü (merak/uslup_olcum.json):
yükseklik dağılımı (nicelik eşleme), ince doku gücü yüksekliğe göre (ovada daha az, dağda %30 dolayı daha çok),
sırt keskinliği (30 km bandında tepe yukarı, vadi aşağı asimetrisi). Bizim kara: sahibin ilk dışa aktarımında elle
dokunulmamış (düz) yerler ve işaretli iki bölge. Kıyı çizgisi ve deniz değişmez.
Kullanım: python3 merak_uslup.py
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from matplotlib.path import Path
from merak_birlestir import coz, kodla, S
from merak_duzelt import BOLGE

Image.MAX_IMAGE_PIXELS = None
P = "/tmp/claude-0/-home-user-Tectonic/1cb340b3-df7d-519b-96f4-5170b8cfb37a/scratchpad/"
Y0, Y1, X0, X1 = 1600, 4200, 6300, 9700
rng = np.random.default_rng(5)


def kazanc(h, tablo):
    xs, ys = zip(*tablo); return np.interp(h, xs, ys)


INCE = [(0, 1.0), (150, 1.0), (300, 0.45), (550, 0.7), (800, 1.2), (2000, 1.05), (8000, 1.05)]


OVA = [(0, 1.0), (150, 1.0), (300, 0.45), (600, 0.75), (900, 1.0), (8000, 1.0)]


def donustur(h, nq_b, nq_a, land, sirt=(1.2, 0.9), ince=INCE, gurultu=0.4, ova=OVA):
    # 1) nicelik eşleme (yalnız yükseklik dağılımı)
    m = np.interp(h, nq_b, nq_a)
    # 2) sırt keskinliği: 30 km bandında asimetrik kazanç
    taban = ndimage.gaussian_filter(m, 12)
    orta = ndimage.gaussian_filter(m, 1.5) - taban
    ince_d = m - ndimage.gaussian_filter(m, 1.5)
    orta = np.where(orta > 0, orta * sirt[0], orta * sirt[1]) * kazanc(taban, ova)
    # 3) ince doku yüksekliğe göre
    g = kazanc(taban + orta, ince)
    n = ndimage.gaussian_filter(rng.standard_normal(h.shape).astype(np.float32), 1.1)
    n *= (8 + 0.025 * np.clip(taban + orta, 0, None)) * gurultu / (n.std() + 1e-6)
    sonuc = taban + orta + ince_d * g + n * (taban + orta > 400)
    return np.where(land, np.maximum(sonuc, 2), h)


def ozet(z, m):
    gy, gx = np.gradient(z, 2500.0); e = np.degrees(np.arctan(np.hypot(gx, gy)))
    hp = z - ndimage.gaussian_filter(z, 1.5)
    q = np.percentile(z[m], [10, 30, 50, 70, 90, 98, 99.5]).round().astype(int).tolist()
    bins = [round(float(hp[m & (z >= lo) & (z < hi)].std()), 1) for lo, hi in [(200, 500), (500, 1000), (1000, 2000), (2000, 7000)]]
    return {"nicelik": q, "ince_std": bins, "egim5": round(float((e[m] > 5).mean()), 4), "egim2": round(float((e[m] > 2).mean()), 4)}


def main():
    A = np.load(P + "A16.npy"); B = np.load(P + "B16.npy"); O = np.load(P + "O16.npy"); LB = np.load(P + "LB16.npy")
    land = A > 0
    _, (iy, ix) = ndimage.distance_transform_edt(LB == 0, return_indices=True)
    LB = LB[iy, ix]                                          # 16k kıyı pikselleri 8k etiketinde denize düşebiliyor
    tr = land & np.isin(LB, [2, 3, 4]) & (A != B)
    qs = np.linspace(0.5, 99.95, 80)
    nq_a, nq_b = np.percentile(A[tr], qs), np.percentile(B[tr], qs)
    nq_a = np.r_[0, nq_a, A[tr].max() * 1.02]; nq_b = np.r_[0, nq_b, B[tr].max() * 1.02]
    nq_a = np.maximum.accumulate(nq_a); nq_b = np.maximum.accumulate(nq_b) + np.arange(len(nq_b)) * 1e-3
    # doğrulama: eğitim bölgesinde bizim sürüme uygula, sahibinkine benziyor mu
    Tb = donustur(B, nq_b, nq_a, land)
    olc = {"sahip": ozet(A, tr), "bizim": ozet(B, tr), "bizim_donusmus": ozet(Tb, tr)}
    print(json.dumps(olc, ensure_ascii=False))
    # bizim kara maskesi: sahibin dokunmadığı düz yerler (ilk dışa aktarım) ve işaretli bölgeler
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(O ** 2, 17) - ndimage.uniform_filter(O, 17) ** 2, 0))
    u = (O > 0) & (O < 330) & (sd < 25)
    u = ndimage.binary_closing(ndimage.binary_opening(u, iterations=5), iterations=12)
    L, n = ndimage.label(u); s = ndimage.sum(u, L, range(1, n + 1)); u = np.isin(L, np.nonzero(s >= 1600)[0] + 1)
    u &= np.isin(LB, [1, 5, 6, 7, 8])
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    pts = np.c_[(xx.ravel() + 0.5) / 2, (yy.ravel() + 0.5) * 3859 / 7719]
    for b in BOLGE:
        u |= Path(b).contains_points(pts).reshape(u.shape)
    u = ndimage.binary_dilation(u, iterations=12) & land
    # kıyı şeridi: bizim bölgeye komşu kıyı bandı da bizim (ilk dışa aktarımda kıyı rampası düz sayılmamıştı)
    kiyi = ndimage.distance_transform_edt(land) < 45
    u |= kiyi & land & np.isin(LB, [1, 5, 6, 7, 8]) & ndimage.binary_dilation(u, iterations=60)
    W = np.clip(ndimage.distance_transform_edt(u) / 30, 0, 1)
    W = np.where(land & ~u, 0, W)
    # kıyıya kadar tam ağırlık: denize karşı geçiş yok
    W = np.where(u, np.clip(ndimage.distance_transform_edt(~(land & ~u)) / 30, 0, 1), 0)
    T = donustur(A, nq_b, nq_a, land)
    R = np.where(land, W * T + (1 - W) * A, A)
    g = np.asarray(Image.open("merak/son_temiz_16k.png")).astype(np.int32)
    out = g.copy(); blok = out[Y0:Y1, X0:X1]
    blok[land] = kodla(np.maximum(R[land], 2)); out[Y0:Y1, X0:X1] = blok
    Image.fromarray(out.astype(np.uint16)).save("merak/uslup_16k.png")
    olc["bizim_bolge_once"] = ozet(A, u & (W > 0.99)); olc["bizim_bolge_sonra"] = ozet(R, u & (W > 0.99))
    olc["kara_farki_px"] = int(((out > S) != (g > S)).sum()); olc["deniz_farki_px"] = int((out[g <= S] != g[g <= S]).sum())
    json.dump(olc, open("merak/uslup_olcum.json", "w"), ensure_ascii=False, indent=1)
    np.save(P + "R16.npy", R); np.save(P + "W16.npy", W.astype(np.float32))
    print({k: olc[k] for k in ("bizim_bolge_once", "bizim_bolge_sonra", "kara_farki_px", "deniz_farki_px")})


if __name__ == "__main__":
    main()
