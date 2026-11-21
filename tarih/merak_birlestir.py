"""Merak'ta elle işlenmiş haritayı (Merak 16 bit dışa aktarımı) bizim yüksekliklerle birleştirir.

Merak kodlaması (sahibin altı test platosundan, metresi metresine): S = 14642 deniz seviyesi;
kara gri = S + 50893 · sqrt(m / 8000); deniz gri = S · (1 − (−m / 6840)^0,607). Okyanus tabanı 3598 ≈ −4298 m.
Elle dokunulmamış düz kara (yaklaşık 250 m, parça 1, 4, 5, 6) bizim yükseklikle doldurulur, sınırda 15 pikselde karışır.
Elle işlenmiş her piksel gri değeri değişmeden kalır. Açık okyanustaki platolar ve lekeler okyanus tabanına indirilir.
Kullanım: python3 merak_birlestir.py DISA_8K.png DISA_16K.png
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize

Image.MAX_IMAGE_PIXELS = None
S, KA, ZM, DD, DP, TABAN = 14642.0, 50893.0, 8000.0, 6840.0, 0.607, 3598
DOLDUR = [1, 5, 6, 7]            # v2_parca: parça 1, 4, 5, 6
GECIS = 15                        # 8k pikseli


def coz(g):
    g = g.astype(np.float32)
    return np.where(g >= S, ZM * np.clip((g - S) / KA, 0, None) ** 2, -DD * np.clip(1 - g / S, 0, None) ** (1 / DP))


def kodla(m):
    return np.round(S + KA * np.sqrt(np.clip(m, 0, ZM) / ZM)).clip(0, 65535)


def main():
    g8 = np.asarray(Image.open(sys.argv[1])).astype(np.int32); H8, W8 = g8.shape
    E = coz(g8); kara8 = E > 0.5
    o = json.load(open("merak/v2_olcek.json")); c0, r0, ww, hh = o["kita_kutusu_px"]
    v2 = np.asarray(Image.open("merak/dunya_maske_v2.png").convert("L")) > 127
    # v2 dünya çerçevesi -> Merak dışa aktarım çerçevesi (kara örtüşmesiyle)
    f = 2; Y, X = np.mgrid[0:H8:f, 0:W8:f].astype(np.float32); mE = kara8[::f, ::f]
    ys, xs = np.nonzero(kara8 & (np.arange(W8)[None, :] > W8 * 0.36))
    sec = (X > xs.min() - 80) & (X < xs.max() + 80) & (Y > ys.min() - 80) & (Y < ys.max() + 80)
    def maliyet(p):
        s, sy, ox, oy = p
        m = ndimage.map_coordinates(v2.astype(np.float32), [Y[sec] * sy + oy, X[sec] * s + ox], order=1) > 0.5
        a = mE[sec]; return -(m & a).sum() / (m | a).sum()
    p0 = [ww / (xs.max() - xs.min()), hh / (ys.max() - ys.min()), c0 - xs.min() * ww / (xs.max() - xs.min()),
          r0 - ys.min() * hh / (ys.max() - ys.min())]
    sim = [p0] + [[v * (1.01 if i == j and j < 2 else 1) + (5 if i == j and j >= 2 else 0) for j, v in enumerate(p0)] for i in range(4)]
    r = optimize.minimize(maliyet, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800, "initial_simplex": sim})
    s, sy, ox, oy = r.x
    # parça etiketleri Merak 8k çerçevesinde
    lab = np.load("merak/v2_parca.npy")
    YY, XX = np.mgrid[0:H8, 0:W8].astype(np.float32)
    vy, vx = YY * sy + oy - r0, XX * s + ox - c0
    LB = ndimage.map_coordinates(lab.astype(np.float32), [vy, vx], order=0, cval=0).astype(np.int16)
    _, (iy, ix) = ndimage.distance_transform_edt(LB == 0, return_indices=True)
    LB = np.where(kara8, LB[iy, ix], 0)
    # elle dokunulmamış düz kara
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(E ** 2, 9) - ndimage.uniform_filter(E, 9) ** 2, 0))
    u = kara8 & (E < 330) & (sd < 25)
    u = ndimage.binary_closing(ndimage.binary_opening(u, iterations=3), iterations=6)
    L, n = ndimage.label(u); sz = ndimage.sum(u, L, range(1, n + 1)); u = np.isin(L, np.nonzero(sz >= 400)[0] + 1)
    u = u & np.isin(LB, DOLDUR)
    u = ndimage.binary_dilation(u, iterations=6) & kara8 & np.isin(LB, DOLDUR)      # kıyıya kadar
    d_in = ndimage.distance_transform_edt(u)
    w8 = np.clip(d_in / GECIS, 0, 1).astype(np.float32)
    # kıyıda (denize karşı) geçiş yok: içeriden yalnız işlenmiş karaya uzaklık
    d_isl = ndimage.distance_transform_edt(~(kara8 & ~u))
    w8 = np.where(u, np.clip(d_isl / GECIS, 0, 1), 0).astype(np.float32)
    # açık okyanus: kıtadan 120 pikselden uzak her şey tabana
    ana = kara8 & (XX > xs.min() - 50)
    L, n = ndimage.label(ana); sz = ndimage.sum(ana, L, range(1, n + 1)); ana = np.isin(L, np.argsort(-sz)[:40] + 1) & (sz[L - 1] > 50)
    uzak8 = ndimage.distance_transform_edt(~ana) > 120
    # bizim yükseklik kaynakları
    z = np.load("cikti/benzer_z.npy").astype(np.float32)
    kk = (np.asarray(Image.open("maske.png").convert("L")) > 127).astype(np.float32)
    kara5 = np.asarray(Image.open("merak/v2_kita_kara_16bit.png")).astype(np.float32) / 65535 * o["en_yuksek_m"]
    ksy, ksx, koy, kox = o["kayit"]
    ozet = {"kayit_iou": round(-r.fun, 4), "kayit": [round(float(v), 4) for v in r.x],
            "doldurulan_pay": {int(k): round(float((u & (LB == k)).sum() / max((kara8 & (LB == k)).sum(), 1)), 3) for k in DOLDUR}}
    for yol in sys.argv[1:3]:
        g = np.asarray(Image.open(yol)).astype(np.int32); H, W = g.shape
        out = g.copy()
        fy, fx = H8 / H, W8 / W
        for a in range(0, H, 1024):
            b = min(H, a + 1024)
            yy, xx = np.mgrid[a:b, 0:W].astype(np.float32)
            y8, x8 = (yy + 0.5) * fy - 0.5, (xx + 0.5) * fx - 0.5
            w = ndimage.map_coordinates(w8, [y8, x8], order=1)
            uz = ndimage.map_coordinates(uzak8.astype(np.float32), [y8, x8], order=0) > 0.5
            gg = g[a:b]; kara = gg >= S + 1
            sel = (w > 0) & kara
            if sel.any():
                ry, rx = y8[sel] * sy + oy - r0, x8[sel] * s + ox - c0          # v2 kıta kutusu
                my_, mx_ = ry * ksy + koy, rx * ksx + kox                       # bizim maske/benzer_z
                biz = ndimage.map_coordinates(z, [my_, mx_], order=1)
                bk = ndimage.map_coordinates(kk, [my_, mx_], order=1) > 0.5
                yedek = ndimage.map_coordinates(kara5, [ry, rx], order=1)
                h = np.maximum(np.where(bk, biz, yedek), 5)
                eski = coz(gg[sel])
                yeni = kodla(w[sel] * h + (1 - w[sel]) * eski)
                blok = out[a:b]; blok[sel] = np.where(w[sel] >= 1e-3, yeni, gg[sel]); out[a:b] = blok
            blok = out[a:b]; blok[uz & (blok > TABAN)] = TABAN; out[a:b] = blok
        ad = "merak/birlesik_" + ("16k" if W > 10000 else "8k")
        Image.fromarray(out.astype(np.uint16)).save(ad + ".png")
        kontrol = out.copy(); pl = (g >= S + 1) | (g > TABAN + 2000)
        xa, xb = int(W * 0.25), int(W * 0.36)
        kontrol[:, xa:xb] = np.where(pl[:, xa:xb], g[:, xa:xb], out[:, xa:xb])  # test platoları geri
        Image.fromarray(kontrol.astype(np.uint16)).save(ad + "_platolu.png")
        print(ad, W, H)
    json.dump(ozet, open("merak/birlesik_olcek.json", "w"), ensure_ascii=False, indent=1)
    print(ozet)


if __name__ == "__main__":
    main()
