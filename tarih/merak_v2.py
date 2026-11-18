"""Sahibin kara maskesi (v2, kare JPEG) üzerine bizim son yüksekliklerimiz: Merak için gri heightmap.

Maske merak/dunya_maske_v2.png ile aynı yerleşimde (kırpılıp bizim enlem aralığımıza ölçeklenir, boylamda ortalanır).
Bizim kara maskesi v2 kırpıntısına benzerlik (ölçek ve kayma) ile oturtulur, örtüşme (IoU) en büyüklenir.
Kara: v2 maskesi. v2'de kara olup bizde deniz olan yerlere en yakın karanın yüksekliği, kıyıya doğru alçalarak verilir.
Deniz: kıyı uzaklığından sahanlık ve yamaç. Kodlama: siyah en derin, %22 gri deniz seviyesi, beyaz en yüksek.
Kullanım: python3 merak_v2.py MASKE.jpg
"""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
import olaylar as O

Image.MAX_IMAGE_PIXELS = None
W, H = 8000, 4000
DENIZ_GRI = 0.22


def temiz(m):
    L, n = ndimage.label(m); s = ndimage.sum(m, L, range(1, n + 1)); m = np.isin(L, np.nonzero(s >= 6)[0] + 1)
    L, n = ndimage.label(~m); s = ndimage.sum(~m, L, range(1, n + 1)); return ~np.isin(L, np.nonzero(s >= 6)[0] + 1)


def main():
    k = np.asarray(Image.open("maske.png").convert("L")) > 127
    z = np.load("cikti/benzer_z.npy").astype(np.float32)
    my, mx = k.shape; der = (O.KM_PX * O.GEN / mx) / 111.2
    ys, xs = np.nonzero(k); lat_ust = 50 - ys.min() * der; lat_alt = 50 - (ys.max() + 1) * der
    a = np.asarray(Image.open(sys.argv[1]).convert("L")) > 127
    m = temiz(a)
    yy, xx = np.nonzero(m); kirp = m[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    ppd = H / 180; hpx = (lat_ust - lat_alt) * ppd; olc = hpx / kirp.shape[0]; wpx = kirp.shape[1] * olc
    hh, ww = int(round(hpx)), int(round(wpx))
    yuk = np.asarray(Image.fromarray((kirp * 255).astype(np.uint8)).resize((ww, hh), Image.LANCZOS)).astype(np.float32)
    v2 = ndimage.gaussian_filter(yuk, 0.6 * olc) > 127          # dunya_maske_v2 ile aynı kıta kırpıntısı
    # bizim maskeyi v2 çerçevesine oturt: v2 pikseli (r, c) -> bizim (r*sy+oy, c*sx+ox)
    f = 4
    v2k = v2[::f, ::f].astype(np.float32); kk = k.astype(np.float32)
    R, C = np.mgrid[0:v2k.shape[0], 0:v2k.shape[1]] * f
    def iou(p):
        sy, sx, oy, ox = p
        s = ndimage.map_coordinates(kk, [R * sy + oy, C * sx + ox], order=1, cval=0) > 0.5
        return -(s & (v2k > 0)).sum() / max((s | (v2k > 0)).sum(), 1)
    p0 = [(ys.max() - ys.min()) / hh, (xs.max() - xs.min()) / ww, ys.min(), xs.min()]
    best = optimize.minimize(iou, p0, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-5, "maxiter": 400,
                             "initial_simplex": [p0, [p0[0] * 1.03, *p0[1:]], [p0[0], p0[1] * 1.03, *p0[2:]],
                                                 [*p0[:2], p0[2] + 20, p0[3]], [*p0[:3], p0[3] + 20]]})
    sy, sx, oy, ox = best.x
    R, C = np.mgrid[0:hh, 0:ww].astype(np.float32)
    bizk = ndimage.map_coordinates(kk, [R * sy + oy, C * sx + ox], order=1, cval=0) > 0.5
    bizz = ndimage.map_coordinates(z, [R * sy + oy, C * sx + ox], order=1, cval=0)
    km = (lat_ust - lat_alt) * 111.2 / hh
    # v2 karası: bizim karada bizim yükseklik; dışında en yakın bizim kara yüksekliği, kıyıya yaklaştıkça alçaltılmış
    _, (iy, ix) = ndimage.distance_transform_edt(~bizk, return_indices=True)
    yakin = bizz[iy, ix]
    kiyi = ndimage.distance_transform_edt(v2) * km                # v2 kıyısına uzaklık (km)
    rampa = np.clip(kiyi / 60, 0, 1)
    kara_h = np.where(bizk, bizz, np.minimum(yakin, 40 + 1500 * rampa) * (0.3 + 0.7 * rampa))
    kara_h = np.where(bizk, np.maximum(kara_h, np.minimum(bizz, 5 + 40 * rampa)), kara_h)
    kara_h = np.maximum(ndimage.gaussian_filter(kara_h, 1.0) * (~bizk) + kara_h * bizk, 1)
    # dünya tuvali
    r0 = int(round((90 - lat_ust) * ppd)); c0 = int(round(W / 2 - ww / 2))
    mk = np.zeros((H, W), bool); mk[r0:r0 + hh, c0:c0 + ww] = v2
    h = np.zeros((H, W), np.float32); h[r0:r0 + hh, c0:c0 + ww] = np.where(v2, kara_h, 0)
    kmx = 360 / W * 111.2 * np.cos(np.radians(20)); kmy = 180 / H * 111.2
    d = ndimage.distance_transform_edt(~mk, sampling=(kmy, kmx))
    gur = ndimage.gaussian_filter(np.random.default_rng(7).standard_normal((H, W)).astype(np.float32), 120 / kmy)
    d = d * (1 + 0.25 * np.clip(gur / (gur.std() + 1e-6), -1.5, 1.5))
    t = np.clip((d - 40) / 360, 0, 1); t = t * t * (3 - 2 * t)
    h = np.where(mk, h, -80 - 3920 * t)
    zmax, zmin = float(h.max()), float(h.min())
    g = np.where(h >= 0, DENIZ_GRI + (1 - DENIZ_GRI) * h / zmax, DENIZ_GRI * (1 - h / zmin))
    Image.fromarray(np.round(g * 65535).astype(np.uint16)).save("merak/v2_yukseklik_16bit.png")
    Image.fromarray(np.round(g * 255).astype(np.uint8)).save("merak/v2_yukseklik_8bit.png")
    gk = g[r0:r0 + hh, c0:c0 + ww]
    Image.fromarray(np.round(gk * 65535).astype(np.uint16)).save("merak/v2_kita_yukseklik_16bit.png")
    sadece = np.where(v2, kara_h / zmax, 0)                       # yalnız kara, deniz siyah
    Image.fromarray(np.round(sadece * 65535).astype(np.uint16)).save("merak/v2_kita_kara_16bit.png")
    json.dump({"boyut": [W, H], "kita_kutusu_px": [c0, r0, ww, hh], "enlem": [round(lat_alt, 3), round(lat_ust, 3)],
               "boylam": [round(-180 + c0 / ppd, 3), round(-180 + (c0 + ww) / ppd, 3)],
               "kayit_iou": round(-best.fun, 4), "kayit": [round(float(v), 4) for v in best.x],
               "kodlama": "gri = 0,22 + 0,78 × m / %.0f (kara); gri = 0,22 × (1 − m / %.0f) (deniz)" % (zmax, zmin),
               "kara_yalniz": "v2_kita_kara_16bit: gri = m / %.0f, deniz 0" % zmax,
               "en_yuksek_m": round(zmax), "en_derin_m": round(zmin),
               "maske_farki_px": int((v2 != bizk).sum())}, open("merak/v2_olcek.json", "w"), ensure_ascii=False, indent=1)
    on = Image.fromarray(np.round(gk * 255).astype(np.uint8)); on.thumbnail((1600, 1600)); on.save("merak/v2_onizleme.jpg", quality=88)
    print(open("merak/v2_olcek.json").read())


if __name__ == "__main__":
    main()
