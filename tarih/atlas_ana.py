"""Sahibin Merak ana haritasını (16 bit, Merak kodlaması) Dünyam Atlası girdisine çevirir; atlas deposuna dokunmaz.

Önceki atlas_cikti.py'den farkı: kıyı artık ATLASIN tuvalinden değil, ana haritadan gelir (Köprü Kıstağı ve sahibin v2
kıyısı). Kayıt: ana haritanın kara maskesi atlas tuvalindeki kıtaya ölçek ve kaymayla (IoU en büyük) oturtulur.
Tuval: yeni kara → kara değeri; yeni deniz ama tuvalde kara → deniz seviyesinin 400 gri altı; ikisi de deniz → tuval aynen.
Eğri doğrusal: değer = deniz + m / max_elev × (65535 − deniz).
Kullanım: python3 atlas_ana.py ANA_16K.png [ATLAS_DEPOSU]
Çıktı: atlas_ana/dunyam_heightmap.png, heightmap_blogu.json, kabartma_kaynak.jpg, kontrol.png, OKU.md
"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from merak_birlestir import coz, S

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maskeden-merak"))
from mk.ciz import renklendir  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CIK = "atlas_ana"
F = 4                                                   # ana haritayı 4'e indirerek çalış (4000 × 1930)


def main():
    ana = sys.argv[1]; ATLAS = sys.argv[2] if len(sys.argv) > 2 else "/home/user/athinker24/dunyam-atlas-mobil"
    os.makedirs(CIK, exist_ok=True)
    w = json.load(open(f"{ATLAS}/dunya/dunyam.json")); hm = w["heightmap"]
    tuval = np.asarray(Image.open(f"{ATLAS}/dunya/dunyam_heightmap.png")).astype(np.float64)
    deniz = float(hm["sea_level_value"]); kara_t = tuval >= deniz
    A = np.asarray(Image.open(ana)).astype(np.int32)
    H, W = A.shape
    m16 = coz(A)
    kara16 = A > S
    # 4'e indirilmiş: kara payı ve yalnız karada ortalama yükseklik
    k4 = kara16.reshape(H // F, F, -1, F)[:, :, :W // F * 1, :] if False else None
    Hc, Wc = H // F * F, W // F * F
    kk = kara16[:Hc, :Wc].reshape(Hc // F, F, Wc // F, F).mean((1, 3))
    mm = np.where(kara16, m16, 0)[:Hc, :Wc].reshape(Hc // F, F, Wc // F, F).sum((1, 3)) / np.maximum(kk * F * F, 1)
    md = np.where(~kara16, m16, 0)[:Hc, :Wc].reshape(Hc // F, F, Wc // F, F).sum((1, 3)) / np.maximum((1 - kk) * F * F, 1)
    maske = kk >= 0.5
    # kayıt: tuval (satır, sütun) → ana/F (y, x) = ((satır − oy) / sy, (sütun − ox) / sx)
    rr, cc = np.nonzero(kara_t); yy, xx = np.nonzero(maske)
    sy = (rr.max() - rr.min()) / (yy.max() - yy.min()); sx = (cc.max() - cc.min()) / (xx.max() - xx.min())
    p0 = np.array([sx, sy, cc.min() - xx.min() * sx, rr.min() - yy.min() * sy])
    r0, r1, c0, c1 = rr.min() - 30, rr.max() + 30, cc.min() - 30, cc.max() + 30
    R, C = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    kt = kara_t[r0:r1, c0:c1]; mf = kk.astype(np.float32)

    def iou_eksi(p):
        a, b, ox, oy = p
        m = ndimage.map_coordinates(mf, [(R - oy) / b, (C - ox) / a], order=1, cval=0) > 0.5
        return -(m & kt).sum() / max((m | kt).sum(), 1)
    son = optimize.minimize(iou_eksi, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
    sx, sy, ox, oy = son.x; iou = -son.fun
    print("kayıt", np.round(son.x, 5), "IoU", round(iou, 4))
    yeni_kara = np.zeros_like(kara_t)
    yeni_kara[r0:r1, c0:c1] = ndimage.map_coordinates(mf, [(R - oy) / sy, (C - ox) / sx], order=1, cval=0) > 0.5
    # küçük benekler (tuvalde 4 pikselden küçük kara) atılır
    L, n = ndimage.label(yeni_kara); s = ndimage.sum(yeni_kara, L, range(1, n + 1)); yeni_kara = np.isin(L, np.nonzero(s >= 4)[0] + 1)
    # yükseklik: tuval pikseli yaklaşık 1/sx ana/F pikseli; örtüşmesin diye o ölçekte yalnız karada süz
    _, (ii, jj) = ndimage.distance_transform_edt(~maske, return_indices=True)
    zk = np.maximum(np.where(maske, mm, mm[ii, jj]), 1.0)
    s_ = 0.45 / sx
    zk = np.maximum(ndimage.gaussian_filter(zk, s_), 1.0)
    h = np.zeros_like(tuval)
    h[r0:r1, c0:c1] = ndimage.map_coordinates(zk, [(R - oy) / sy, (C - ox) / sx], order=1, mode="nearest")
    max_m = float(np.ceil(h[yeni_kara].max() / 10) * 10)
    out = tuval.copy()
    v = deniz + np.clip(h / max_m, 0, 1) * (65535 - deniz)
    out[yeni_kara] = np.maximum(np.round(v[yeni_kara]), deniz + 1)
    batan = kara_t & ~yeni_kara
    out[batan] = deniz - 400
    i = np.unravel_index(np.where(yeni_kara, h, -1).argmax(), h.shape); out[i] = 65535
    out = out.astype(np.uint16)
    Image.fromarray(out).save(f"{CIK}/dunyam_heightmap.png")
    cikan = yeni_kara & ~kara_t
    blok = {k: v for k, v in hm.items() if k != "kabartma"}
    blok["file"] = "dunya/dunyam_heightmap.png"
    blok.update({"max_elev_m": max_m, "elev_gamma": 1.0, "elev_curve": [[0.0, 0.0], [1.0, max_m]]})
    blok["kabartma"] = {k: hm["kabartma"][k] for k in ("file", "sx", "sy", "ox", "oy")} | {"iou": hm["kabartma"].get("iou")}
    json.dump({"heightmap": blok,
               "_not": (f"Sahibin Merak ana haritası (dunyam_ana_16k, 16000 × 7719). Kıyı ana haritadan: tuvalde {int(cikan.sum())} "
                        f"piksel denizden karaya, {int(batan.sum())} piksel karadan denize döndü (ikincisi deniz seviyesinin 400 gri altına). "
                        f"Kayıt: tuval = (ana/4) × (sx {sx:.5f}, sy {sy:.5f}) + (ox {ox:.2f}, oy {oy:.2f}), IoU {iou:.4f}. "
                        "Eğri doğrusal: değer metreyi doğrudan taşır.")},
              open(f"{CIK}/heightmap_blogu.json", "w"), ensure_ascii=False, indent=1)
    # kabartma: atlasın editör görseli çerçevesinde (görsel → tuval: satır = y·sy_k + oy_k, sütun = x·sx_k + ox_k)
    kb = hm["kabartma"]
    ky, kx = Image.open(f"{ATLAS}/{kb['file']}").size[::-1]
    gy, gx = np.mgrid[0:ky, 0:kx].astype(np.float64)
    tr = gy * kb["sy"] + kb["oy"]; tc = gx * kb["sx"] + kb["ox"]
    ay, ax = (tr - oy) / sy, (tc - ox) / sx                        # ana/F koordinatı
    tam = np.where(maske, mm, np.minimum(md, -5))
    olc = kb["sx"] / sx                                            # bir görsel pikseli kaç ana/F pikseli
    tam_s = ndimage.gaussian_filter(tam, 0.45 * olc) if olc > 1 else tam
    zk_k = ndimage.map_coordinates(tam_s, [ay, ax], order=1, mode="nearest")
    kara_k = ndimage.map_coordinates(mf, [ay, ax], order=1, mode="nearest") > 0.5
    zk_k = np.where(kara_k, np.maximum(zk_k, 1), np.minimum(zk_k, -5))
    km_px = 40030.0 / (2 * H) * F * olc
    Image.fromarray(renklendir(zk_k, km_px)).save(f"{CIK}/kabartma_kaynak.jpg", quality=92)
    # kontrol
    oran = np.maximum(out.astype(float) - deniz, 0) / (65535 - deniz)
    m_ac = np.where(out >= deniz, oran * max_m, -1)
    kes = m_ac[r0:r1, c0:c1]
    rgb = np.stack([np.interp(kes, [-1, 0, 500, 1500, 3000, 5000], c) for c in
                    ([30, 90, 170, 200, 160, 245], [50, 140, 170, 160, 130, 245], [90, 80, 100, 110, 110, 245])], -1)
    rgb[cikan[r0:r1, c0:c1]] = [255, 0, 0]; rgb[batan[r0:r1, c0:c1]] = [0, 120, 255]
    Image.fromarray(rgb.astype(np.uint8)).save(f"{CIK}/kontrol.png")
    ak = m_ac[out >= deniz]
    ozet = {"kayit_iou": round(iou, 4), "max_elev_m": max_m, "kara_hucresi": int((out >= deniz).sum()),
            "denizden_karaya": int(cikan.sum()), "karadan_denize": int(batan.sum()),
            "kara_yuksekligi_p10_p50_p90": np.percentile(ak, [10, 50, 90]).round().tolist(), "geri_acilan_en_yuksek": round(float(ak.max()))}
    json.dump(ozet, open(f"{CIK}/ozet.json", "w"), ensure_ascii=False, indent=1)
    print(ozet)


if __name__ == "__main__":
    main()
