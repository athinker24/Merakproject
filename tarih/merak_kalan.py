"""Merak'ta elle işlenmemiş kalan bölgeler (batı: parça 1, 4, 5; ada: parça 6) için referanslar, v2 çerçevesinde.

Girdi: merak/v2_kita_kara_16bit.png (merak_v2.py), merak/v2_parca.npy (parça etiketleri v2 kıta kutusunda).
Çıktı: bölge başına renk basamaklı (merak_renk.py paleti) ve kontrastı bölgeye göre gerilmiş gri referans,
dünya çerçevesinde yalnız bu bölgeleri içeren saydam (RGBA) gri heightmap.
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from merak_renk import BASAMAK

BOLGE = {"bati": [1, 5, 6], "ada": [7]}            # v2_parca etiketleri: 1=parça 1, 5=parça 4, 6=parça 5, 7=parça 6


def main():
    o = json.load(open("merak/v2_olcek.json")); c0, r0, ww, hh = o["kita_kutusu_px"]; zmax = o["en_yuksek_m"]
    m = np.asarray(Image.open("merak/v2_kita_kara_16bit.png")).astype(np.float32) / 65535 * zmax
    lab = np.load("merak/v2_parca.npy")
    pal = np.array([[int(r[i:i + 2], 16) for i in (0, 2, 4)] for *_, r in BASAMAK], np.uint8)
    sinir = np.array([b for _, b, *_ in BASAMAK[:-1]])
    g16 = np.asarray(Image.open("merak/v2_yukseklik_16bit.png")).astype(np.float32) / 257
    dunya = np.zeros(g16.shape + (4,), np.uint8)
    ozet = {}
    for ad, ks in BOLGE.items():
        b = np.isin(lab, ks)
        bb = ndimage.binary_dilation(b, iterations=12)
        yy, xx = np.nonzero(bb); y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        mm, bk = m[y0:y1, x0:x1], b[y0:y1, x0:x1]
        kara = mm > 0
        renk = pal[np.searchsorted(sinir, np.where(kara, mm, -1), side="right")]
        renk[~bk & kara] = (renk[~bk & kara] * 0.35).astype(np.uint8)   # bölge dışı kara soluk
        Image.fromarray(renk).resize((renk.shape[1] * 2, renk.shape[0] * 2), Image.NEAREST).save(f"merak/kalan_{ad}_renk.png")
        hi = np.percentile(mm[bk], 99.5)
        gri = np.where(bk, 0.15 + 0.85 * np.sqrt(np.clip(mm / hi, 0, 1)), 0)
        Image.fromarray(np.round(gri * 255).astype(np.uint8)).resize((gri.shape[1] * 2, gri.shape[0] * 2), Image.LANCZOS) \
            .save(f"merak/kalan_{ad}_gri.png")
        Y, X = np.nonzero(b)
        dunya[r0 + Y, c0 + X, :3] = np.round(g16[r0 + Y, c0 + X])[:, None].astype(np.uint8)
        dunya[r0 + Y, c0 + X, 3] = 255
        ppd = 4000 / 180
        ozet[ad] = {"en_yuksek_m": int(mm[bk].max()), "ortanca_m": int(np.median(mm[bk])),
                    "gri_ust_sinir_m": int(hi),
                    "enlem": [round(90 - (r0 + y1) / ppd, 2), round(90 - (r0 + y0) / ppd, 2)],
                    "boylam": [round(-180 + (c0 + x0) / ppd, 2), round(-180 + (c0 + x1) / ppd, 2)]}
    Image.fromarray(dunya, "RGBA").save("merak/kalan_yukseklik_saydam.png")
    json.dump(ozet, open("merak/kalan_olcek.json", "w"), ensure_ascii=False, indent=1)
    print(ozet)


if __name__ == "__main__":
    main()
