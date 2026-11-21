"""Merak'ta denizde benek benek kara çıkmasını önler.

1) 30 pikselden (16k; 8k'da 8) küçük kara parçaları denize çevrilir (Merak dışa aktarımındaki tek piksellik benekler).
2) Deniz seviyesine çok yakın sığ deniz (−40 m'den sığ) −40 m'ye indirilir; Merak'ın ayrıntı gürültüsü bunları karaya çıkarıyordu.
Başka hiçbir piksel değişmez. Kullanım: python3 merak_temizle.py merak/saf merak/son
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from merak_birlestir import S, DD, DP

Image.MAX_IMAGE_PIXELS = None
SIG = -40.0
GRI_SIG = int(np.floor(S * (1 - (abs(SIG) / DD) ** DP)))


def temizle(g, en_az):
    kara = g > S
    L, n = ndimage.label(kara); s = ndimage.sum(kara, L, range(1, n + 1))
    kucuk = np.isin(L, np.nonzero(s < en_az)[0] + 1)
    out = g.copy()
    out[kucuk] = GRI_SIG
    deniz = out <= S
    out[deniz] = np.minimum(out[deniz], GRI_SIG)
    return out, int(kucuk.sum()), int((kucuk.max() and 0) + ((g <= S) & (g > GRI_SIG)).sum()), int((s < en_az).sum())


def main():
    for onek in sys.argv[1:]:
        for r, en_az in (("16k", 30), ("8k", 8)):
            g = np.asarray(Image.open(f"{onek}_{r}.png")).astype(np.int32)
            out, k, d, nk = temizle(g, en_az)
            Image.fromarray(out.astype(np.uint16)).save(f"{onek}_temiz_{r}.png")
            print(onek, r, "silinen benek", nk, "parça,", k, "piksel; sığlaştırılan deniz", d, "piksel")


if __name__ == "__main__":
    main()
