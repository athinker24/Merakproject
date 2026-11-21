"""Sahibin elle katkısı olmadan, yalnız bizim yüksekliklerimiz: Merak dışa aktarım çerçevesinde ve kodlamasında (16 bit).

Kara maskesi ve deniz pikselleri Merak'taki güncel haritadan (birlesik_*) aynen alınır; her kara pikseline bizim yükseklik
(tam çözünürlüklü benzer_z, kıyı farkı olan yerlerde v2 dolgusu) yazılır. Kullanım: python3 merak_saf.py
"""
import numpy as np
from PIL import Image
from merak_birlestir import kodla, S
from merak_duzelt import bizim, H8, W8

Image.MAX_IMAGE_PIXELS = None


def main():
    for r in ("16k", "8k"):
        g = np.asarray(Image.open(f"merak/birlesik_{r}.png")).astype(np.int32); H, W = g.shape
        out = g.copy(); fy, fx = H8 / H, W8 / W
        for a in range(0, H, 1024):
            b = min(H, a + 1024)
            gg = g[a:b]; sel = gg > S
            if not sel.any():
                continue
            Y, X = np.nonzero(sel)
            h = bizim((Y + a + 0.5) * fy - 0.5, (X + 0.5) * fx - 0.5)
            blok = out[a:b]; blok[sel] = kodla(h); out[a:b] = blok
        Image.fromarray(out.astype(np.uint16)).save(f"merak/saf_{r}.png")
        print(r, "kara farkı", int(((out > S) != (g > S)).sum()), "deniz farkı", int((out[g <= S] != g[g <= S]).sum()))


if __name__ == "__main__":
    main()
