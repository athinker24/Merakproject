"""Merak Geography Lab için bugünkü levhaların şeması, Merak dışa aktarım çerçevesinde (kiyi_8k).

Levhalar olaylar.LEVHALAR'dan; yön Merak'taki gibi pusula açısı (0 kuzey, 90 doğu), hız cm/yıl, Kuzey Levhası sabit.
Çıktı: merak/levha_semasi.png
"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from matplotlib.path import Path
from merak_birlestir import coz

Image.MAX_IMAGE_PIXELS = None
Y0, Y1, X0, X1 = 600, 2300, 2950, 5050
# ad, (açı, hız), renk
LEV = {
    1: ("Kuzey Levhası", (None, 0.0), (214, 170, 110)),
    2: ("Doğu Dil Levhası", (0, 1.8), (120, 190, 110)),
    3: ("Orta Blok", (270, 2.8), (200, 120, 200)),
    4: ("Uzun Yarımada Bloğu", (315, 4.6), (240, 230, 120)),
    5: ("Ada Bloğu", (180, 0.7), (240, 150, 70)),
    6: ("Doğu Havza Levhası", (315, 0.8), (90, 160, 210)),
    7: ("Doğu Okyanusu Levhası", (270, 6.5), (70, 90, 190)),
    8: ("Batı Okyanusu Levhası", (270, 2.4), (60, 120, 150)),
    9: ("Kutup Levhası", (0, 0.6), (120, 120, 190)),
    10: ("Güney Okyanusu Levhası", (180, 1.5), (90, 140, 120)),
}
CARP, DAL, YAN, ACIL = "Continental collision", "Subduction", "Transform", "Divergent"
TUR = {CARP: (230, 40, 40), DAL: (255, 120, 0), YAN: (255, 255, 255), ACIL: (255, 225, 0)}
KENAR = {(1, 2): CARP, (1, 3): CARP, (2, 3): ACIL, (3, 4): YAN, (2, 4): ACIL, (1, 5): ACIL, (5, 6): DAL,
         (1, 6): YAN, (2, 6): YAN, (3, 6): YAN, (4, 6): YAN, (2, 7): DAL, (1, 7): YAN, (1, 8): ACIL, (1, 9): ACIL,
         (1, 10): ACIL, (2, 10): ACIL, (3, 10): ACIL, (4, 10): ACIL, (7, 10): YAN, (7, 9): YAN, (8, 9): YAN,
         (8, 10): YAN, (3, 5): ACIL, (4, 7): YAN, (3, 8): ACIL}
HAVZA = [(3960, 1300), (4060, 1300), (4110, 1450), (4050, 1560), (3990, 1600), (3990, 1690), (3870, 1690),
         (3800, 1640), (3720, 1600), (3700, 1580), (3830, 1570), (3950, 1540), (3960, 1420), (3920, 1360)]
HENDEK = [(800, 4800), (1300, 4720), (1480, 4600), (1600, 4520), (1750, 4480), (1900, 4460), (2000, 4400), (2300, 4380)]


def main():
    E = coz(np.asarray(Image.open("merak/kiyi_8k.png")).astype(np.int32))[Y0:Y1, X0:X1]
    LB = np.load("/tmp/claude-0/-home-user-Tectonic/1cb340b3-df7d-519b-96f4-5170b8cfb37a/scratchpad/LB8.npy")[Y0:Y1, X0:X1]
    kara = E > 0
    esle = {1: 1, 2: 1, 3: 1, 5: 1, 4: 2, 6: 3, 8: 4, 7: 5}
    pl = np.zeros(E.shape, np.int16)
    for k, v in esle.items():
        pl[(LB == k) & kara] = v
    _, (iy, ix) = ndimage.distance_transform_edt(pl == 0, return_indices=True)
    pl = np.where(pl == 0, pl[iy, ix], pl)
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    pts = np.c_[xx.ravel(), yy.ravel()]
    havza = Path(HAVZA).contains_points(pts).reshape(E.shape)
    pl[havza & ~kara] = 6
    hx = np.interp(yy, [h[0] for h in HENDEK], [h[1] for h in HENDEK])
    pl[(xx > hx) & ~kara] = 7
    pl[(xx < 3080) & ~kara] = 8
    pl[(yy < 760) & ~kara & (xx <= hx)] = 9
    pl[(yy > 2160) & ~kara & (xx <= hx)] = 10
    # görüntü: gölgeli kabartma + levha rengi
    gy, gx = np.gradient(np.where(kara, E, 0))
    gol = np.clip(0.75 - (gx - gy) / 60, 0.35, 1.2)
    renk = np.array([[0, 0, 0]] + [LEV[k][2] for k in sorted(LEV)], np.float32)[pl]
    rgb = np.where(kara[..., None], renk * gol[..., None], renk * 0.45 + 15)
    # sınırlar
    a = pl[:, :-1]; b = pl[:, 1:]; c = pl[:-1, :]; d = pl[1:, :]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    kenarlar = {}
    for (u, v, ys, xs) in [(a, b, *np.nonzero(a != b)), (c, d, *np.nonzero(c != d))]:
        p1 = u[ys, xs]; p2 = v[ys, xs]
        for y, x, q1, q2 in zip(ys, xs, p1, p2):
            kenarlar.setdefault((min(q1, q2), max(q1, q2)), []).append((x, y))
    for key, pts_ in kenarlar.items():
        r = TUR.get(KENAR.get(key), (255, 0, 255))
        if KENAR.get(key) is None:
            print("tanımsız kenar", key, len(pts_))
        for x, y in pts_:
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=r)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        fk = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        f = fk = ImageFont.load_default()
    # oklar ve adlar
    yer = {1: (3500, 1150), 2: (4250, 1650), 3: (3770, 1700), 4: (4120, 2010), 5: (3800, 1450), 6: (4010, 1420),
           7: (4620, 2050), 8: (3010, 1300), 9: (3900, 690), 10: (3700, 2230)}
    for k, (x, y) in yer.items():
        ad, (aci, hiz), _ = LEV[k]
        x -= X0; y -= Y0
        if aci is not None:
            L = 40 + 22 * hiz
            dx, dy = np.sin(np.radians(aci)) * L, -np.cos(np.radians(aci)) * L
            dr.line([(x, y), (x + dx, y + dy)], fill=(0, 0, 0), width=11)
            dr.line([(x, y), (x + dx, y + dy)], fill=(255, 255, 255), width=6)
            ang = np.arctan2(dy, dx)
            for s in (2.6, -2.6):
                dr.line([(x + dx, y + dy), (x + dx + 22 * np.cos(ang + s), y + dy + 22 * np.sin(ang + s))], fill=(255, 255, 255), width=6)
            yazi = f"{k}. {ad}\n{aci}°  {hiz} cm/yıl"
        else:
            yazi = f"{k}. {ad}\nsabit (0 cm/yıl)"
        dr.multiline_text((x + 12, y + 8), yazi, font=fk, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
    # lejant
    lx, ly = 30, 1440
    dr.rectangle([lx - 15, ly - 15, lx + 520, ly + 235], fill=(15, 20, 35))
    dr.text((lx, ly), "Sınır türü (Merak 'Regime')", font=f, fill=(255, 255, 255))
    for i, (ad, r) in enumerate(TUR.items()):
        dr.rectangle([lx, ly + 45 + i * 45, lx + 50, ly + 70 + i * 45], fill=r)
        dr.text((lx + 65, ly + 42 + i * 45), ad, font=fk, fill=(255, 255, 255))
    im.thumbnail((1800, 1800))
    im.save("merak/levha_semasi.png")
    for k in sorted(LEV):
        print(k, LEV[k][0], int((pl == k).sum()))


if __name__ == "__main__":
    main()
