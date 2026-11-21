"""Elle işleme rehberi: kalan bölgeler (parça 1, 4, 5, 6, 7) için bizim yükseklikten eş yükseklik eğrileri, nokta yükseklikleri
ve yer adları; Merak 8k dışa aktarım çerçevesinde.

Çıktılar: merak/rehber_dunya.png (8000 × 3859, saydam; Merak'ta referans katmanı olarak tam dünyaya oturur),
merak/rehber_kuzeybati.png, merak/rehber_guneybati.png, merak/rehber_ada.png (yakın plan, gölgeli kabartma üstünde).
"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from merak_birlestir import coz
import benzer, olaylar as O

Image.MAX_IMAGE_PIXELS = None
P = "/tmp/claude-0/-home-user-Tectonic/1cb340b3-df7d-519b-96f4-5170b8cfb37a/scratchpad/"
SEVIYE = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
RENK = {100: (120, 200, 120), 250: (180, 220, 100), 500: (240, 230, 90), 750: (245, 190, 70), 1000: (240, 140, 50),
        1500: (225, 80, 60), 2000: (200, 60, 150), 2500: (150, 80, 220), 3000: (255, 255, 255)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
YAKIN = {"kuzeybati": (880, 1500, 3230, 3880), "guneybati": (1350, 2110, 3180, 4180), "ada": (1330, 1620, 3650, 3990)}


def main():
    Z = coz(np.asarray(Image.open("merak/saf_temiz_8k.png")).astype(np.int32)); H, W = Z.shape
    LB = np.load(P + "LB8.npy")
    kara = Z > 0
    _, (iy, ix) = ndimage.distance_transform_edt(LB == 0, return_indices=True); LB = np.where(kara, LB[iy, ix], 0)
    bolge = kara & np.isin(LB, [1, 5, 6, 7, 8])
    Zs = ndimage.gaussian_filter(np.where(kara, Z, 0), 1.2)
    # eğriler
    seviye = np.digitize(Zs, SEVIYE)
    kenar = (seviye != np.roll(seviye, 1, 0)) | (seviye != np.roll(seviye, 1, 1))
    katman = np.zeros((H, W, 4), np.uint8)
    for i, s in enumerate(SEVIYE):
        m = kenar & bolge & (np.maximum(seviye, np.roll(seviye, 1, 0)) == i + 1) | kenar & bolge & (np.maximum(seviye, np.roll(seviye, 1, 1)) == i + 1)
        katman[m, :3] = RENK[s]; katman[m, 3] = 255
    # kıyı çizgisi (bölge)
    kiyi = bolge & ~ndimage.binary_erosion(kara)
    katman[kiyi] = (0, 0, 0, 255)
    # nokta yükseklikleri: yerel en yüksekler
    Zt = ndimage.gaussian_filter(np.where(kara, Z, 0), 2)
    tepe = (Zt == ndimage.maximum_filter(Zt, 45)) & bolge & (Zt > 700)
    ys, xs = np.nonzero(tepe); o = np.argsort(-Zt[ys, xs])[:45]; tepeler = [(int(xs[k]), int(ys[k]), int(round(Z[ys[k], xs[k]], -1))) for k in o]
    # adlar: bizim harita -> Merak 8k
    v = json.load(open("merak/v2_olcek.json")); c0, r0 = v["kita_kutusu_px"][:2]; ksy, ksx, koy, kox = v["kayit"]
    s, sy, ox, oy = json.load(open("merak/birlesik_olcek.json"))["kayit"]
    f = 3808 / O.GEN
    adlar = []
    for ad, (x, y), _ in benzer.etiketler():
        my, mx = y * f, x * f
        ky, kx = (my - koy) / ksy, (mx - kox) / ksx
        ey, ex = (ky + r0 - oy) / sy, (kx + c0 - ox) / s
        if 0 <= ey < H and 0 <= ex < W and bolge[int(ey), int(ex)] or ad in ("Köz Masifleri",):
            adlar.append((ad, int(ex), int(ey)))
    im = Image.fromarray(katman, "RGBA"); d = ImageDraw.Draw(im)
    fk = ImageFont.truetype(FONT, 9); fa = ImageFont.truetype(FONT, 13)
    for x, y, h in tepeler:
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0, 255))
        d.text((x + 3, y - 5), str(h), font=fk, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for ad, x, y in adlar:
        d.text((x, y), ad.upper(), font=fa, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255), anchor="mm")
    im.save("merak/rehber_dunya.png")
    # yakın planlar
    gy, gx = np.gradient(Z); gol = np.clip(0.55 - (gx - gy) / 70, 0.15, 1.0)
    taban = np.where(kara[..., None], (np.stack([gol] * 3, -1) * 200).astype(np.uint8), np.array([25, 35, 60], np.uint8))
    for ad, (y0, y1, x0, x1) in YAKIN.items():
        b = Image.fromarray(taban[y0:y1, x0:x1].astype(np.uint8)).convert("RGBA")
        b.alpha_composite(im.crop((x0, y0, x1, y1)))
        olc = 1800 / max(b.size)
        b = b.resize((int(b.size[0] * olc), int(b.size[1] * olc)), Image.LANCZOS)
        dd = ImageDraw.Draw(b); fl = ImageFont.truetype(FONT, 18)
        for i, sv in enumerate(SEVIYE):
            dd.rectangle([10, 10 + i * 26, 40, 30 + i * 26], fill=RENK[sv] + (255,), outline=(0, 0, 0, 255))
            dd.text((48, 10 + i * 26), f"{sv} m", font=fl, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
        km = 100 * olc / 5
        dd.rectangle([10, b.size[1] - 30, 10 + km, b.size[1] - 22], fill=(255, 255, 255, 255))
        dd.text((14 + km, b.size[1] - 36), "100 km", font=fl, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
        b.convert("RGB").save(f"merak/rehber_{ad}.png")
    print(adlar); print(tepeler[:20])


if __name__ == "__main__":
    main()
