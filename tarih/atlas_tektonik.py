"""Atlas (dunyam-atlas-mobil) tektonik katmanı için yeni hikâyeye göre levha paketi. Atlas deposuna dokunmaz.

Girdi: sahibin Merak ana haritası (16 bit, 16000 × 7719; Merak kodlaması). Çıktı: atlas_tektonik/
  levha_kimlik_8k.png   8000 × 3859, her piksel levha kimliği (0 yok); 16k haritaya göre ×2 ölçek
  levhalar.json         levhalar: kimlik, ad, kapsam, hız (pusula açısı, cm/yıl ve doğu/kuzey bileşeni), sıra, tohum pikseli
  sinirlar.json         levha çiftleri: amaçlanan tür, uzunluk, açıklama
  levhalar.geojson      levha çokgenleri, 16k görüntü pikseli koordinatında (x sağa, y aşağı)
  levha_haritasi.png    önizleme
  HIKAYE.md             atlas için kendi başına okunur hikâye özeti (dönemler, olaylar, levhalar, sınırlar)
Kullanım: python3 atlas_tektonik.py ANA_16K.png
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from matplotlib.path import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from merak_birlestir import coz, S
import merak_levha as ML
import olaylar as O

Image.MAX_IMAGE_PIXELS = None
CIK = "atlas_tektonik"
KAPSAM = {1: "1, 2 (şerit dahil), 4; Köprü Kıstağı; iç denizin batısı; kuzey, batı ve güney sahanlıkları",
          2: "3 ve doğu kıyısının önü, Doğu Hendeği'ne kadar", 3: "5 (Sarı Sıradağları'nın doğu şeridi, Sarı Ön Havzası)",
          4: "7 (Uzun Yarımada)", 5: "6 (iç denizdeki ada)", 6: "iç denizin doğusu ve güneyi (eski Mor Okyanusu kalıntısı)",
          7: "Doğu Hendeği'nin dışındaki okyanus", 8: "batıda, uzak sırtın ötesi", 9: "kuzeyde, uzak sırtın ötesi",
          10: "güneyde, uzak sırtın ötesi"}
SIRA = {1: 10, 2: 6, 3: 5, 4: 4, 5: 7, 6: 3, 7: 2, 8: 2, 9: 2, 10: 2}
ACIKLAMA = {(1, 2): "Mor Sıradağları: 3, 2'nin altına giriyor (Himalaya benzeri, 50 MYÖ'den beri)",
            (1, 3): "Sarı Sıradağları: 5, 4'ün altına giriyor (Zagros benzeri, 8 MYÖ'den beri)",
            (2, 4): "Güneydoğu Kanalı: 22 MYÖ'den beri açılan yarık, kuzeyinde okyanus kabuğu (Kızıldeniz benzeri)",
            (2, 3): "Güneydoğu Kanalı'nın kuzey ucu", (3, 4): "Kanal fayı: 7, 5'e göre kuzeybatıya kayıyor (Ölü Deniz fayı benzeri)",
            (1, 5): "adanın kuzey ve batısı: yay ardı gerilmesi, 6 güneye çekiliyor", (5, 6): "Ada Hendeği: doğu iç deniz tabanı 6'nın altına dalıyor",
            (2, 7): "Doğu Hendeği: Doğu Okyanusu 3'ün altına dalıyor; Doğu Yay Dağları", (1, 7): "2'nin doğu açığı",
            (1, 8): "batı okyanus sırtı (uzak)", (1, 9): "kuzey okyanus sırtı (uzak)", (1, 10): "güney okyanus sırtı (uzak)"}


def main():
    A = np.asarray(Image.open(sys.argv[1])).astype(np.int32)
    H16, W16 = A.shape
    kara16 = A > S
    kara = np.asarray(Image.fromarray((kara16 * 255).astype(np.uint8)).resize((8000, 3859), Image.BOX)) > 127
    E = coz(np.asarray(Image.fromarray(coz(A).astype(np.float32), "F").resize((8000, 3859), Image.BOX)))
    LB = np.load("/tmp/claude-0/-home-user-Tectonic/1cb340b3-df7d-519b-96f4-5170b8cfb37a/scratchpad/LB8.npy")
    esle = {1: 1, 2: 1, 3: 1, 5: 1, 4: 2, 6: 3, 8: 4, 7: 5}
    pl = np.zeros(kara.shape, np.int16)
    for k, v in esle.items():
        pl[(LB == k) & kara] = v
    _, (iy, ix) = ndimage.distance_transform_edt(pl == 0, return_indices=True)
    pl = pl[iy, ix]
    yy, xx = np.mgrid[0:3859, 0:8000]
    havza = np.zeros(kara.shape, bool)
    b = np.array(ML.HAVZA); x0, y0 = b.min(0) - 2; x1, y1 = b.max(0) + 3
    havza[y0:y1, x0:x1] = Path(ML.HAVZA).contains_points(np.c_[xx[y0:y1, x0:x1].ravel(), yy[y0:y1, x0:x1].ravel()]).reshape(y1 - y0, x1 - x0)
    pl[havza & ~kara] = 6
    hx = np.interp(yy, [h[0] for h in ML.HENDEK], [h[1] for h in ML.HENDEK])
    pl[(xx > hx) & ~kara] = 7
    pl[(xx < 3080) & ~kara] = 8
    pl[(yy < 760) & ~kara & (xx <= hx)] = 9
    pl[(yy > 2160) & ~kara & (xx <= hx)] = 10
    os.makedirs(CIK, exist_ok=True)
    Image.fromarray(pl.astype(np.uint8)).save(f"{CIK}/levha_kimlik_8k.png")
    # levhalar
    lev = {}
    for k, (ad, (aci, hiz), renk) in ML.LEV.items():
        m = pl == k; ys, xs = np.nonzero(m)
        ty, tx = int(np.median(ys)), int(np.median(xs))
        if not m[ty, tx]:
            d = ndimage.distance_transform_edt(m); ty, tx = np.unravel_index(d.argmax(), d.shape)
        de, ku = (0.0, 0.0) if aci is None else (round(hiz * np.sin(np.radians(aci)), 2), round(hiz * np.cos(np.radians(aci)), 2))
        lev[ad] = {"kimlik": k, "kapsam": KAPSAM[k], "yon_pusula_derece": aci, "hiz_cm_yil": hiz, "hiz_dogu_kuzey_cm_yil": [de, ku],
                   "sira": SIRA[k], "tohum_16k_px": [int(tx * 2), int(ty * 2)], "renk": list(renk),
                   "kara_payi_cercevede": round(float(kara[m].mean()), 3)}
    json.dump({"cerceve": "Kuzey Levhası sabit; hız cm/yıl; yön pusula açısı (0 kuzey, 90 doğu)",
               "koordinat": "piksel, sahibin 16k Merak haritası (16000 × 7719); levha_kimlik_8k.png ×2",
               "levhalar": lev}, open(f"{CIK}/levhalar.json", "w"), ensure_ascii=False, indent=1)
    # sınırlar
    say = {}
    for u, v in ((pl[:, :-1], pl[:, 1:]), (pl[:-1, :], pl[1:, :])):
        f = u != v; a_, b_ = np.minimum(u[f], v[f]), np.maximum(u[f], v[f])
        for q, n in zip(*np.unique(a_.astype(int) * 100 + b_, return_counts=True)):
            say[(q // 100, q % 100)] = say.get((q // 100, q % 100), 0) + int(n)
    ad = {k: v[0] for k, v in ML.LEV.items()}
    TR = {ML.CARP: "kıta çarpışması (Continental collision)", ML.DAL: "dalma (Subduction)", ML.YAN: "yanal (Transform)", ML.ACIL: "ayrılma (Divergent)"}
    sin = []
    for (a_, b_), n in sorted(say.items(), key=lambda t: -t[1]):
        if n < 20:
            continue
        sin.append({"levhalar": [ad[a_], ad[b_]], "tur": TR.get(ML.KENAR.get((a_, b_)), "belirsiz"),
                    "uzunluk_km_yaklasik": int(n * 5 * 0.79), "aciklama": ACIKLAMA.get((a_, b_), "")})
    json.dump(sin, open(f"{CIK}/sinirlar.json", "w"), ensure_ascii=False, indent=1)
    # çokgenler (16k piksel)
    feats = []
    for k in ML.LEV:
        m = ndimage.binary_opening(pl == k, iterations=1).astype(float)
        cs = plt.contour(np.pad(m, 1), levels=[0.5]); plt.close()
        for seg in cs.allsegs[0]:
            if len(seg) < 20:
                continue
            ring = [[round(float((x - 1 + 0.5) * W16 / 8000), 1), round(float((y - 1 + 0.5) * H16 / 3859), 1)] for x, y in seg[::3]]
            ring.append(ring[0])
            feats.append({"type": "Feature", "properties": {"kimlik": k, "ad": ad[k]}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(f"{CIK}/levhalar.geojson", "w"), ensure_ascii=False)
    # önizleme
    Y0, Y1, X0, X1 = 600, 2300, 2950, 5050
    c = pl[Y0:Y1, X0:X1]; k_ = kara[Y0:Y1, X0:X1]; e = np.where(k_, E[Y0:Y1, X0:X1], 0)
    gy, gx = np.gradient(e); gol = np.clip(0.75 - (gx - gy) / 60, 0.35, 1.2)
    renk = np.array([[0, 0, 0]] + [ML.LEV[k][2] for k in sorted(ML.LEV)], np.float32)[c]
    rgb = np.where(k_[..., None], renk * gol[..., None], renk * 0.45 + 15)
    sinir = (c != np.roll(c, 1, 0)) | (c != np.roll(c, 1, 1))
    rgb[ndimage.binary_dilation(sinir, iterations=2)] = 255
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    fk = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    for a_, v in lev.items():
        x, y = v["tohum_16k_px"][0] / 2 - X0, v["tohum_16k_px"][1] / 2 - Y0
        if 0 < x < X1 - X0 and 0 < y < Y1 - Y0:
            d.text((x, y), f"{v['kimlik']}. {a_}", font=fk, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0), anchor="mm")
    im.thumbnail((1800, 1800)); im.save(f"{CIK}/levha_haritasi.png")
    hikaye(lev, sin)
    print(json.dumps(sin[:8], ensure_ascii=False))


def hikaye(lev, sin):
    t = ["# Dünyam tektonik hikâyesi (yeni sürüm)", "",
         "Bu belge kendi başına okunur; atlasın tektonik katmanını (tektonik/OZET.md, TEKTONIK.md, levhalar.py) eski hikâyeden "
         "bu hikâyeye geçirmek için hazırlandı. Kaynak: Merakproject deposu, tarih/olaylar.py (tek kaynak). Arazi: sahibin Merak ana "
         "haritası (dunyam_ana_16k.png). Koordinatlar o görüntünün pikselidir (16000 × 7719, x sağa, y aşağı); atlas kendi enlem ve "
         "boylamına kendi kaydıyla çevirir.", "",
         "## Eski hikâyeden farkı", "",
         "Eski atlas hikâyesindeki adlar ve olaylar (Duvar, Alt Kıta, İber, Ege, Yarımada, Güneybatı ve batıdan gelen And tipi çarpışma) "
         "geçersizdir. Yeni hikâyede kıta yedi parçadan oluşur ve bugün iki çarpışma etkindir: 3, 2'nin altına (Mor Sıradağları); "
         "5, 4'ün altına (Sarı Sıradağları). 3 milyon yıl önce 1 ile 4 arasındaki boğaz Köprü Kıstağı ile kapandı.", "",
         "## Parçalar", "",
         "1 kuzeybatı (Ulu Dağlar, Ak Kalkan, Mavi Dağlar); 2 kuzey (Çanak Sıraları, Uzak Sıra, Tuzak Çanağı, Kaynak Sırası, Boz Kalkan); "
         "şerit: 2'nin güney kuşağı (Mor Sıradağları); 3 doğu dil (Kızıl Basamaklar, Doğu Yay Dağları); 4 güneybatı (Köprü Dağları, "
         "Köz Yaylası, Köz Masifleri, Sarı Sıradağları); 5 Sarı'nın doğu şeridi (Sarı Ön Havzası); 6 iç denizdeki ada; 7 Uzun Yarımada.", ""]
    t += ["## Dönemler ve olaylar", ""]
    for dn in O.DONEMLER:
        if not dn:
            continue
        a, b = dn["aralik"]
        t += [f"### {dn['ad']} ({a} ile {b} milyon yıl önce)", "", dn["ozet"], ""]
        for z, bas, ac in sorted(dn["olaylar"], key=lambda q: -q[0]):
            t.append(f"- **{z} MYÖ, {bas}:** {ac}")
        t.append("")
    t += ["## Bugünkü levhalar", "", "Kuzey Levhası sabit; yön pusula açısı; hız cm/yıl. Tohum: levhanın içinde bir nokta (16k piksel).", "",
          "| kimlik | levha | kapsam | yön | hız | doğu, kuzey | tohum |", "|---|---|---|---|---|---|---|"]
    for ad, v in lev.items():
        yon = "sabit" if v["yon_pusula_derece"] is None else f"{v['yon_pusula_derece']}°"
        t.append(f"| {v['kimlik']} | {ad} | {v['kapsam']} | {yon} | {v['hiz_cm_yil']} | {v['hiz_dogu_kuzey_cm_yil']} | {v['tohum_16k_px']} |")
    t += ["", "Doğu Dil Levhası bizde dönerek ilerler (doğu ucu 2,4, batı ucu 1,2 cm/yıl, saat yönünün tersine milyon yılda 0,3°); "
          "tabloda ortalaması var. Güney Okyanusu'nun hızı öneridir. Kaçış Bloğu (2'nin güneydoğu köşesi, doğuya 1,2 cm/yıl, Kuzey ile "
          "sol yanal fay) çok küçük olduğu için levha haritasına konmadı.", "",
          "Batı Okyanusu, Kutup ve Güney Okyanusu levhaları (8, 9, 10) yer tutucudur: haritada çerçeveye düz çizgilerle bölündüler ve "
          "gerçek sırtlar kıyıdan çok uzakta. Atlasın kendi okyanus levhaları (tüm gezegeni kaplayanlar) korunabilir; esas olan kıta "
          "levhaları 1 ile 7 ve aralarındaki sınırlardır. Edilgen kıyılarda kıta ile önündeki okyanus aynı levhadır.", "",
          "## Sınırlar (amaçlanan tür)", "", "| levhalar | tür | uzunluk (km, yaklaşık) | açıklama |", "|---|---|---|---|"]
    for s_ in sin:
        t.append(f"| {s_['levhalar'][0]} / {s_['levhalar'][1]} | {s_['tur']} | {s_['uzunluk_km_yaklasik']} | {s_['aciklama']} |")
    t += ["", "Atlasın Bird (2003) sınıflandırması türü göreli hızdan hesaplar; yukarıdaki hızlarla hesaplanan tür bu tablodakinden "
          "ayrılırsa hız değil tablo esas alınmalı ya da fark not edilmeli.", "",
          "## Dosyalar", "", "- `levha_kimlik_8k.png`: 8000 × 3859, piksel değeri levha kimliği (16k haritaya ×2).",
          "- `levhalar.geojson`: levha çokgenleri, 16k piksel.", "- `levhalar.json`, `sinirlar.json`: tablolardaki bilgiler, makine için.",
          "- `levha_haritasi.png`: önizleme.", ""]
    open(f"{CIK}/HIKAYE.md", "w", encoding="utf-8").write("\n".join(t))


if __name__ == "__main__":
    main()
