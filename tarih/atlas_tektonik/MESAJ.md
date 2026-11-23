Tektonik hikâye değişti; atlastaki tektonik katmanı (tektonik/, TEKTONIK.md, OZET.md) eski hikâyeden bu yeni hikâyeye geçir.
Ekte: HIKAYE.md (kendi başına okunur hikâye: dönemler, olaylar, bugünkü levhalar, sınırlar), levha_kimlik_8k.png (her piksel levha
kimliği), levhalar.json, sinirlar.json, levhalar.geojson ve yeni ana arazi dunyam_ana_16k.png.

1. Arazi: dunyam_ana_16k.png yeni ana harita (Merak 16 bit, 16000 × 7719). Öncekinden tek farkı 1 ile 4 arasındaki boğaza eklenen
   Köprü Kıstağı (3328 piksel, 80 ile 400 m). Her zamanki yolla (worldinfo → _mk_heightmap.py → _egri_kur.py --yaz → kıyı değiştiği
   için _akinti_kur.py) gir. İç denizin batı okyanusuna bağlantısı artık kapalı; akıntıları bu yüzden yeniden kur.
2. Levhalar: koordinatlar dunyam_ana_16k.png pikselidir. _mk_heightmap.py'nin bu görüntü için kullandığı kayıtla enlem ve boylama çevir.
   Kıta levhalarını (1 ile 7) ve sınırlarını levha_kimlik_8k.png'den al (×2 ile 16k piksel); hızlar ve yönler levhalar.json'da
   (Kuzey sabit). Okyanus levhaları 8, 9, 10 yer tutucu; kendi gezegen ölçeğindeki okyanus levhalarını koruyabilirsin.
3. Eski adlar (Duvar, Alt Kıta, İber, Ege, Yarımada, Güneybatı, And tipi batı çarpışması) geçersiz. TEKTONIK.md'ye yeni bir bölüm
   olarak ekle; eski kararları silme, "geçersiz" diye işaretle. OZET.md'yi ozet.py ile yeniden üret.
4. Bird sınıflandırması yeni hızlarla sinirlar.json'daki amaçlanan türden ayrılırsa farkı not et ve bana bildir.
Bitince levha haritasının ve sınır tablosunun özetini gönder.
