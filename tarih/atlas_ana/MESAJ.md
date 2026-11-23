Arazi değişti: sahibin Merak ana haritasının bozkır düzeltmeli son sürümü (Köprü Kıstağı dahil, kıyı v2; Kaynak Sırası'nın güney ucunda iç deniz kıyısı yükseltildi). Ekteki dosyalar atlas girdisi, kayıt yapıldı.
1. dunyam_heightmap.png → dunya/dunyam_heightmap.png (4096 × 2048, 16 bit). Kıyı artık ana haritadan: tuvalde 1874 piksel denizden
   karaya, 20110 piksel karadan denize döndü (çoğu kıyı boyunca bir piksellik şerit; ikincisi deniz seviyesinin 400 gri altına kondu).
   Okyanus pikselleri aynen.
2. heightmap_blogu.json → dunya/dunyam.json içindeki heightmap bloğunda yalnız max_elev_m (5490), elev_gamma (1.0) ve elev_curve
   (doğrusal) alanlarını değiştir; diğer alanlar aynı.
3. kabartma_kaynak.jpg → dunya/kabartma_kaynak.jpg (aynı çerçeve, aynı sx sy ox oy).
4. Kıyı değiştiği için akıntıları yeniden kur (_akinti_kur.py), sonra uret.py; su/, bitki, toprak ve diğer katmanlar araziye bağlıysa
   onları da yeniden üret. kontrol.png'de kırmızı denizden karaya, mavi karadan denize dönen pikseller.
Bitince kara payını, iklim bölgelerinin ve bozkırın değişimini sayılarla bildir.
