# AirPods 4 · Ejderha Kılıf (v2)

Kılıfı baştan sona saran büyük bir ejderha. Kapak gövdeye menteşeyle bağlı, sağ yanda çanta halkası için delik var.

![Genel görünüm](onizleme.png)

## v1'den farkı

- **Kapak artık düşmüyor:** Kapak parçası gövdeye iki yandan pimle bağlanıyor. Pimin ekseni AirPods kutusunun kendi menteşesiyle aynı hizada, bu yüzden kutu açılıp kapanırken kılıf zorlanmıyor.
- **Çanta halkası:** Sağ yanda 5,6 mm delikli kulp var. Anahtarlık halkası ya da küçük bir karabina takılır.
- **Ejderha büyüdü:**
  - Kafa kapağın üstünde, sağ omuzdan aşağı bakıyor: açık ağız, dişler, geriye yatık büyük boynuzlar.
  - Gövde iki kat kalın, kılıfı bir buçuk tur sarıyor.
  - Dört pençeli bacak, arkada yarasa kanatları, mızrak uçlu kuyruk.
  - Kılıfın geri kalan her yeri kertenkele derisi gibi pullarla kaplı.

## Dosyalar

| Dosya | Ne |
|---|---|
| `airpods4-ejderha-govde.stl` | Alt parça: USB-C ve ön ışık delikleri, halka kulbu, menteşe kulakları |
| `airpods4-ejderha-kapak.stl` | Üst parça: ejderhanın kafası, menteşe pimleri |
| `onizleme.png` | Kapalı hâli, altı açıdan |
| `onizleme-acik.png` | Kapak 100° açık |
| `onizleme-yakin.png` | Kafa ve menteşe yakın plan (kapak sarı, gövde kırmızı) |
| `ejderha_kilif.py` | Modeli üreten betik |

İki STL de kapalı (watertight) ağdır. Ölçüler mm cinsindendir.

## Takma

1. Gövde parçasını kutunun altına geçir.
2. Kutunun kapağını sonuna kadar aç.
3. Kapak parçasını kutunun kapağına geçir. İki yandaki pimler gövde kulaklarındaki C biçimli ağızdan girip deliğe tık diye oturur.

Kapak kapalıyken pimler ağızdan çıkamaz, kılıf kapakla birlikte açılıp kapanır.

## Baskı ayarları

- **Yön:**
  - Gövdeyi dik bas (altı tablaya).
  - Kapağı ters bas (üstü tablaya gelecek şekilde).
- **Destek:** Gerekli. Ejderha ve kafa dışa taşıyor, ağaç (tree) destek önerilir. Halka kulbunun altı 45° pahlı, oraya destek gerekmez.
- **Katman:** 0,12–0,16 mm (pul, diş ve pençe detayları için).
- **Duvar:** En az 3 çevre (perimeter).
- **Malzeme:** PLA. Gövdede 0,20 mm, kapakta 0,15 mm boşluk bırakıldı.
- **Tavsiye:** Önce kapağı bas ve kutuya oturuşunu dene. Ardından gövdeyi bas, iki parçayı takıp kapağı birkaç kez aç-kapa.

## Ölçüler

- **Kutu:** 46,2 × 50,1 × 21,2 mm (Apple'ın verdiği AirPods 4 kutu ölçüsü).
- **Kılıf dış boyutu:** ejderha, kulp ve boynuzlarla birlikte yaklaşık 67 × 41 × 66 mm.
- **Ağırlık (PLA, dolu):** gövde yaklaşık 20 g, kapak yaklaşık 13 g.

## Tahmine dayanan değerler

Aşağıdaki değerler kutunun fotoğrafından ve Apple ölçülerinden tahmin edildi:
- kapak çizgisinin yüksekliği (`EK_YERI`, alttan 32 mm),
- kutunun menteşe ekseninin yeri (`MENTESE_ICERI` 0,3 mm, `MENTESE_ALTI` 0,5 mm),
- ön ışığın yeri (`LED_Z`).

Pim deliği pimden 0,35 mm bol açıldı, küçük sapmaları tolere eder.

Kapak açılırken zorlanırsa ya da kılıf kapaktan sıyrılırsa, `ejderha_kilif.py` dosyasının başındaki MENTEŞE değerleri düzeltilip model yeniden üretilir:

```
python ejderha_kilif.py
```

Kapak açılınca kapak parçasının arka alt kenarı geriye doğru döner. Bu yüzden gövdenin arkasında, kapak çizgisinin hemen altında yaklaşık 3,5 mm'lik bir menteşe boşluğu var. Bu boşluk bilerek bırakıldı. Betik bu boşluğu, kapağı 0°'den 100°'ye döndürüp çarpışan yerleri oyarak hesaplar.
