# 📡 IoT Saldırı Tespit Sistemi – Büyük Veri ve Yapay Zekâ Tabanlı

Bu proje, IoT (Nesnelerin İnterneti) cihazlarına yönelik olası siber saldırıların tespit edilmesini amaçlayan bir yapay zekâ sistemidir. Büyük veri analitiği ve makine öğrenmesi teknikleri kullanılarak geliştirilmiştir.

---

## 🔍 Amaç

IoT ağlarında oluşabilecek saldırıları tespit etmek için makine öğrenmesi (Random Forest, XGBoost) modelleri ile anomali ve saldırı sınıflandırması yapılmaktadır.

---

## 📁 Proje Yapısı

| Dosya/Dizin Adı | Açıklama |
|------------------|---------|
| `NewDataSetML-EdgeModelTrain.py` | Verisetinin yüklenmesi, işlenmesi ve modellerin eğitilmesi işlemlerini yapar. |
| `NewFeatureRF.py` | Rastgele orman tabanlı özellik seçimi işlemini gerçekleştirir. |
| `PredictIoTAttack.py` | Yeni veriler üzerinde saldırı tahmini yapan modüldür. |
| `AttackTypeCLF.pkl` | Saldırı türü sınıflandırması yapan model dosyası. |
| `label_encoder.pkl` | Etiketleri sayısal forma dönüştüren kodlayıcı. |
| `random_forest_best_model.pkl` | Eğitilmiş en iyi Random Forest modeli. |
| `xgboost_best_model.pkl` | Eğitilmiş en iyi XGBoost modeli. |
| `README.md` | Proje dökümantasyonu. |

---

## ⚙️ Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyaç vardır:

```bash
pip install pandas scikit-learn xgboost joblib


---

Bu yapı:

- Dosya açıklamaları içerir
- Kurulum ve kullanım adımlarını verir
- Modellerin işlevini belirtir
- Geliştirici bilgilerine yer açar

İstersen bu yapının bir de **İngilizce versiyonunu** ya da **proje sunumu için PDF halini** de hazırlayabilirim. Yardımcı olmamı ister misin?
