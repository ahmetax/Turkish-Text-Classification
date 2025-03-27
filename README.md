# Türkçe Metin Kategori Sınıflandırıcı

Bu projede, Türkçe metinleri 12 farklı kategoriye göre sınıflandıran makine öğrenmesi modelleri eğitilmiştir. Proje, geleneksel makine öğrenmesi yaklaşımları (TF-IDF + SVM/Naive Bayes) ve modern derin öğrenme yaklaşımları (LSTM ve BERT) için ayrı script'ler içermektedir.

## Kategoriler

Modelimiz aşağıdaki 12 kategoriyi sınıflandırabilmektedir:

- tarih
- bilim
- teknoloji
- din
- edebiyat
- eğitim
- ekonomi
- felsefe
- medya
- siyaset
- spor
- kültür-sanat

Metin örneklerini oluşturmak için Claude 3.7 Sonnet, Grok3 ve Gemma3'ten yararlandık.
Metin örnekleri önce markdown formatında oluşturuldu. Eğitim öncesinde tüm veriyi yalın hale dönüştürdük.
Bunun için markdown_to_plaintext.py betiğini kullandık.

Çalışmalarımızda ağırlığı Geleneksel Makine Öğrenmesi yerine BERT Tabanlı Modele ağırlık verdik.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız olacaktır:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

Derin öğrenme modelleri için:

```bash
pip install tensorflow  # LSTM modeli için
pip install torch transformers  # BERT modeli için
```

## Kullanım

### 1. Geleneksel Makine Öğrenmesi (TF-IDF + SVM/Naive Bayes)

Bu model daha hızlı eğitilir ve daha az kaynak gerektirir. Büyük veri setlerinde bile makul sonuçlar verir.

```bash
python text-categorization-model.py
```

Bu script:
1. Veriyi yükleyecek
2. TF-IDF vektörleme + SVM modeli eğitecek
3. TF-IDF vektörleme + Naive Bayes modeli eğitecek
4. Karmaşıklık matrislerini (confusion matrix) görselleştirecek
5. Modelleri `.joblib` formatında kaydedecek

### 2. BERT Tabanlı Model

Bu model daha iyi sonuçlar verir ancak daha fazla GPU kaynağı ve eğitim süresi gerektirir.

```bash
python bert-model.py
```

Bu script:
1. Veriyi yükleyecek
2. Türkçe BERT modelini (dbmdz/bert-base-turkish-cased) eğitecek
3. En iyi modeli her epoch sonunda kaydedecek
4. Sonuçları ve karmaşıklık matrisini görselleştirecek

### 3. Eğitilmiş Modelle Tahmin Yapma

Eğitilmiş modeli kullanarak yeni metinleri sınıflandırmak için test scriptini kullanabilirsiniz:

```bash
# Bir metin için tahmin yapma
python model-test-script.py --text "Osmanlı İmparatorluğu'nun kuruluş dönemi..." --model turkce_metin_siniflandirici_svm.joblib

# Bir dosyayı sınıflandırma
python model-test-script.py --file ornek.txt --model turkce_metin_siniflandirici_svm.joblib

# Bir dizindeki tüm txt dosyalarını sınıflandırma
python model-test-script.py --dir test_dosyalari/ --model turkce_metin_siniflandirici_svm.joblib
```

BERT modeli ile tahmin:

```bash
python model-test-script.py --text "Örnek metin..." --model turkce_bert_siniflandirici_epoch4 --model-type deep_learning
```

## Performans Değerlendirmesi

Farklı modellerin performansı aşağıdaki metriklerle değerlendirilecektir:

- Doğruluk (Accuracy)
- Kesinlik (Precision)
- Hassasiyet (Recall)
- F1 skoru
- Karmaşıklık matrisi (Confusion matrix)

## Özelleştirme

### 1. Veri Boyutu Ayarlama

Eğer donanımınızda bellek sınırlamaları varsa, veri seti boyutunu azaltabilirsiniz:

```python
# Her kategoriden daha az örnek kullanmak için:
sample_size = 100  # Her kategoriden 100 örnek
```

### 2. BERT Model Parametreleri

BERT modelinin parametrelerini değiştirebilirsiniz:

```python
# BERT model parametreleri
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"  # Türkçe BERT modeli
MAX_LENGTH = 256  # Maksimum token uzunluğu
BATCH_SIZE = 16   # Batch boyutu
EPOCHS = 4        # Eğitim turları
LEARNING_RATE = 2e-5  # Öğrenme oranı
```

### 3. Farklı Modeller Deneme

Farklı BERT modellerini deneyebilirsiniz:

- `dbmdz/bert-base-turkish-uncased`: Küçük harfli Türkçe BERT
- `dbmdz/convbert-base-turkish-cased`: Türkçe ConvBERT
- `xlm-roberta-base`: Çok dilli RoBERTa
- `bert-base-multilingual-cased`: Çok dilli BERT

## İpuçları

1. **Veri Dengesi**: Her kategorinin yaklaşık olarak aynı sayıda örneğe sahip olması, daha dengeli bir model eğitimine yardımcı olur.

2. **Metin Ön-İşleme**: Türkçe metinlerin daha iyi işlenmesi için stemming, lemmatization veya noktalama işaretlerini kaldırma gibi ek ön-işleme adımları ekleyebilirsiniz.

3. **Model Seçimi**: Küçük veri setleri için geleneksel modeller (SVM), büyük veri setleri için derin öğrenme modelleri (BERT) daha iyi performans gösterebilir.

4. **GPU Kullanımı**: BERT modeli eğitimi için GPU kullanılması önerilir. GPU yoksa, daha küçük batch boyutu ve daha az epoch sayısı ile eğitim yapabilirsiniz.

## Bilinen Sorunlar ve Çözümler

1. **Bellek Yetersizliği**: Büyük veri setleri bellek yetersizliğine neden olabilir. `sample_size` değerini düşürerek daha az veri ile çalışabilirsiniz.

2. **Uzun Metinler**: BERT modeli maksimum 512 token işleyebilir. Uzun metinler otomatik olarak kırpılacaktır. Eğer metinleriniz genellikle uzunsa, metin özetleme teknikleri kullanabilirsiniz.

3. **Türkçe Karakterler**: Türkçe karakterlerin düzgün işlendiğinden emin olmak için dosyaların UTF-8 formatında olması önemlidir.

4. **Kategori Dengesizliği**: Bazı kategorilerin diğerlerinden çok daha fazla örneğe sahip olması durumunda `class_weight='balanced'` parametresini kullanabilirsiniz.

## Kodlama

Kodlama çalışmalarını ağırlıklı olarak Claude 3.7 Sonnet ile birlikte yürüttük.
Ürettiğimiz modelin bazı dosyaları 100MB'dan büyük olduğu için github yüklememize izin vermiyor.
Ama bu modeli siz de kolaylıkla üretebilirsiniz.

## Kullandığımız sistem:
Python3.12
## Hardware Information:
- **Memory:**                                      32.0 GiB
- **Processor:**                                   Intel® Core™ i7-7700HQ × 8
- **Graphics:**                                    NVIDIA GeForce GTX 1060 with Max-Q Design
- **Graphics 1:**                                  NVIDIA GeForce GTX 1060 with Max-Q Design
- **Disk Capacity:**                               2.0 TB

## Software Information:
- **Firmware Version:**                            1.17.0
- **OS Name:**                                     Ubuntu 24.04.2 LTS
- **OS Build:**                                    (null)
- **OS Type:**                                     64-bit
- **GNOME Version:**                               46
- **Windowing System:**                            X11
- **Kernel Version:**                              Linux 6.11.0-19-generic

