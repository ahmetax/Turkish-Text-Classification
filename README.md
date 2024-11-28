# Turkish-Text-Classification
Türkçe Belge Sınıflandırma Projesi

Bu projede Türkçe Belgeleri genel bir sınıflandırmaya tabi tutmak için bir veriseti oluşturacak ve kendi modelimizi eğiteceğiz.
Gelişti,receğimiz modeli kullanarak, AKTA projesine eklediğimiz belgelerin katagorilerini belirleyeceğiz.
Gelişmeleri https://gurmezin.com adresinden de duyuracağız.

Başlangıç aşamasında kategori başlıklarımız aşağıdaki gibi olacak:

    sağlık
    tarih
    bilim-teknoloji
    din-maneviyat
    edebiyat
    egitim
    ekonomi
    felsefe
    medya
    siyaset
    spor

Metin örneklerini oluşturmak için ilgili kategorilere ilişkin konu listeleri oluşturduk.
Transformers.pipeline ve gpt2 modelinden yararlanan bir betik aracılığıyla, topic listelerinden
ürettiğimiz promptları kullanarak maksimum 400 karakter uzunluğunda metinler hazırladık.
Metinler önce İngilizce oluşturuldu, sonra Türkçeye çevrildi. Çevirilerde deep_translator kütüphanesinin
GoogleTranslator modülünü kullandık.
Her bir kategoriyi eğitmek için 1000'er adet metin dosyası kullanıyoruz. Klasör adları ile kategori 
adları birbirinin aynısıdır.
Eğitim için kullanacağımız metin dosyalarını burada da paylaşacağız.
Metin dosyaları tamamen otomatik olarak hazırlandığı için bazı uyumsuzluklar çıkabilir.
Tespit edildiğinde, bu tür uyumsuzlukları, en kısa zamanda ortadan kaldıracağız.
Gerekli olursa kategori başlıklarında değişiklik yapabilir, yeni kategoriler ekleyebiliriz.

## Yapılacak İşler
- Kategori başlıklarını belirlemek
- Project Gutenberg adresinden indirip Türkçeye çevirdiğimiz belgelerden yararlanarak bir veriseti oluşturmak
- Verisetine ekleyeceğimiz belge içeriklerini seçmek için anahtar kelimelerden yararlanan bir kod geliştirmek
- 
