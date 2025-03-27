#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Metin Kategorilendirme Modeli
Bu script, Türkçe metinleri 12 farklı kategoriye göre sınıflandıran bir model eğitir.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Derin öğrenme için gerekli kütüphaneler (opsiyonel)
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout


def load_dataset(data_dir, categories=None, sample_size=None, test_ratio=0.2):
    """Veri setini yükler ve train/test olarak böler."""
    data = []
    labels = []
    all_categories = []
    
    # Kategori dizinindeki klasörleri tara
    if categories is None:
        categories = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    
    print(f"Yüklenen kategoriler: {categories}")
    all_categories = categories
    
    # Her kategorideki dosyaları yükle
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            print(f"Uyarı: {category_path} dizini bulunamadı, atlanıyor.")
            continue
            
        files = list(Path(category_path).glob("*.txt"))
        
        # Eğer örnekleme isteniyorsa, dosyaların bir kısmını al
        if sample_size is not None and len(files) > sample_size:
            files = random.sample(files, sample_size)
            
        print(f"{category} kategorisinden {len(files)} dosya yükleniyor...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                data.append(text)
                labels.append(category)
            except Exception as e:
                print(f"Hata: {file_path} dosyası okunurken sorun oluştu: {e}")
    
    # Train/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_ratio, random_state=42, stratify=labels
    )
    
    print(f"Eğitim seti: {len(X_train)} örnek")
    print(f"Test seti: {len(X_test)} örnek")
    
    return X_train, X_test, y_train, y_test, all_categories


def train_tfidf_model(X_train, y_train, X_test, y_test, categories, model_type='svm'):
    """TF-IDF ve seçilen modeli kullanarak bir pipeline eğitir."""
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    # TF-IDF Vektörleyici
    tfidf = TfidfVectorizer(
        max_features=50000,  # En sık kullanılan kelimeler
        min_df=5,           # En az 5 dokümanda geçen kelimeler
        max_df=0.8,         # Dokümanların en fazla %80'inde geçen kelimeler
        ngram_range=(1, 2), # Unigram ve bigram
        sublinear_tf=True   # Logaritmik ölçekleme
    )
    
    # Model seçimi
    if model_type == 'svm':
        classifier = LinearSVC(C=1.0, class_weight='balanced')
    elif model_type == 'naive_bayes':
        classifier = MultinomialNB(alpha=0.1)
    elif model_type == 'logistic':
        classifier = LogisticRegression(C=10, class_weight='balanced', max_iter=1000)
    else:
        raise ValueError(f"Desteklenmeyen model türü: {model_type}")
    
    # Pipeline oluşturma
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])
    
    # Modeli eğit
    print(f"{model_type.upper()} modeli eğitiliyor...")
    pipeline.fit(X_train, y_train)
    
    # Test seti üzerinde değerlendir
    y_pred = pipeline.predict(X_test)
    
    # Performans raporunu yazdır
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    # Modeli kaydet
    joblib.dump(pipeline, f'turkce_metin_siniflandirici_{model_type}.joblib')
    print(f"Model 'turkce_metin_siniflandirici_{model_type}.joblib' olarak kaydedildi.")
    
    return pipeline, cm, categories


def plot_confusion_matrix(cm, categories, model_type):
    """Karmaşıklık matrisini görselleştirir."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title(f'Karmaşıklık Matrisi - {model_type.upper()}')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_type}.png')
    print(f"Karmaşıklık matrisi 'confusion_matrix_{model_type}.png' olarak kaydedildi.")


def train_deep_learning_model(X_train, y_train, X_test, y_test, categories, max_words=20000, maxlen=500):
    """LSTM tabanlı derin öğrenme modeli eğitir."""
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("TensorFlow kurulu değil. Derin öğrenme modeli atlanıyor.")
        return None
    
    # Label Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # One-hot encoding
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    # Tokenization
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
    
    # Model tanımla
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories), activation='softmax'))
    
    # Model derleme
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # Early stopping ve model checkpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_accuracy'),
        tf.keras.callbacks.ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Modeli eğit
    print("LSTM modeli eğitiliyor...")
    history = model.fit(
        X_train_pad, y_train_categorical,
        epochs=10,
        batch_size=32,
        validation_data=(X_test_pad, y_test_categorical),
        callbacks=callbacks
    )
    
    # En iyi modeli yükle
    model = tf.keras.models.load_model('best_lstm_model.h5')
    
    # Test seti üzerinde değerlendir
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Performans raporu
    print("\nLSTM Sınıflandırma Raporu:")
    print(classification_report(y_test_encoded, y_pred, target_names=categories))
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Tokenizer'ı kaydet
    with open('tokenizer.pickle', 'wb') as handle:
        import pickle
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Label encoder'ı kaydet
    with open('label_encoder.pickle', 'wb') as handle:
        import pickle
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Tokenizer ve label encoder kaydedildi.")
    
    return model, cm, categories, history


def save_sample_predictions(pipeline, X_test, y_test, categories, n_samples=5):
    """Test setinden rastgele örnekler seçer ve tahminleri gösterir."""
    indices = random.sample(range(len(X_test)), n_samples)
    samples = [X_test[i] for i in indices]
    true_labels = [y_test[i] for i in indices]
    
    # Tahminleri al
    predictions = pipeline.predict(samples)
    
    # Sonuçları bir dosyaya kaydet
    with open('ornek_tahminler.txt', 'w', encoding='utf-8') as f:
        for i, (sample, true_label, pred_label) in enumerate(zip(samples, true_labels, predictions)):
            f.write(f"Örnek {i+1}:\n")
            f.write(f"Metin (ilk 200 karakter): {sample[:200]}...\n")
            f.write(f"Gerçek Kategori: {true_label}\n")
            f.write(f"Tahmin Edilen Kategori: {pred_label}\n")
            f.write("-" * 80 + "\n\n")
    
    print("Örnek tahminler 'ornek_tahminler.txt' dosyasına kaydedildi.")


def main():
    data_dir = "kategoriverisi"  # Veri seti dizini
    
    # Kategorileri tanımla
    categories = ["tarih", "bilim", "teknoloji", "din", "edebiyat", "eğitim", 
                  "ekonomi", "felsefe", "medya", "siyaset", "spor", "kültür-sanat"]
    
    # Her kategoriden kaç örnek alınacak (None=tümü)
    sample_size = None
    
    # Veri setini yükle
    X_train, X_test, y_train, y_test, all_categories = load_dataset(
        data_dir, categories, sample_size=sample_size
    )
    
    # SVM modeli eğit
    svm_pipeline, svm_cm, _ = train_tfidf_model(
        X_train, y_train, X_test, y_test, all_categories, model_type='svm'
    )
    
    # Karmaşıklık matrisini görselleştir
    plot_confusion_matrix(svm_cm, all_categories, 'svm')
    
    # Naive Bayes modeli eğit
    nb_pipeline, nb_cm, _ = train_tfidf_model(
        X_train, y_train, X_test, y_test, all_categories, model_type='naive_bayes'
    )
    
    # Karmaşıklık matrisini görselleştir
    plot_confusion_matrix(nb_cm, all_categories, 'naive_bayes')
    
    # Örnek tahminleri kaydet
    save_sample_predictions(svm_pipeline, X_test, y_test, all_categories)
    
    # İsteğe bağlı: Derin öğrenme modelini eğit
    # train_deep_learning_model(X_train, y_train, X_test, y_test, all_categories)


if __name__ == "__main__":
    main()
