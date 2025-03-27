#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe BERT Tabanlı Metin Sınıflandırıcı
Bu script, Türkçe metinleri kategorilerine göre sınıflandırmak için
BERTurk veya multilingual BERT kullanan bir model eğitir.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# BERT modeli seçenekleri:
# - dbmdz/bert-base-turkish-cased (Türkçe BERT)
# - dbmdz/bert-base-turkish-uncased (Türkçe BERT, küçük harfli)
# - dbmdz/convbert-base-turkish-cased (Türkçe ConvBERT)
# - xlm-roberta-base (Çok dilli RoBERTa)
# - bert-base-multilingual-cased (Çok dilli BERT)

# BERTurk ekibi tarafından eğitilmiş Türkçe BERT modelidir
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 256  # Maksimum token uzunluğu
BATCH_SIZE = 16   # Batch boyutu
EPOCHS = 4        # Eğitim turları
LEARNING_RATE = 2e-5  # Öğrenme oranı


class TextDataset(Dataset):
    """BERT için metin veri seti sınıfı."""
    
    def __init__(self, texts, labels, label_dict, tokenizer, max_length):
        self.texts = texts
        self.labels = [label_dict[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(data_dir, categories=None, sample_size=None, test_ratio=0.2):
    """Veri setini yükler ve train/test olarak böler."""
    data = []
    labels = []
    
    # Kategori dizinindeki klasörleri tara
    if categories is None:
        categories = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    
    print(f"Yüklenen kategoriler: {categories}")
    
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
                    # Uzun metinleri kısalt (BERT için)
                    if len(text) > 10000:
                        text = text[:10000]
                data.append(text)
                labels.append(category)
            except Exception as e:
                print(f"Hata: {file_path} dosyası okunurken sorun oluştu: {e}")
    
    # Verileri karıştır
    indices = list(range(len(data)))
    random.shuffle(indices)
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Train/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_ratio, random_state=42, stratify=labels
    )
    
    print(f"Eğitim seti: {len(X_train)} örnek")
    print(f"Test seti: {len(X_test)} örnek")
    
    return X_train, X_test, y_train, y_test, categories


def train_bert_model(X_train, y_train, X_test, y_test, categories, 
                     model_name=BERT_MODEL_NAME, max_length=MAX_LENGTH, 
                     batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE):
    """BERT tabanlı bir sınıflandırıcı modeli eğitir."""
    
    # GPU kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Tokenizer ve model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Kategori sözlüğü oluştur
    label_dict = {category: idx for idx, category in enumerate(categories)}
    
    # Veri setlerini hazırla
    train_dataset = TextDataset(X_train, y_train, label_dict, tokenizer, max_length)
    test_dataset = TextDataset(X_test, y_test, label_dict, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(categories)
    )
    
    model.to(device)
    
    # Optimizer ve scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Toplam adım sayısı
    total_steps = len(train_dataloader) * epochs
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Eğitim fonksiyonu
    def train_epoch(model, dataloader, optimizer, scheduler, device):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Eğitim", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    # Değerlendirme fonksiyonu
    def evaluate(model, dataloader, device):
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Değerlendirme", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        return all_preds, all_labels
    
    # Eğitim döngüsü
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Eğitim
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Eğitim kaybı: {train_loss:.4f}")
        
        # Değerlendirme
        y_pred, y_true = evaluate(model, test_dataloader, device)
        
        # Sonuçları sınıf isimlerine dönüştür
        y_pred_labels = [categories[pred] for pred in y_pred]
        y_true_labels = [categories[true] for true in y_true]
        
        # Performans değerlendirmesi
        accuracy = sum(1 for pred, true in zip(y_pred, y_true) if pred == true) / len(y_true)
        print(f"Doğruluk: {accuracy:.4f}")
        
        # En iyi modeli kaydet
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            # Modeli kaydet
            model_path = f"turkce_bert_siniflandirici_epoch{epoch+1}"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Label sözlüğünü kaydet
            import json
            with open(f"{model_path}/label_dict.json", 'w') as f:
                json.dump(label_dict, f)
            
            print(f"En iyi model kaydedildi: {model_path}")
    
    # Son değerlendirme
    y_pred, y_true = evaluate(model, test_dataloader, device)
    
    # Sonuçları sınıf isimlerine dönüştür
    y_pred_labels = [categories[pred] for pred in y_pred]
    y_true_labels = [categories[true] for true in y_true]
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories)
    print(report)
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=categories)
    
    # Sonuçları kaydet
    with open('bert_siniflandirici_sonuclar.txt', 'w', encoding='utf-8') as f:
        f.write("BERT Sınıflandırıcı Sonuçları\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Doğruluk: {best_accuracy:.4f}\n\n")
        f.write("Sınıflandırma Raporu:\n")
        f.write(report)
    
    print("Sonuçlar 'bert_siniflandirici_sonuclar.txt' dosyasına kaydedildi.")
    
    return model, tokenizer, label_dict, cm, categories


def plot_confusion_matrix(cm, categories, filename='bert_confusion_matrix.png'):
    """Karmaşıklık matrisini görselleştirir."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('BERT Sınıflandırıcı Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Karmaşıklık matrisi '{filename}' olarak kaydedildi.")


def bert_predict_text(model, tokenizer, label_dict, text, device=None):
    """BERT modeli ile metin sınıflandırma."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Kategori indeksleri ile kategori isimleri arasında eşleştirme yap
    id2label = {v: k for k, v in label_dict.items()}
    
    # Metni tokenize et
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Tensörleri cihaza taşı
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Değerlendirme modu
    model.eval()
    
    # Tahmini yap
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Sınıf olasılıklarını hesapla
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # En yüksek olasılıklı sınıfı seç
    _, preds = torch.max(probabilities, dim=1)
    pred_class_idx = preds.item()
    
    # Sınıf indeksini kategori ismine dönüştür
    prediction = id2label[pred_class_idx]
    confidence = probabilities[0][pred_class_idx].item()
    
    return prediction, confidence


def main():
    data_dir = "kategoriverisi"  # Veri seti dizini
    
    # Kategorileri tanımla
    categories = ["tarih", "bilim", "teknoloji", "din", "edebiyat", "eğitim", 
                  "ekonomi", "felsefe", "medya", "siyaset", "spor", "kültür-sanat"]
    
    # Her kategoriden kaç örnek alınacak (None=tümü)
    sample_size = None  # Tüm veriyi kullan
    
    # GPU belleği sınırlı ise her kategoriden daha az örnek kullanılabilir
    # sample_size = 100  # Her kategoriden 100 örnek
    
    # Veri setini yükle
    X_train, X_test, y_train, y_test, all_categories = load_dataset(
        data_dir, categories, sample_size=sample_size
    )
    
    # BERT modelini eğit
    model, tokenizer, label_dict, cm, categories = train_bert_model(
        X_train, y_train, X_test, y_test, all_categories,
        model_name=BERT_MODEL_NAME,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Karmaşıklık matrisini görselleştir
    plot_confusion_matrix(cm, categories)
    
    # Örnek tahmin
    test_texts = [
        "Osmanlı İmparatorluğu'nun kuruluş döneminde önemli rol oynayan kişiler ve olaylar",
        "Küresel iklim değişikliğinin deniz ekosistemi üzerindeki etkileri ve çözüm önerileri",
        "Yapay zeka algoritmalarının günlük hayatımızdaki uygulamaları ve geleceğe dair öngörüler",
        "İslam felsefesinin Batı düşüncesi üzerindeki etkileri ve kültürlerarası etkileşimler"
    ]
    
    print("\nÖrnek Tahminler:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for i, text in enumerate(test_texts):
        prediction, confidence = bert_predict_text(model, tokenizer, label_dict, text, device)
        print(f"Örnek {i+1}:")
        print(f"Metin: {text[:100]}...")
        print(f"Tahmin: {prediction} (güven: {confidence:.4f})")
        print("-" * 50)


if __name__ == "__main__":
    main()