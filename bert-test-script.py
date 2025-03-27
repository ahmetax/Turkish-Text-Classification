#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Model Test Scripti
Bu script, eğitilmiş BERT modelini kullanarak yeni metinleri sınıflandırır.
"""

import argparse
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 256

def load_bert_model(model_path):
    """Eğitilmiş BERT modelini ve gerekli bileşenlerini yükler."""
    try:
        # Model dizini kontrolü
        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_dir():
            print(f"Hata: '{model_path}' geçerli bir model dizini değil.")
            sys.exit(1)
            
        # Modeli yükle
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Tokenizer'ı yükle
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Kategori sözlüğünü yükle
        label_dict_path = model_path / "label_dict.json"
        if not label_dict_path.exists():
            print(f"Uyarı: '{label_dict_path}' dosyası bulunamadı. Kategori bilgileri eksik olabilir.")
            # Kategori bilgisi yoksa varsayılan oluştur
            label_dict = {}
            for i in range(model.config.num_labels):
                label_dict[f"Kategori_{i}"] = i
        else:
            with open(label_dict_path, 'r', encoding='utf-8') as f:
                label_dict = json.load(f)
                
        # Label dict'i ters çevir (id -> label)
        id2label = {v: k for k, v in label_dict.items()}
        
        print(f"BERT modeli başarıyla yüklendi: {model_path}")
        print(f"Kategoriler: {', '.join(label_dict.keys())}")
        
        return model, tokenizer, label_dict, id2label
        
    except Exception as e:
        print(f"Hata: BERT modeli yüklenirken sorun oluştu: {e}")
        sys.exit(1)

def predict_text_with_bert(model, tokenizer, id2label, text, device=None):
    """BERT modeli ile metin sınıflandırma."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
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
    
    # Tüm kategorilerin olasılıklarını al
    all_probs = probabilities[0].cpu().numpy()
    category_probs = [(id2label[i], float(prob)) for i, prob in enumerate(all_probs)]
    category_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Sınıf indeksini kategori ismine dönüştür
    prediction = id2label[pred_class_idx]
    confidence = probabilities[0][pred_class_idx].item()
    
    return prediction, confidence, category_probs

def classify_file(model, tokenizer, id2label, file_path, device=None):
    """Bir dosyadaki metni BERT ile sınıflandırır."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if len(text) <= 1000:
                print(f"Dosya: {file_path} çok küçük.")
                return None, None
            # Çok uzun metinleri kısalt
            if len(text) > 11000:
                text = text[1000:11000] # ilk 1000 karakteri atla
        
        prediction, confidence, all_categories = predict_text_with_bert(
            model, tokenizer, id2label, text, device
        )
        
        print(f"Dosya: {file_path}")
        print(f"Tahmin edilen kategori: {prediction} (güven: {confidence:.4f})")
        print("En olası 3 kategori:")
        for category, prob in all_categories[:3]:
            print(f"  - {category}: {prob:.4f}")
        
        return prediction, all_categories
        
    except Exception as e:
        print(f"Hata: {file_path} dosyası işlenirken sorun oluştu: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='BERT Türkçe metin sınıflandırıcı')
    parser.add_argument('--model', type=str, required=True,
                        help='BERT model dizini (turkce_bert_siniflandirici_epochX)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Sınıflandırılacak metin')
    group.add_argument('--file', type=str, help='Sınıflandırılacak metin dosyası')
    group.add_argument('--dir', type=str, help='Sınıflandırılacak metin dosyalarının bulunduğu dizin')
    
    parser.add_argument('--top', type=int, default=3, 
                        help='Gösterilecek en olası kategori sayısı (varsayılan: 3)')
    
    args = parser.parse_args()
    
    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model yükleme
    model, tokenizer, label_dict, id2label = load_bert_model(args.model)
    
    # Sınıflandırma işlemi
    if args.text:
        prediction, confidence, all_categories = predict_text_with_bert(
            model, tokenizer, id2label, args.text, device
        )
        
        print(f"Metin: {args.text[:100]}...")
        print(f"Tahmin edilen kategori: {prediction} (güven: {confidence:.4f})")
        print(f"En olası {args.top} kategori:")
        for category, prob in all_categories[:args.top]:
            print(f"  - {category}: {prob:.4f}")
    
    elif args.file:
        classify_file(model, tokenizer, id2label, args.file, device)
    
    elif args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"Hata: {args.dir} geçerli bir dizin değil.")
            return
        
        # Tüm txt dosyalarını bul
        files = list(directory.glob("**/*.txt"))
        if not files:
            print(f"Uyarı: {args.dir} dizininde hiç .txt dosyası bulunamadı.")
            return
        
        print(f"Toplam {len(files)} dosya işlenecek...")
        
        # Her dosyayı sınıflandır
        results = {}
        for file_path in files:
            prediction, _ = classify_file(model, tokenizer, id2label, file_path, device)
            if prediction:
                results[prediction] = results.get(prediction, 0) + 1
        
        # Sonuçları göster
        print("\nSonuçlar:")
        for category, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{category}: {count} dosya ({count/len(files)*100:.1f}%)")

if __name__ == "__main__":
    main()
