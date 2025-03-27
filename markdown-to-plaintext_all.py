#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown dosyalarını yalın metin formatına dönüştüren script.
Bu script, giriş dizinindeki kategori yapısını koruyarak çıkış dizinine aynı yapıyı oluşturur.

Kullanım: python markdown_to_plaintext.py <input_directory> <output_directory>
Örnek: python markdown_to_plaintext.py categories kategoridizisi
"""

import os
import re
import sys
import glob
from pathlib import Path

def markdown_to_plaintext(markdown_text):
    """Markdown metnini yalın metne dönüştürür."""
    # Markdown başlık formatlarını kaldır (# İşaretlerini)
    text = re.sub(r'^#+\s+', '', markdown_text, flags=re.MULTILINE)
    
    # Kalın ve italik formatları kaldır
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **kalın** -> kalın
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italik* -> italik
    text = re.sub(r'__(.*?)__', r'\1', text)      # __kalın__ -> kalın
    text = re.sub(r'_(.*?)_', r'\1', text)        # _italik_ -> italik
    
    # Karmaşık madde işaretlerini basitleştir
    # - Yıldız, artı işaretleri vs. tire işaretine dönüştür
    text = re.sub(r'^\s*[*+]\s+', '- ', text, flags=re.MULTILINE)
    
    # Numaralı listeleri düzenle (opsiyonel)
    # text = re.sub(r'^\s*\d+\.\s+', '- ', text, flags=re.MULTILINE)
    
    # Bağlantı formatlarını kaldır [metin](url) -> metin
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Inline kod formatlarını kaldır `kod` -> kod
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Kod bloklarını kaldır
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Yatay çizgileri kaldır
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Fazla boş satırları tek boş satıra indir
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_directory(input_dir, output_dir):
    """Belirtilen dizindeki tüm markdown dosyalarını işler ve kategori yapısını korur."""
    # Çıktı dizinini oluştur (yoksa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Markdown dosyalarını bul
    markdown_files = []
    for extension in ['*.md', '*.markdown', '*.txt']:
        pattern = os.path.join(input_dir, '**', extension)
        markdown_files.extend(glob.glob(pattern, recursive=True))
    
    if not markdown_files:
        print(f"UYARI: {input_dir} içinde markdown dosyası bulunamadı.")
        return 0
    
    categories = set()
    file_count = 0
    for md_file in markdown_files:
        # Giriş ve çıkış dosya yollarını belirle
        rel_path = os.path.relpath(md_file, input_dir)
        out_file = os.path.join(output_dir, rel_path)
        
        # Kategori bilgisini al (ilk alt dizin)
        parts = Path(rel_path).parts
        if len(parts) > 1:
            categories.add(parts[0])
        
        # Çıkış dosyasının dizini oluştur (yoksa)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        # Dosyayı işle
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            plain_content = markdown_to_plaintext(markdown_content)
            
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(plain_content)
            
            file_count += 1
            if file_count % 100 == 0:
                print(f"{file_count} dosya işlendi...")
                
        except Exception as e:
            print(f"Hata: {md_file} dosyası işlenirken bir sorun oluştu: {e}")
    
    if categories:
        print(f"\nİşlenen kategoriler: {', '.join(sorted(categories))}")
    
    return file_count

def main():
    # if len(sys.argv) != 3:
    #     print("Kullanım: python markdown_to_plaintext.py <input_directory> <output_directory>")
    #     print("Örnek: python markdown_to_plaintext.py categories kategoridizisi")
    #     sys.exit(1)
    
    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    
    # if not os.path.isdir(input_dir):
    #     print(f"Hata: {input_dir} geçerli bir dizin değil.")
    #     sys.exit(1)
    
    input_dir = "categories/"
    output_dir = "kategoriverisi/"
    
    print(f"Markdown dosyaları {input_dir} dizininden okunuyor...")
    print(f"Yalın metin dosyaları {output_dir} dizinine yazılacak, kategori yapısı korunacak...")
    
    file_count = process_directory(input_dir, output_dir)
    
    print(f"\nİşlem tamamlandı. Toplam {file_count} dosya işlendi.")

if __name__ == "__main__":
    main()
