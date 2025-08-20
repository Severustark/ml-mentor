# 📘 ML Mentor

**ML Mentor**, makine öğrenmesi konularında etkileşimli bir asistan sağlar.  
PDF ders notlarını ve gömülü bilgileri kullanarak sorularınızı yanıtlar.

---

## 🚀 Özellikler
- 📄 Belirtilen PDF içeriğini yükleyip analiz eder  
- 💬 Doğrudan soru-cevap etkileşimi  
- 🔎 Benzerlik eşik değeri ve `top_k` ayarlanabilir  
- 🧠 PDF + tohum metin desteği (karma bilgi tabanı)  

---

## ⚙️ Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/<kullanıcı-adı>/ml-mentor.git
   cd ml-mentor
   ```

2. Sanal ortam oluşturun ve aktif edin:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux / macOS
   ```

3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Kullanım

1. PDF dosyanızı `data/pdfs/` içine ekleyin.  
   Örn: `data/pdfs/ml_intro.pdf`

2. Uygulamayı çalıştırın:
   ```bash
   python mentor.py
   ```

3. Konsolda sorularınızı yazabilirsiniz:
   ```
   👤 Siz: Makine öğrenmesi nedir?
   🤖 Asistan: Makine öğrenmesi, verilerden öğrenerek tahmin veya karar verebilen algoritmalar bütünüdür...
   ```
   <img width="1671" height="345" alt="bu şekilde " src="https://github.com/user-attachments/assets/608eb319-2706-4456-be9e-38e11fb782f1" />


Çıkmak için `çık` yazmanız yeterlidir.

---

## 📂 Proje Yapısı

```
ml-mentor/
│── mentor.py          # Ana uygulama
│── requirements.txt   # Bağımlılıklar
│── README.md          # Dokümantasyon
└── data/
    └── pdfs/          # PDF dosyaları
```

---

## 💡 Örnek Sorular

- "Makine öğrenmesi nedir?"  
- "Denetimli öğrenme ile denetimsiz öğrenme arasındaki fark nedir?"  
- "Overfitting nasıl önlenir?"  
- "Gradient descent nasıl çalışır?"  
- "Karar ağaçlarının avantajları nelerdir?"  

---

👤 Geliştirici
Ad Soyad: DAMLA ARPA

Bu proje, **Kairu Bootcamp Eğitimleri** kapsamında bir ödev/proje olarak geliştirilmiştir.
