# mentor.py
# 🎓 ML Mentor Asistanı (PDF + FAISS + 5+ konu asistanı)
# Çalıştırma:
#   1) .env dosyasına OPENAI_API_KEY ve ML_PDF_PATH yaz
#   2) python mentor.py
# Çıkış: 'çık' / 'exit' / 'quit'

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

from dotenv import load_dotenv

# --------- Ortam ---------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PDF_PATH       = os.getenv("ML_PDF_PATH", "data/pdfs/ml_intro.pdf").strip()
TOP_K          = int(os.getenv("TOP_K", "4"))
SIM_THRESHOLD  = float(os.getenv("SIM_THRESHOLD", "0.30"))
USE_LLM        = os.getenv("USE_LLM", "1") == "1"

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY eksik. Lütfen .env dosyasına ekleyin.")

# --------- LangChain ---------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --------- Tohum Bilgiler (docs) ---------
SEED_DOCS: List[str] = [
    "Makine Öğrenmesi: Bilgisayarların veriden öğrenerek, açıkça programlanmadan tahmin veya karar vermesini sağlayan yapay zeka dalıdır.",
    "Denetimli Öğrenme: Girdi ve çıktı örnekleri ile eğitilen, sınıflandırma ve regresyon problemlerinde kullanılan yöntemdir.",
    "Denetimsiz Öğrenme: Çıkış etiketleri olmadan, verideki gizli yapıları keşfetmeye çalışan yöntemdir (örneğin kümeleme).",
    "Yarı Denetimli Öğrenme: Küçük bir etiketli veri ve büyük miktarda etiketsiz veri kullanılarak eğitilen yöntemdir.",
    "Pekiştirmeli Öğrenme: Ajanın çevresiyle etkileşime girip ödül/ceza mekanizmasıyla öğrenmesini sağlayan yöntemdir.",
    "Eğitim/Test Ayrımı: Eğitim seti öğrenme için, test seti genelleme başarısını ölçmek için kullanılır.",
    "Overfitting: Modelin eğitim verisine aşırı uyum sağlaması ve genelleme yeteneğinin düşmesidir.",
    "Underfitting: Modelin yeterince öğrenememesi; hem eğitim hem testte düşük başarı.",
    "Bias-Variance Dengesi: Düşük bias ve düşük varyans arasındaki denge modelin başarısı için kritiktir.",
    "Cross Validation: Veriyi katmanlara ayırarak modelin farklı parçalarda test edilmesini sağlar.",
    "Feature Engineering: Özellik oluşturma/dönüştürme ile model performansını artırma süreci.",
    "Gradient Descent: Kayıp fonksiyonunu minimize etmek için kullanılan optimizasyon algoritması.",
    "Karar Ağaçları: Sınıflandırma ve regresyonda dallanma kuralları ile karar veren modeller.",
    "k-NN: Komşuluk ilişkisine dayalı sınıflandırma/regresyon algoritması.",
    "SVM: Veriyi ayıran en uygun hiperdüzlemi arayan güçlü sınıflandırma algoritması.",
    "Değerlendirme Metrikleri: Accuracy, Precision, Recall, F1, ROC, AUC vb.",
    "Boyut Azaltma: Yüksek boyutlu veride PCA gibi yöntemlerle daha düşük boyutlu temsil.",
    "Uygulamalar: Görüntü işleme, NLP, öneri sistemleri, sağlık, finans vb."
]

# --------- PDF yükleme ve parçalama ---------
def load_pdf_chunks(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF bulunamadı: {pdf_path}. Sadece tohum metinlerle devam edilecek.")
        return []
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    # Basit bölüm etiketleme (heuristic)
    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata["source"] = os.path.basename(pdf_path)
        t = c.page_content.lower()
        if any(k in t for k in ["giriş", "nedir", "tanım"]):
            c.metadata["section"] = "temeller"
        elif any(k in t for k in ["öğrenme tür", "denetimli", "denetimsiz", "takviyeli"]):
            c.metadata["section"] = "turler"
        elif any(k in t for k in ["özellik", "feature", "veri hazırlama", "ön işleme"]):
            c.metadata["section"] = "veri"
        elif any(k in t for k in ["algoritma", "model", "sınıflandırma", "regresyon"]):
            c.metadata["section"] = "modelleme"
        elif any(k in t for k in ["metrik", "performans", "doğruluk", "precision", "recall", "f1", "roc", "auc", "çapraz doğrulama"]):
            c.metadata["section"] = "degerlendirme"
        elif any(k in t for k in ["optimizasyon", "hiperparametre", "grid", "bayesian"]):
            c.metadata["section"] = "optimizasyon"
        else:
            c.metadata["section"] = "diger"
    return chunks

# --------- Vektör veritabanı (FAISS) ---------
def build_faiss(seed_texts: List[str], pdf_chunks: List[Document]) -> FAISS:
    seed_docs = [Document(page_content=t, metadata={"source": "seed", "section": "temeller"}) for t in seed_texts]
    all_docs = seed_docs + pdf_chunks
    if not all_docs:
        raise RuntimeError("Hiç belge yok: PDF bulunamadı ve seed de boş!")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(all_docs, embeddings)
    return db

# --------- Asistan yapısı ---------
@dataclass
class Assistant:
    name: str
    section_keys: List[str]
    keywords: List[str]

def make_assistants() -> Dict[str, Assistant]:
    return {
        "temeller":      Assistant("Temeller", ["temeller"], ["nedir", "giriş", "tanım", "ai", "ml", "dl", "temel"]),
        "turler":        Assistant("Öğrenme Türleri", ["turler"], ["denetimli", "denetimsiz", "takviyeli", "yarı", "reinforcement"]),
        "veri":          Assistant("Veri & Özellik Mühendisliği", ["veri"], ["özellik", "feature", "ön işleme", "temizleme", "normalizasyon"]),
        "modelleme":     Assistant("Modelleme", ["modelleme"], ["algoritma", "sınıflandırma", "regresyon", "svm", "knn", "ağaç"]),
        "degerlendirme": Assistant("Değerlendirme & Metrikler", ["degerlendirme"], ["doğruluk", "precision", "recall", "f1", "roc", "auc", "cross validation"]),
        "optimizasyon":  Assistant("Optimizasyon", ["optimizasyon"], ["hiperparametre", "grid", "bayesian", "optimizasyon"]),
        "diger":         Assistant("Diğer", ["diger"], []),
    }

def select_assistant(query: str, assistants: Dict[str, Assistant]) -> Assistant:
    q = query.lower()
    best_key, best_score = "temeller", 0
    for key, a in assistants.items():
        score = sum(1 for kw in a.keywords if kw in q)
        if score > best_score:
            best_key, best_score = key, score
    return assistants[best_key]

# --------- Benzerlik skoru (heuristic) ---------
def relevance_score(query: str, text: str) -> float:
    q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    toks = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]
    if not toks:
        return 0.0
    hits = sum(1 for t in q_terms if t in toks)
    return hits / (len(set(toks)) ** 0.5)

# --------- Yanıt üretici (invoke ile) ---------
def answer_query(db: FAISS, query: str, assistant: Assistant) -> Tuple[str, float]:
    # FAISS retriever
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    # Yeni API: get_relevant_documents yerine invoke
    candidates = retriever.invoke(query)  # -> List[Document]
    if not candidates:
        return "", 0.0

    # Asistanın section'ına filtre uygula (yoksa tümü)
    filtered = [d for d in candidates if d.metadata.get("section") in assistant.section_keys] or candidates

    # En alakalıyı seç ve basit eşik uygula
    best = max(filtered, key=lambda d: relevance_score(query, d.page_content))
    score = relevance_score(query, best.page_content)
    if score < SIM_THRESHOLD:
        return "", score

    # Bağlamı oluştur
    context = "\n\n".join(d.page_content for d in filtered)

    # LLM kullanmadan sadece pasaj döndürmek istersen:
    if not USE_LLM:
        snippet = best.page_content.strip()
        return (snippet[:800] + ("..." if len(snippet) > 800 else "")), score

    # LLM ile, SADECE bağlama dayanarak yanıt
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
    system = (
        "Sen bir makine öğrenimi mentörüsün. SADECE verilen bağlamdan yararlan. "
        "Bağlamda olmayan bilgi uydurma; emin değilsen 'belgede yok' de. "
        "Kısa ve öz anlat."
    )
    prompt = f"{system}\n\n[BAĞLAM]\n{context}\n\n[SORU]\n{query}\n\n[CEVAP]"
    out = llm.invoke(prompt).content
    return out, score

# --------- main ---------
def main():
    pdf_chunks = load_pdf_chunks(PDF_PATH)
    db = build_faiss(SEED_DOCS, pdf_chunks)
    assistants = make_assistants()

    print("🎓 ML Mentor Asistanı hazır. ('çık' / 'exit' / 'quit' ile çıkış)")
    print(f"📄 PDF: {PDF_PATH if os.path.exists(PDF_PATH) else 'YOK (sadece seed)'}")
    print(f"🔎 Eşik: {SIM_THRESHOLD} | TOP_K: {TOP_K}\n")

    while True:
        try:
            q = input("👤 Siz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşmek üzere!")
            break

        if q.lower() in ["çık", "exit", "quit"]:
            print("Görüşmek üzere!")
            break

        a = select_assistant(q, assistants)
        ans, sc = answer_query(db, q, a)

        if not ans:
            print(f"🤖 {a.name}: Üzgünüm, belgelerde bu soruya dair yeterli bilgi bulamadım. (skor={sc:.2f})\n")
        else:
            print(f"🤖 {a.name}: {ans}\n   (skor={sc:.2f})\n")

if __name__ == "__main__":
    main()
