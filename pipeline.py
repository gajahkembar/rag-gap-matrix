"""Pipeline RAG enam tahap untuk menghasilkan Literature Gap Matrix.

Tahap: (1) document processing -> (2) chunking -> (3) embedding lokal ->
(4) vector store ChromaDB -> (5) retrieval -> (6) augmented generation (Groq).
Mode --mock memakai embedding hash dan LLM tiruan untuk menguji alur tanpa jaringan.
"""
import csv, hashlib, json, os, random, re, time
from pathlib import Path

import pandas as pd

# Kolom matriks (sesuai ppt5) dan kueri retrieval (Inggris, karena artikel berbahasa Inggris).
FIELDS = {
    "Problem": "problem statement, motivation, research gap, challenge the paper addresses",
    "Metode": "proposed method, approach, architecture, framework, models and techniques used",
    "Dataset": "dataset, benchmark, corpus, data used in the experiments, number of samples",
    "Temuan": "main results, findings, performance, improvement over baselines, conclusion",
    "Keterbatasan": "limitations, weaknesses, threats to validity, what the approach cannot do",
    "Future work": "future work, future research directions, we plan to, open challenges",
}
NOT_FOUND = "Tidak ditemukan"

EXTRACT_SYSTEM = (
    "Kamu mengekstrak informasi dari artikel ilmiah untuk satu sel matriks literatur. "
    "Jawab HANYA berdasarkan KONTEKS yang diberikan; jangan menambah dari pengetahuan sendiri. "
    f"Jika informasi yang diminta tidak ada di konteks, jawab persis satu frasa: {NOT_FOUND}. "
    "Format jawaban: teks polos bahasa Indonesia (terjemahkan kalimat berbahasa Inggris), 1-3 kalimat, "
    "maksimal 60 kata, tanpa markdown, tanpa awalan seperti 'Problem:' atau 'Answer:', tanpa justifikasi. "
    "Pertahankan istilah teknis dan angka persis seperti di konteks; jangan menulis angka yang terpotong. "
    "Untuk 'Keterbatasan' dan 'Future work', ambil hanya yang dinyatakan penulis tentang penelitian MEREKA SENDIRI, "
    "bukan kelemahan metode lain atau masalah yang sudah ada sebelumnya. Untuk 'Problem', tulis masalah penelitian "
    "dalam bahasa biasa, bukan notasi matematis. Jika artikel adalah survei/tinjauan pustaka, katakan itu survei "
    "dan ringkas cakupannya (jangan menulis detail metode dari studi yang ditinjau seolah-olah milik artikel). "
    "Akhiri dengan nomor potongan sumber, mis. [2][4]."
)

GAP_SYSTEM = (
    "Kamu membantu menulis draf analisis research gap dari matriks literatur. Gunakan HANYA isi matriks "
    "yang diberikan dan rujuk paper dengan kode [P01] dst. Pada korpus ini hanya ada beberapa paper, jadi "
    "jangan mengklaim 'belum ada penelitian'; gunakan frasa 'pada korpus paper ini'. Jangan menyebut sebuah "
    "gap jika ada paper di matriks yang sudah mengerjakannya. Tulis dalam bahasa Indonesia, paragraf mengalir, "
    "tanpa tabel, tanpa daftar, tanpa cetak tebal. Usulan harus realistis untuk proyek kuliah 3 orang "
    "(satu pertanyaan penelitian, eksperimen pada dataset yang sudah ada, tanpa membuat dataset baru)."
)

_LABEL = re.compile(r"^\s*(\*\*)?(answer|jawaban|problem|masalah|metode|dataset|temuan|keterbatasan|future work)(\*\*)?\s*[:：]\s*", re.I)


def clean_answer(text: str) -> str:
    """Rapikan keluaran LLM: buang label, markdown, kebocoran 'Justification', dan penanda sumber di sel."""
    t = text.strip()
    t = re.split(r"\n\s*(justification|justifikasi|alasan)\s*[:：]", t, flags=re.I)[0].strip()
    t = _LABEL.sub("", t).replace("**", "").replace("\\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^Future work (should|will|akan|perlu)\s+", "Penelitian selanjutnya perlu ", t, flags=re.I)
    if t.lower().startswith(NOT_FOUND.lower()):
        return NOT_FOUND
    return re.sub(r"(\s*\[\d+\])+\s*$", "", t).strip()


# ---------- Tahap 1: document processing ----------
def extract_text(pdf_path: Path) -> str:
    import pymupdf
    parts = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    text = "\n".join(parts)
    text = re.sub(r"-\n(?=[a-z])", "", text)          # sambung kata yang terpotong
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # buang daftar pustaka agar tidak ikut di-retrieve
    m = list(re.finditer(r"\n\s*(references|bibliography|daftar pustaka)\s*\n", text, re.I))
    if m and m[-1].start() > len(text) * 0.5:
        text = text[: m[-1].start()]
    return text.strip()


# ---------- Tahap 2: chunking (ukuran tetap + overlap, per dokumen) ----------
def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # rapikan di batas kalimat/spasi terdekat
            cut = max(text.rfind(". ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 40:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- Tahap 3: embedding lokal ----------
class HashEmbedder:
    """Embedding tiruan (bag-of-words hash) hanya untuk uji alur tanpa unduh model."""
    dim = 256

    def encode(self, texts, **_):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def get_embedder(mock: bool, model_name: str):
    if mock:
        return HashEmbedder()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _embed(embedder, texts):
    vecs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True) \
        if not isinstance(embedder, HashEmbedder) else embedder.encode(texts)
    return [list(map(float, v)) for v in vecs]


# ---------- Tahap 4: vector store ----------
def build_index(papers, pdf_dir, db_dir, embedder, size, overlap):
    import chromadb
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection("papers")
    except Exception:
        pass
    col = client.create_collection("papers", metadata={"hnsw:space": "cosine"})
    stats = {}
    for p in papers:
        pdf = Path(pdf_dir) / p["pdf_file"]
        if not pdf.exists():
            print(f"[LEWAT] {p['paper_id']}: {pdf} tidak ada")
            continue
        chunks = chunk_text(extract_text(pdf), size, overlap)   # chunking per dokumen
        if not chunks:
            print(f"[LEWAT] {p['paper_id']}: teks kosong (PDF hasil scan?)")
            continue
        ids = [f"{p['paper_id']}::{i}" for i in range(len(chunks))]
        metas = [{"paper_id": p["paper_id"], "chunk_no": i} for i in range(len(chunks))]
        for s in range(0, len(chunks), 64):
            col.add(ids=ids[s:s+64], documents=chunks[s:s+64], metadatas=metas[s:s+64],
                    embeddings=_embed(embedder, chunks[s:s+64]))
        stats[p["paper_id"]] = len(chunks)
        print(f"[OK] {p['paper_id']}: {len(chunks)} chunk")
    return col, stats


# ---------- Tahap 5: retrieval ----------
# kata kunci penguat (hybrid): retrieval vektor sering melewatkan bagian "Limitations"/"Future work"
BOOST = {
    "Keterbatasan": r"limitations?\b|threats to validity|drawbacks?",
    "Future work": r"future (work|research|direction)s?|we plan to|in the future|open (challenges|problems)",
}


def retrieve(col, embedder, paper_id, query, k, boost=None):
    res = col.query(query_embeddings=_embed(embedder, [query]), n_results=k,
                    where={"paper_id": paper_id})
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    out = [{"chunk_no": m["chunk_no"], "text": d, "distance": float(x)} for d, m, x in zip(docs, metas, dists)]
    if boost:  # tambahkan sampai 2 potongan paling akhir yang memuat kata kunci (biasanya bagian penutup)
        have = {c["chunk_no"] for c in out}
        allc = col.get(where={"paper_id": paper_id})
        hits = sorted((m["chunk_no"], d) for d, m in zip(allc["documents"], allc["metadatas"])
                      if re.search(boost, d, re.I) and m["chunk_no"] not in have)
        for no, d in hits[-2:]:
            out.append({"chunk_no": no, "text": d, "distance": -1.0})
    return out


# ---------- Tahap 6: augmented generation ----------
class MockLLM:
    def chat(self, system, user):
        m = re.search(r"\[1\]\s*(.{0,160})", user, re.S)
        return (f"(MOCK) {m.group(1).strip()}... [1]" if m else NOT_FOUND), {"prompt_tokens": 0, "completion_tokens": 0}


class GroqLLM:
    def __init__(self, model):
        from openai import OpenAI
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY belum diset (lihat README).")
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        self.model = model

    def chat(self, system, user, retries=6):
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=2000,
                    extra_body={"reasoning_effort": "low"} if "gpt-oss" in self.model else None,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                u = r.usage
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("jawaban kosong")
                return text, {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0)}
            except Exception as e:  # rate limit / gangguan sementara
                if type(e).__name__ in ("NotFoundError", "AuthenticationError", "PermissionDeniedError", "BadRequestError"):
                    raise SystemExit(f"Error permanen dari Groq ({type(e).__name__}): {e}\nCek nama model (--model) dan GROQ_API_KEY.")
                if "tokens per day" in str(e) or "TPD" in str(e):
                    raise SystemExit("Kuota harian token Groq habis untuk model ini. Hasil yang sudah jadi tersimpan di out/cache.json. "
                                     "Jalankan lagi nanti (kuota pulih bertahap) atau pakai --model lain, mis. openai/gpt-oss-20b.")
                wait = min(60, 2 ** attempt * 3)
                print(f"  LLM error ({type(e).__name__}); ulangi dalam {wait}s")
                time.sleep(wait)
        raise RuntimeError("Groq gagal setelah beberapa percobaan")


def augmented_prompt(paper, field, chunks):
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return (f"Artikel: {paper['title']}\nInformasi yang diminta: {field}\n\nKONTEKS:\n{ctx}\n\n"
            f"Isi sel '{field}' untuk artikel ini.")


def generate_matrix(papers, col, embedder, llm, k, cache_path, out_dir):
    cache = json.loads(Path(cache_path).read_text()) if Path(cache_path).exists() else {}
    rows, evidence, usage, latencies = [], [], {"prompt": 0, "completion": 0}, []
    for p in papers:
        row = {"Kode": p["paper_id"], "Paper": f"{p['title']} ({p['authors']}, {p['year']})",
               "DOI": p.get("doi") or "-", "Status": p.get("status", "")}
        for field, query in FIELDS.items():
            answer, chunks = None, []
            for kk in (k, k * 2):             # jika tidak ditemukan, coba lagi dengan konteks lebih luas
                chunks = retrieve(col, embedder, p["paper_id"], query, kk, BOOST.get(field))
                if not chunks:
                    answer = NOT_FOUND
                    break
                prompt = augmented_prompt(p, field, chunks)
                key = hashlib.sha1((EXTRACT_SYSTEM + prompt).encode()).hexdigest()
                if key in cache:
                    raw = cache[key]
                else:
                    t0 = time.time()
                    raw, u = llm.chat(EXTRACT_SYSTEM, prompt)
                    latencies.append(time.time() - t0)
                    usage["prompt"] += u["prompt_tokens"]; usage["completion"] += u["completion_tokens"]
                    cache[key] = raw
                    Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False))
                answer = clean_answer(raw)
                if answer != NOT_FOUND:
                    break
            row[field] = answer
            evidence.append({"paper_id": p["paper_id"], "field": field, "answer": answer,
                             "chunks": chunks})
        rows.append(row)
        print(f"[MATRIKS] {p['paper_id']} selesai")
    return rows, evidence, usage, latencies


def gap_analysis(rows, llm):
    compact = "\n".join(
        f"[{r['Kode']}] {r['Paper']} | Status: {r['Status']}\n  Problem: {r['Problem']}\n  Metode: {r['Metode']}\n"
        f"  Dataset: {r['Dataset']}\n  Temuan: {r['Temuan']}\n  Keterbatasan: {r['Keterbatasan']}\n  Future work: {r['Future work']}"
        for r in rows)
    user = (f"MATRIKS LITERATUR ({len(rows)} paper, domain biometrik/identitas dan RAG multimodal):\n{compact}\n\n"
            "Tulis empat paragraf bernomor 1-4 tanpa tabel: (1) pola umum; (2) tiga gap, masing-masing "
            "disebut jenisnya (pengetahuan/metodologi/empiris-data) dan dirujuk ke paper [Pxx] yang mendukungnya; "
            "(3) satu pernyataan gap berpola 'Meskipun ..., ... masih terbatas pada korpus ini. Oleh karena itu, "
            "penelitian ini bertujuan ...'; (4) usulan paper kelompok yang sempit dan bisa dikerjakan.")
    text, _ = llm.chat(GAP_SYSTEM, user)
    return text


# ---------- keluaran ----------
def write_outputs(rows, evidence, usage, latencies, gap_text, out_dir, cfg, stats):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gap_matrix.csv", index=False, encoding="utf-8-sig")
    df.to_excel(out / "gap_matrix.xlsx", index=False)
    (out / "gap_matrix.md").write_text(df.to_markdown(index=False) if _has_tabulate() else df.to_string(), encoding="utf-8")
    with open(out / "evidence.jsonl", "w", encoding="utf-8") as f:
        for e in evidence:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "gap_analysis_draft.md").write_text(gap_text, encoding="utf-8")
    # lembar pemeriksaan manual: sampel sel untuk diverifikasi terhadap PDF
    random.seed(0)
    sample = random.sample(evidence, min(20, len(evidence)))
    with open(out / "manual_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "kolom", "jawaban_pipeline", "benar_(Y/N)", "catatan"])
        for e in sample:
            w.writerow([e["paper_id"], e["field"], e["answer"], "", ""])
    log = {"config": cfg, "chunks_per_paper": stats, "prompt_tokens": usage["prompt"],
           "completion_tokens": usage["completion"], "llm_calls": len(latencies),
           "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None}
    (out / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False