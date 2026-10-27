# Literature Gap Matrix dengan Pipeline RAG: RAG Multimodal untuk Biometrik dan Identitas

Tugas mata kuliah Introduction to Generative AI (Magister AI, UDINUS).

Anggota kelompok:
1. Rahmanda Afridiansyah (P36.2026.00018)
2. Mohammad Rafii Burhanuddin (P36.2026.00003)

Domain mengikuti use-case Tugas 1/2: *Adaptive Quality-Aware Score Fusion pada Sistem Identifikasi Multimodal Face, Fingerprint, dan Iris dalam Kondisi Non-Ideal*. Korpus berisi 10 makalah tentang RAG pada deteksi deepfake yang dapat dijelaskan, keamanan dan privasi RAG multimodal, survei RAG multimodal, dan satu studi Graph RAG untuk verifikasi identitas (KYC).

## Hasil akhir (folder `hasil_final/`)
- `gap_matrix_final.csv`: Literature Gap Matrix 10 makalah (Paper, DOI, Status, Problem, Metode, Dataset, Temuan, Keterbatasan, Future work). Kolom `kolom_dikoreksi_manual` dan `perlu_dicek` menandai sel yang dikoreksi atau belum terverifikasi.
- `gap_analysis_revisi.md`: paragraf gap analysis dan usulan paper kelompok.
- `manual_check_terisi.csv`: hasil pemeriksaan 20 sel acak terhadap PDF sumber.

## Pipeline (6 tahap RAG)
1. Document processing: PyMuPDF, daftar pustaka dibuang.
2. Chunking: ukuran tetap 1000 karakter, overlap 150, per dokumen.
3. Embedding lokal: `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers).
4. Vector store: ChromaDB (cosine).
5. Retrieval: top-k chunk hanya dari paper yang sedang diisi (filter `paper_id`); untuk kolom Keterbatasan dan Future work, chunk yang memuat kata kunci terkait ikut ditambahkan.
6. Augmented generation: Groq (API kompatibel OpenAI), instruksi "jawab hanya dari konteks, kalau tidak ada tulis *Tidak ditemukan*".

Judul, penulis, DOI, dan status dibaca dari `papers.csv` (diisi manual), bukan dibuat LLM.

## Cara menjalankan
```bash
pip install -r requirements.txt
export GROQ_API_KEY=...            # Colab: from google.colab import userdata
# letakkan P01.pdf sampai P10.pdf di pdfs/ sesuai papers.csv
python run.py                      # hasil di out/
python run.py --mock               # uji alur tanpa jaringan/API
```
Opsi: `--k 4`, `--model openai/gpt-oss-120b`, `--chunk-size`, `--overlap`. Jawaban disimpan di `out/cache.json`, sehingga run yang terhenti (misalnya kena batas kuota) dapat dilanjutkan tanpa mengulang sel yang sudah jadi.

PDF makalah tidak disertakan di repositori. Daftar makalah ada di `papers.csv` (tautan arXiv/ACL/AAAI).

## Model yang dipakai
Matriks dibuat dengan `openai/gpt-oss-120b` (Groq). Karena kuota harian Groq habis, makalah terakhir (P09 dan P10) dibuat dengan `openai/gpt-oss-20b`. Konfigurasi dan jumlah token tiap run ada di `out/run_log.json`.

## Verifikasi dan penggunaan AI
- Sel matriks diisi oleh LLM lewat pipeline RAG di atas. Dari 20 sel acak yang diperiksa terhadap PDF: 13 benar, 6 sebagian benar, 1 salah (`manual_check_terisi.csv`).
- Sel yang bermasalah dikoreksi berdasarkan teks PDF sumber, dibantu asisten AI (Claude) yang mencocokkan sel dengan isi PDF. Koreksi ditandai di kolom `kolom_dikoreksi_manual`.
- Lima sel diverifikasi langsung oleh Rahmanda terhadap PDF sumber: P01 (Keterbatasan), P03 (Keterbatasan), P04 (Keterbatasan), P06 (Dataset), P10 (Temuan).
- Kode pipeline dan draf gap analysis disusun dengan bantuan asisten AI (Claude), lalu ditinjau dan disesuaikan oleh anggota kelompok. Usulan penelitian pada gap analysis adalah usulan, bukan klaim hasil.

## Keterbatasan
- Ekstraksi LLM dapat keliru atau terlalu ringkas; sel bertanda `perlu_dicek` belum terverifikasi penuh (Keterbatasan P06 karena paper tidak punya bagian Limitations; Temuan dan Keterbatasan P07).
- Korpus hanya 10 makalah, sehingga klaim celah berlaku untuk korpus ini, bukan seluruh literatur.
- Sumber campuran: peer-reviewed dan preprint (kolom `Status`).
- PDF hasil scan tidak terbaca (tanpa OCR).
