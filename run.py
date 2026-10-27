"""Jalankan: python run.py --papers papers.csv --pdf-dir pdfs --out out"""
import argparse, csv
from pathlib import Path
import pipeline as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers.csv")
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--out", default="out")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--k", type=int, default=4, help="jumlah chunk per pertanyaan")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--model", default="llama-3.3-70b-versatile", help="model Groq")
    ap.add_argument("--embed-model", default="paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--mock", action="store_true", help="uji alur tanpa jaringan/API")
    a = ap.parse_args()

    papers = list(csv.DictReader(open(a.papers, encoding="utf-8")))
    embedder = pl.get_embedder(a.mock, a.embed_model)
    col, stats = pl.build_index(papers, a.pdf_dir, a.db, embedder, a.chunk_size, a.overlap)
    papers = [p for p in papers if p["paper_id"] in stats]
    if not papers:
        raise SystemExit("Tidak ada PDF yang terbaca. Letakkan PDF di folder pdfs/ sesuai kolom pdf_file.")
    llm = pl.MockLLM() if a.mock else pl.GroqLLM(a.model)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows, ev, usage, lat = pl.generate_matrix(papers, col, embedder, llm, a.k, Path(a.out) / "cache.json", a.out)
    gap = pl.gap_analysis(rows, llm)
    cfg = {k: v for k, v in vars(a).items()}
    pl.write_outputs(rows, ev, usage, lat, gap, a.out, cfg, stats)
    print(f"Selesai. Lihat folder {a.out}/")


if __name__ == "__main__":
    main()
