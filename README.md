# ttrpgFileParser

A Python AI application that **catalogues, parses, and semantically searches** through TTRPG files — PDFs, images, plain text, and more. Ask questions like *"give me all PDFs that have crafting rules"* and get back ranked results with the matching file path and the relevant passage.

---

## Features

| Feature | Details |
|---|---|
| **File parsing** | PDF (PyMuPDF), Images/OCR (Pillow + Tesseract), plain text / Markdown / CSV |
| **Semantic search** | Sentence-transformers embeddings (`all-MiniLM-L6-v2`) with cosine-similarity ranking |
| **Persistence** | SQLite — all file metadata, text chunks, and vector embeddings stored locally |
| **REST API** | FastAPI with auto-generated OpenAPI docs at `/docs` |

---

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH` (required for image files only)

---

## Installation

```bash
# clone the repo, then:
pip install -r requirements.txt
```

---

## Configuration

All settings are read from environment variables (prefix `TTRPG_`):

| Variable | Default | Description |
|---|---|---|
| `TTRPG_DATA_DIR` | `./data` | Directory for the SQLite database |
| `TTRPG_DB_PATH` | `./data/ttrpg.db` | Full path to the SQLite file |
| `TTRPG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `TTRPG_CHUNK_SIZE` | `500` | Characters per text chunk |
| `TTRPG_CHUNK_OVERLAP` | `50` | Overlap characters between chunks |
| `TTRPG_TOP_K_RESULTS` | `10` | Default number of search results |
| `TTRPG_BASE_DIR` | *(home directory)* | Root directory that restricts which paths can be catalogued.  Any `POST /catalogue` request targeting a path outside this root is rejected with HTTP 403. |

---

## Running the API

```bash
uvicorn app.main:app --reload
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

### `POST /catalogue`
Scan a directory, extract text from every supported file, and build the vector index.

```json
{
  "directory": "/path/to/my/ttrpg/books",
  "recursive": true
}
```

Returns a summary with counts of indexed, failed, and skipped files.

---

### `POST /search`
Natural-language semantic search across all indexed files.

```json
{
  "query": "give me all PDFs that have crafting rules",
  "top_k": 5
}
```

Returns a ranked list of matching files with their paths and the best-matching excerpt:

```json
{
  "query": "give me all PDFs that have crafting rules",
  "results": [
    {
      "file_id": 3,
      "name": "dungeon_masters_guide.pdf",
      "path": "/books/dungeon_masters_guide.pdf",
      "file_type": "pdf",
      "score": 0.8712,
      "matched_text": "Crafting a magic item requires …"
    }
  ]
}
```

---

### `GET /files`
List all catalogued files with their status (`pending`, `indexed`, `error`).

### `GET /files/{file_id}`
Get details for a single file.

### `DELETE /files/{file_id}`
Remove a file and all its indexed data.

---

## Supported File Types

| Extension | Parser |
|---|---|
| `.pdf` | PyMuPDF (text extraction) |
| `.png` `.jpg` `.jpeg` `.gif` `.bmp` `.tiff` `.tif` `.webp` | Tesseract OCR |
| `.txt` `.md` `.rst` `.csv` | Plain text (UTF-8, Latin-1, CP-1252) |

---

## Running Tests

```bash
pytest
```

---

## Architecture

```
app/
├── main.py          # FastAPI app and route handlers
├── config.py        # Pydantic-settings configuration
├── database.py      # SQLite schema and CRUD helpers
├── models.py        # Pydantic request / response models
├── cataloguer.py    # Directory scanning and indexing orchestration
├── chunker.py       # Text splitting with overlap
├── parsers/
│   ├── factory.py   # Returns the right parser for a file extension
│   ├── pdf_parser.py
│   ├── image_parser.py
│   └── text_parser.py
└── search/
    ├── embeddings.py   # sentence-transformers wrapper
    └── vector_store.py # numpy cosine-similarity search over SQLite-stored vectors
```