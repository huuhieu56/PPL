"""Fresh source extraction/indexing with system code; native Ragas generation."""
import dataclasses
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
# Ragas invokes asyncio.run between stages; keep its HTTP client on one loop.
import nest_asyncio
nest_asyncio.apply()
os.environ.setdefault('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/4.00/tessdata')
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from src.chunking import build_hierarchy
from src.documents import extract_document
from src.config import load_settings
from src.embeddings import api_embeddings, embedding_config
from src.index import RetrievalIndex
from src.rag import chat_model

folder = ROOT / 'eval'
source_folder = ROOT / 'data'
source_paths = [source_folder / relative for relative in (
    'triet_hoc/ptit-philosophy-lecture-2021.pdf',
    'kinh_te_chinh_tri/ptit-economics-lecture-2021.pdf',
    'phap_luat_dai_cuong/ptit-law-lecture-2019.pdf',
    'lich_su_dang/bai_giang_2021.pdf',
    'cnxh_khoa_hoc/socialism-national-textbook-unverified-edition.pdf',
)]
missing = [str(path) for path in source_paths if not path.is_file()]
if missing:
    raise SystemExit('Thiếu PDF nguồn: ' + ', '.join(missing) + '. Chưa xóa kết quả cũ.')
if not source_paths:
    raise SystemExit(f'Không tìm thấy PDF nguồn tại {source_folder}. Chưa xóa kết quả cũ.')
# Every generation invocation starts from scratch; preserve scripts and documentation.
for path in folder.iterdir():
    if path.suffix not in {'.py', '.md'}:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
(folder / 'results').mkdir()
settings = load_settings(ROOT)
rng = random.Random(20261005)
chunks, nodes, documents, source_records = [], [], [], []
for path in source_paths:
    course = path.parent.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pages = extract_document(path)
    assert any(p.text.strip() for p in pages), f'No extracted text: {path.name}'
    new_chunks, new_nodes = build_hierarchy(pages, doc_id=digest[:24], course=course, source_type='textbook', chunk_tokens=450, overlap_tokens=75, title=path.name, file_type='pdf')
    chunks.extend(new_chunks)
    nodes.extend(new_nodes)
    # Fixed random source excerpts for a small pilot, before any benchmark exists.
    eligible = [i for i in range(7, len(pages)-4) if all(len(p.text.split()) >= 140 for p in pages[i:i+5])]
    start = rng.choice(eligible)
    excerpt = pages[start:start+5]
    documents.append(Document(page_content='\n\n'.join(p.text for p in excerpt), metadata={'source': path.name, 'pages': [p.page for p in excerpt]}))
    source_records.append({'source': str(path), 'sha256': digest, 'extracted_pages': len(pages), 'chunks': len(new_chunks), 'benchmark_source_pages': [p.page for p in excerpt]})
    print('Fresh extraction/chunking:', course, len(pages), len(new_chunks), flush=True)
(folder / 'sources.json').write_text(json.dumps(source_records, ensure_ascii=False, indent=2))
embedding = embedding_config(os.environ['EMBEDDING_MODEL'])
client = QdrantClient(path=str(folder / 'qdrant'))
try:
    RetrievalIndex.build(chunks, folder / 'index', 'whitespace', embedding['model'], qdrant_client=client, hierarchy_nodes=nodes)
finally:
    client.close()
print('Fresh system index built:', len(chunks), flush=True)
embeddings = LangchainEmbeddingsWrapper(api_embeddings(embedding['model'], embedding['base_url'], os.environ['EMBEDDING_API_KEY']))
model = chat_model(settings)
model.max_retries = 0
run_config = RunConfig(timeout=120, max_retries=0, max_workers=1)
llm = LangchainLLMWrapper(model, run_config=run_config, bypass_n=True)
generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
try:
    testset = generator.generate_with_langchain_docs(documents, testset_size=25, query_distribution=[(SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)], run_config=run_config)
    rows = testset.to_evaluation_dataset().to_list()
    (folder / 'dataset25.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    print('Native unchanged Ragas samples:', len(rows), flush=True)
except Exception as error:
    (folder / 'results/generation_error.json').write_text(json.dumps({'error_type': type(error).__name__, 'message': str(error)[:300]}))
    raise
