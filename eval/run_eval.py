"""Run the existing RAG and score unchanged outputs with four default Ragas metrics."""
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
# Ragas invokes asyncio.run between stages; keep its HTTP client on one loop.
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from qdrant_client import QdrantClient
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from src.config import load_settings, load_yaml
from src.embeddings import api_embeddings, embedding_config
from src.index import RetrievalIndex
from src.models import RagConfig
import src.rag as rag

folder = ROOT / 'eval/results'
source = [json.loads(line) for line in (ROOT / 'eval/dataset25.jsonl').read_text().splitlines()]
settings = load_settings(ROOT)
config_values = load_yaml(ROOT / 'configs/default.yaml')['retrieval']
config = RagConfig(**{key: value for key, value in config_values.items() if key in RagConfig.__dataclass_fields__})
client = QdrantClient(path=str(ROOT / 'eval/qdrant'))
index = RetrievalIndex.load(ROOT / 'eval/index', qdrant_client=client)
model = rag.chat_model(settings)
path = folder / 'responses.jsonl'
for old_name in ['responses.jsonl','scores.jsonl','summary.json']:
    old_path=folder / old_name
    if old_path.exists():old_path.unlink()
records = []
trace = []
original_prompt = rag._prompt

def capture_prompt(query, results):
    trace[:] = [result.chunk.text for result in results]
    return original_prompt(query, results)

rag._prompt = capture_prompt
try:
    for i, sample in enumerate(source):
        trace.clear()
        try:
            answer = rag.answer_question(sample['user_input'], index, config, model, settings.llm_model)
            record = {'sample': sample, 'answer': dataclasses.asdict(answer), 'retrieved_contexts': list(trace), 'error': None}
        except Exception as error:
            record = {'sample': sample, 'error': type(error).__name__}
        with path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        records.append(record)
        print('RAG', i + 1, '/', len(source), record.get('error') or 'saved', flush=True)
finally:
    client.close()
assert not any(record['error'] for record in records), 'Pipeline errors preserved; do not silently exclude samples'
assert not (folder / 'scores.jsonl').exists(), 'Do not overwrite scores'
data = EvaluationDataset.from_list([{**record['sample'], 'response': record['answer']['text'], 'retrieved_contexts': record['retrieved_contexts']} for record in records])
embedding = embedding_config(os.environ['EMBEDDING_MODEL'])
embeddings = LangchainEmbeddingsWrapper(api_embeddings(embedding['model'], embedding['base_url'], os.environ['EMBEDDING_API_KEY']))
run_config = RunConfig(timeout=90, max_retries=0, max_workers=1)
metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
# Request separate generations: an OpenAI-compatible endpoint may ignore n=3.
result = evaluate(data, metrics=metrics, llm=LangchainLLMWrapper(model, run_config=run_config, bypass_n=True), embeddings=embeddings, run_config=run_config, raise_exceptions=False)
frame = result.to_pandas()
frame.to_json(folder / 'scores.jsonl', orient='records', lines=True, force_ascii=False)
summary = {'samples': len(source), 'metrics': {metric.name: {'mean': None if frame[metric.name].notna().sum() == 0 else float(frame[metric.name].mean()), 'missing': int(frame[metric.name].isna().sum())} for metric in metrics}}
(folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False), flush=True)
