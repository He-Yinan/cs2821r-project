from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "HippoRAG" / "src"))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

dataset_name = "musique"
num_samples = 50
# save_dir = "/n/home13/yinan/cs2821r-project/results/experiments/musique_demo/hipporag_workspace"
save_dir = "/n/home13/yinan/cs2821r-project/results/experiments/musique_demo/hipporag_workspace_1"
corpus_path = f"/n/holylabs/ydu_lab/Lab/yinan/28-results/datasets/{dataset_name}/subset_{num_samples}/{dataset_name}_corpus.json"
samples_file = f"/n/holylabs/ydu_lab/Lab/yinan/28-results/datasets/{dataset_name}/subset_{num_samples}/{dataset_name}.json"

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


with open(corpus_path, "r") as f:
    corpus = json.load(f)

docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

samples = json.load(open(samples_file, "r"))
all_queries = [s['question'] for s in samples]

gold_answers = get_gold_answers(samples)
try:
    gold_docs = get_gold_docs(samples, dataset_name)
    assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
except:
    gold_docs = None

config = BaseConfig(
    save_dir=save_dir,
    llm_base_url="http://holygpu7c26105.rc.fas.harvard.edu:8001/v1",
    llm_name="Qwen/Qwen3-8B",
    dataset=dataset_name,
    embedding_model_name="facebook/contriever-msmarco",
    force_index_from_scratch=False,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
    force_openie_from_scratch=False,
    # rerank_dspy_file_path="/n/home13/yinan/cs2821r-project/HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
    rerank_dspy_file_path=None,
    retrieval_top_k=200,
    linking_top_k=5,
    max_qa_steps=3,
    qa_top_k=5,
    graph_type="facts_and_sim_passage_node_unidirectional",
    embedding_batch_size=8,
    max_new_tokens=None,
    corpus_len=len(corpus),
    openie_mode="online",
)
config.disable_rerank_filter = True
config.num_facts_without_rerank = 10

hipporag = HippoRAG(global_config=config)
hipporag.index(docs)

# Retrieval and QA
hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)