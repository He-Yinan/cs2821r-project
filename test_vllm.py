import os, sys
# Add rag/src to path so we can import hipporag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag", "src"))
from hipporag import HippoRAG


save_dir = "outputs"  # Define save directory for hipporag objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = "Qwen/Qwen3-8B"  # Any OpenAI model name
embedding_model_name = "facebook/contriever"  # Embedding model name (NV-Embed, GritLM or Contriever for now)
llm_base_url= "http://holygpu7c26105.rc.fas.harvard.edu:8000/v1"  # Base url for your deployed LLM (i.e. http://localhost:8000/v1)
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is born in Minsk.",
    "Montebello is a part of Rockland County."
]

hipporag = HippoRAG(save_dir=save_dir,
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name,
                    llm_base_url=llm_base_url)

# Run indexing
hipporag.index(docs=docs)

# Separate Retrieval & QA
queries = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?"
]

retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
qa_results = hipporag.rag_qa(retrieval_results)

# Combined Retrieval & QA
rag_results = hipporag.rag_qa(queries=queries)

# For Evaluation
answers = [
    ["Politician"],
    ["By going to the ball."],
    ["Rockland County"]
]

gold_docs = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.",
    "Montebello is a part of Rockland County."]
]

rag_results = hipporag.rag_qa(queries=queries, 
                              gold_docs=gold_docs,
                              gold_answers=answers)

print(rag_results)