"""Task preservation: Recall@10 on held-out query decomposition test set."""

import torch
from transformers import AutoTokenizer, PreTrainedModel

from sot.retrieval.encoder import Encoder
from sot.retrieval.index import search
from sot.retrieval.recall import decomposition_recall


def evaluate_task_preservation(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    test_data: list[dict],
    encoder: Encoder,
    faiss_index,
    doc_ids: list,
    k: int = 10,
    nprobe: int = 64,
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate whether the model's decomposition skill is preserved.

    Generates sub-query decompositions for held-out test questions,
    retrieves documents, and computes Recall@10.

    Args:
        model: Updated model to evaluate.
        tokenizer: Tokenizer.
        test_data: List of test examples with 'question' and 'gold_articles'.
        encoder: BGE-M3 encoder.
        faiss_index: Pre-built FAISS index.
        doc_ids: FAISS ID -> article ID mapping.
        k: Top-k for retrieval.
        nprobe: FAISS nprobe.
        max_new_tokens: Max generation length for decompositions.

    Returns:
        Dict with Recall@10 stats.
    """
    model.eval()
    all_sub_query_results = []
    all_gold_ids = []

    for item in test_data:
        question = item["question"]
        gold = set(item.get("gold_articles", []))

        # Generate decomposition
        chat = [
            {
                "role": "system",
                "content": (
                    "You are a financial search expert. Given a complex financial question, "
                    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
                    "relevant documents from a financial news database."
                ),
            },
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {ki: v.to(model.device) for ki, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Parse sub-queries from response (expect "- sub_query" format)
        sub_queries = [
            line.lstrip("- ").strip()
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        if not sub_queries:
            sub_queries = [question]  # Fallback to original question

        # Embed and retrieve per sub-query
        sq_embeddings = encoder.encode(sub_queries, show_progress=False)
        scores, indices = search(faiss_index, sq_embeddings, k=k, nprobe=nprobe)

        # Collect retrieved doc IDs per sub-query
        sq_results = []
        for row in indices:
            retrieved = set()
            for idx in row:
                if 0 <= idx < len(doc_ids):
                    retrieved.add(doc_ids[idx])
            sq_results.append(retrieved)

        all_sub_query_results.append(sq_results)
        all_gold_ids.append(gold)

    return decomposition_recall(all_sub_query_results, all_gold_ids, k=k)
