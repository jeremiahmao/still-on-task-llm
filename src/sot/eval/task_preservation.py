"""Task preservation: Recall@10 on held-out query decomposition test set."""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.retrieval.encoder import Encoder
from sot.retrieval.index import search
from sot.retrieval.recall import decomposition_recall

_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


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
    batch_size: int = 8,
    chunk_to_article: list | None = None,
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
        batch_size: Generation batch size.

    Returns:
        Dict with Recall@10 stats.
    """
    model.eval()

    # Build all prompts upfront
    prompts = []
    for item in test_data:
        # Examples may have a top-level "question" OR nested in "messages" (chat format).
        question = item.get("question")
        if question is None:
            msgs = item.get("messages", [])
            for msg in msgs:
                if msg.get("role") == "user":
                    question = msg.get("content")
                    break
        if not question:
            continue
        chat = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    # Generate decompositions in batches
    all_raw_responses = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Preservation eval"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {ki: v.to(model.device) for ki, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(len(batch_prompts)):
            response = tokenizer.decode(
                outputs[j][prompt_len:], skip_special_tokens=True
            ).strip()
            all_raw_responses.append(response)

    # Parse sub-queries and retrieve per item
    all_sub_query_results = []
    all_gold_ids = []

    for item, response in zip(test_data, all_raw_responses):
        gold = set(item.get("gold_articles", []))

        # Extract question (same fallback as above)
        item_question = item.get("question")
        if item_question is None:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    item_question = msg.get("content")
                    break

        sub_queries = [
            line.lstrip("- ").strip()
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        if not sub_queries and item_question:
            sub_queries = [item_question]

        sq_embeddings = encoder.encode(sub_queries, show_progress=False)
        scores, indices = search(faiss_index, sq_embeddings, k=k, nprobe=nprobe)

        sq_results = []
        for row in indices:
            retrieved = set()
            for idx in row:
                if 0 <= idx < len(doc_ids):
                    # Map chunk index to article index if chunking was used
                    if chunk_to_article is not None and idx < len(chunk_to_article):
                        retrieved.add(int(chunk_to_article[idx]))
                    else:
                        retrieved.add(doc_ids[idx])
            sq_results.append(retrieved)

        all_sub_query_results.append(sq_results)
        all_gold_ids.append(gold)

    return decomposition_recall(all_sub_query_results, all_gold_ids, k=k)
