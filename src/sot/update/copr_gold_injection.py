"""COPR + gold injection: our novel contribution for knowledge injection.

Extends the paper-faithful `COPRUpdate` with a single change: the gold answer
is always included in the candidate set before ranking. Everything else — MSE
fit loss, SFT anchor, replay — is unchanged from the paper.

**Why this matters for knowledge injection.** COPR was designed for preference
alignment, where candidate responses are ranked according to human preferences.
In that setting, at least one reasonable response is typically in the sampled
set because the model already produces reasonable outputs — preferences just
re-rank among them.

In the knowledge-injection setting, the model has never seen the fact being
injected. All K self-samples routinely miss the gold answer entirely. Then
advantage-based ranking just reorders "least-wrong" wrong responses, teaching
the model to prefer plausible-sounding-but-wrong outputs over others.

Gold injection guarantees at least one correct response in the candidate set
for every fact, anchoring the ranking. The COPR paper does not need this
because the preference setting does not have this failure mode.
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.copr import COPRUpdate


class COPRGoldInjectionUpdate(COPRUpdate):
    """COPR with gold injection into the candidate set.

    Inherits all paper-faithful behaviour (MSE fit loss, SFT anchor, replay reg)
    from `COPRUpdate`. Only `_prepare_fit_data` is overridden.
    """

    @property
    def name(self) -> str:
        return "copr_gold_injection"

    def _prepare_fit_data(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        K: int,
        max_new_tokens: int,
        partial_match_threshold: float,
    ) -> list[dict]:
        """Sample K responses, INJECT the gold answer, dedupe, rank.

        Guarantees at least one exact-match response in the candidate set for
        every fact. The rest of the COPR pipeline (P* computation, MSE fit
        loss, gold NLL anchor, replay reg) is unchanged.
        """
        model.eval()
        fit_data = []

        for i, qa in enumerate(tqdm(fact_qa_pairs, desc="Sampling responses")):
            question = qa["question"]
            gold_answer = qa["answer"]

            responses = self._sample_responses(model, tokenizer, question, K, max_new_tokens)

            # Gold injection: prepend gold to candidates, dedup (case-insensitive).
            candidates = [gold_answer] + [r for r in responses if r.strip()]
            seen = set()
            unique = []
            for r in candidates:
                key = r.strip().lower()
                if key and key not in seen:
                    unique.append(r)
                    seen.add(key)
            # Keep gold + up to K sampled unique candidates.
            if len(unique) > K + 1:
                unique = [unique[0]] + unique[1 : K + 1]

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            ranked = self._rank_responses(unique, gold_answer, partial_match_threshold)
            # Linear advantages from worst (j=0) to best (j=len-1).
            advantages = [
                (2 * j - len(ranked) + 1) / max(len(ranked), 1) for j in range(len(ranked))
            ]

            fit_data.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "ranked_responses": ranked,
                    "advantages": advantages,
                    "log_p_star": None,
                }
            )

        model.train()
        return fit_data
