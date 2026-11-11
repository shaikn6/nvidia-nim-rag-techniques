r"""Answer-quality proxy: lexical overlap between a generated answer and a reference.

The latency benchmark and the retrieval eval say nothing about the generated
answer itself. This module adds a deliberately simple, offline proxy so answer
quality is at least visible alongside recall@k and MRR.

`answer_overlap_proxy` is a PROXY, not a true answer-correctness metric. It is
the fraction of a reference's content words (lowercased ``\w+`` tokens, minus a
small stopword list) that also appear in the answer -- pure lexical overlap.
What it does NOT measure:

- semantic correctness: a correct paraphrase that avoids the reference's words
  scores low, and a fluent but wrong answer that happens to reuse those words
  scores high;
- faithfulness or factuality beyond word presence;
- anything an LLM-as-judge or an entailment model would catch.

It needs no live LLM call, so it is deterministic and CI-safe. With the bundled
extractive `DemoLLM` (used when no NVIDIA NIM key is set), the generated answer
is a slice of the retrieved context, so the proxy there largely reflects whether
the correct passage reached the context -- not genuine answer correctness. Read
the number as a coarse grounding signal, nothing more.
"""

from __future__ import annotations

import re

# Function words carry no topical signal, so counting them would inflate overlap
# on every answer regardless of content.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
        "as", "by", "is", "are", "was", "were", "be", "been", "it", "its", "that",
        "this", "these", "those", "at", "from", "into", "than", "then", "so", "such",
        "can", "does", "do", "did", "which", "what", "how", "why", "when", "where",
        "who", "whom", "if", "not", "no", "yes", "i", "you", "your", "about",
    }
)

_TOKEN = re.compile(r"\w+")


def content_tokens(text: str) -> set[str]:
    r"""Lowercased ``\w+`` tokens with stopwords removed."""
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def answer_overlap_proxy(answer: str, reference: str) -> float:
    """Fraction of the reference's content tokens that also appear in the answer.

    Lexical overlap only -- a PROXY, not true answer-correctness; see the module
    docstring for what it does NOT measure. Returns 0.0 when the reference has no
    content tokens. Result is in [0.0, 1.0].
    """
    reference_tokens = content_tokens(reference)
    if not reference_tokens:
        return 0.0
    answer_tokens = content_tokens(answer)
    return len(reference_tokens & answer_tokens) / len(reference_tokens)
