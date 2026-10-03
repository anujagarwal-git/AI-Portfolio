"""HOW BIG is the batch-size difference in embeddings, and does it matter?

test_batching_does_not_change_vectors asserts BIT-IDENTICAL floats between
batch_size=1 and batch_size=3. Floating-point addition is not associative and
batching reshapes every matmul, so small differences are expected rather than
wrong - but "expected" is not a number. This measures it.

WHAT DECIDES IT
    The test's real invariant is not "identical bits". It is "batch size is a
    performance knob that cannot change retrieval results". Retrieval compares
    vectors by COSINE. So the question is whether the two vectors are the same
    direction to well within any distance the index could resolve.

    cosine 1.0 to ~7 decimals  -> the invariant HOLDS; the assertion is too
                                  strict and should use a tolerance.
    anything visible in cosine -> the invariant is BROKEN; do not loosen the
                                  test, fix the embedder.

    uv run python scripts/diagnose_batching.py
"""
from __future__ import annotations

import math

from regrag.index.embedder import embed_documents

TEXTS = ["Banks must validate models before use in capital planning.",
         "What is the definition of a model?",
         "A short third passage about capital buffers."]


def main() -> int:
    a = embed_documents(TEXTS, batch_size=1)
    b = embed_documents(TEXTS, batch_size=3)
    print(f"{len(a)} vectors, {len(a[0])} dims\n")
    print(f"  {'#':>3}{'max abs diff':>16}{'cosine':>22}{'1 - cosine':>14}")
    worst_cos = 1.0
    for i, (x, y) in enumerate(zip(a, b)):
        diff = max(abs(p - q) for p, q in zip(x, y))
        dot = sum(p * q for p, q in zip(x, y))
        nx = math.sqrt(sum(p * p for p in x))
        ny = math.sqrt(sum(q * q for q in y))
        cos = dot / (nx * ny)
        worst_cos = min(worst_cos, cos)
        print(f"  {i:>3}{diff:>16.3e}{cos:>22.15f}{1 - cos:>14.3e}")
    print()
    if 1 - worst_cos < 1e-9:
        print("  VERDICT: identical in direction to ~9+ decimals. Batch size cannot")
        print("  change a retrieval result. The invariant holds; the ASSERTION is")
        print("  too strict and should compare with a tolerance.")
    else:
        print("  VERDICT: the difference is visible in cosine. Do NOT loosen the test.")
        print("  Something in the embedder or its dependencies actually changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
