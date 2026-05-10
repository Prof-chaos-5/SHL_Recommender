"""
Recall@K evaluation.
Usage: python -m eval.run_eval
"""

def recall_at_k(relevant: list[str], recommended: list[str], k: int = 10) -> float:
    """
    relevant: list of expected assessment names (ground truth)
    recommended: list of recommended assessment names from agent
    k: cutoff
    """
    if not relevant:
        return 1.0  # nothing expected, nothing to miss
    top_k = recommended[:k]
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def recall_at_k_fuzzy(relevant: list[str], recommended: list[str], k: int = 10, threshold: int = 80) -> float:
    """
    Fuzzy matching version - handles product name variants.
    A recommendation counts if it's >= threshold% similar to an expected product.
    
    relevant: list of expected assessment names (ground truth)
    recommended: list of recommended assessment names from agent
    k: cutoff
    threshold: minimum fuzzy match score (0-100)
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Warning: fuzzywuzzy not available, falling back to exact matching")
        return recall_at_k(relevant, recommended, k)
    
    if not relevant:
        return 1.0
    
    top_k = recommended[:k]
    hits = 0
    
    for exp in relevant:
        # Find best match in recommended
        best_score = max([fuzz.token_set_ratio(exp, rec) for rec in top_k], default=0)
        if best_score >= threshold:
            hits += 1
    
    return hits / len(relevant)


def mean_recall_at_k(traces: list[dict], k: int = 10) -> float:
    """
    traces: list of {"expected": [...names], "recommended": [...names]}
    """
    if not traces:
        return 0.0
    scores = [recall_at_k(t["expected"], t["recommended"], k) for t in traces]
    return sum(scores) / len(scores)


def mean_recall_at_k_fuzzy(traces: list[dict], k: int = 10, threshold: int = 80) -> float:
    """
    Fuzzy matching version - handles product name variants.
    """
    if not traces:
        return 0.0
    scores = [recall_at_k_fuzzy(t["expected"], t["recommended"], k, threshold) for t in traces]
    return sum(scores) / len(scores)
