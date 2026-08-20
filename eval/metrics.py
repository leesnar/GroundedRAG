"""Retrieval metrics: recall@k and MRR against a (source, page) ground truth."""


def is_relevant(doc, expected_source, expected_page):
    return (
        doc.metadata.get("source") == expected_source
        and doc.metadata.get("page") == expected_page
    )


def hit_at_k(retrieved_docs, expected_source, expected_page):
    return any(is_relevant(d, expected_source, expected_page) for d in retrieved_docs)


def reciprocal_rank(retrieved_docs, expected_source, expected_page):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if is_relevant(doc, expected_source, expected_page):
            return 1.0 / rank
    return 0.0
