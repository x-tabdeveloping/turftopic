def test_cvp():
    from turftopic import ConceptVectorProjection

    cuteness_seeds = (
        ["Absolutely adorable", "I love how he dances with his little feet"],
        [
            "What a big slob of an abomination",
            "A suspicious old man sat next to me on the bus today",
        ],
    )
    bullish_seeds = (
        [
            "We are going to the moon",
            "This stock will prove an incredible investment",
        ],
        [
            "I will short the hell out of them",
            "Uber stocks drop 7% in value after down-time.",
        ],
    )
    seeds = [("cuteness", cuteness_seeds), ("bullish", bullish_seeds)]
    cvp = ConceptVectorProjection(seeds=seeds)
    test_documents = ["What an awesome investment", "Tiny beautiful kitty-cat"]
    doc_concept_matrix = cvp.transform(test_documents)
    assert doc_concept_matrix.shape == (2, 2)
