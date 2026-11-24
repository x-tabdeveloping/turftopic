import numpy as np

from turftopic import DSL
from turftopic.encoders import SoftLabelEncoder

DOCS = [
    "the hockey team won the game in overtime last night",
    "the senate passed the budget bill after a long debate",
    "new graphics card drivers improve game performance on windows",
    "the patient was given antibiotics for the infection",
] * 6


def test_dsl_fit_transform_and_export():
    teacher = SoftLabelEncoder(
        "sshleifer/tiny-gpt2", batch_size=8, device="cpu"
    )
    model = DSL(3, encoder=teacher, n_epochs=2, random_state=0)
    doc_topic = model.fit_transform(DOCS)
    assert doc_topic.shape == (len(DOCS), 3)
    assert np.allclose(doc_topic.sum(1), 1, atol=1e-4)
    assert model.components_.shape == (3, len(model.get_vocab()))
    assert model.transform(DOCS[:2]).shape == (2, 3)
    topic_data = model.prepare_topic_data(DOCS)
    assert topic_data.document_topic_matrix.shape == (len(DOCS), 3)
    assert len(model.get_top_words()) == 3


def test_dsl_accepts_precomputed_teacher_outputs():
    teacher = SoftLabelEncoder(
        "sshleifer/tiny-gpt2", batch_size=8, device="cpu"
    )
    model = DSL(3, encoder=teacher, n_epochs=1, random_state=0)
    model.vectorizer.fit(DOCS)
    logits, hidden, _ = teacher.extract(DOCS, list(model.get_vocab()))
    first = model.fit_transform(DOCS, embeddings=hidden, logits=logits)
    second = DSL(3, encoder=teacher, n_epochs=1, random_state=0).fit_transform(
        DOCS, embeddings=hidden, logits=logits
    )
    assert np.allclose(first, second, atol=1e-5)
