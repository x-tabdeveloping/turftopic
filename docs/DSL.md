# DSL (Distilling Soft Labels)

DSL (Li et al., 2026) is a way of training neural topic models on targets produced by a small language model, instead of on bag-of-words.

A causal language model reads each document inside a short prompt that asks for its theme.
The label is never generated. Instead, DSL keeps two things from that single forward pass:

 1. The next-token distribution at the end of the prompt, restricted to the vocabulary and softened with a temperature. This **soft label** replaces the bag-of-words vector as the reconstruction target.
 2. The last hidden state at the end of the prompt. This replaces sentence embeddings as the document representation.

Bag-of-words targets can only give weight to words that occur in a document.
A soft label also gives weight to theme words that the document never uses, such as "hockey" for a post about a trade rumour, which matters most for short texts.

The topic model in Turftopic's implementation is a ProdLDA-style variational autoencoder, and its loss is the KL divergence between the soft label and the reconstruction, plus the usual prior term.
The paper applies the same target to ECRTM and FASTopic as well.

## Usage

```python
from turftopic import DSL

documents = [...]

model = DSL(20)
doc_topic_matrix = model.fit_transform(documents)
model.print_topics()
```

The language model is run once over the corpus. A GPU is recommended for this step; the default teacher has 0.3B parameters.
You can choose another causal language model from the Hugging Face Hub, or configure the teacher yourself:

```python
from turftopic import DSL
from turftopic.encoders import SoftLabelEncoder

teacher = SoftLabelEncoder("Qwen/Qwen3.5-0.8B", batch_size=32, max_length=512)
model = DSL(20, encoder=teacher)
```

The teacher's outputs can be computed once and reused across models:

```python
model = DSL(20, encoder=teacher)
model.vectorizer.fit(documents)
logits, hidden_states, _ = teacher.extract(documents, list(model.get_vocab()))

model.fit(documents, embeddings=hidden_states, logits=logits)
```

!!! note
    Only vocabulary words that are a single alphabetic token for the language model are part of the soft label.
    The default vectorizer keeps the 2000 most frequent non-stop words; phrase vectorizers are not a good fit for this model.

## Citation

Please cite the authors of the paper, and Turftopic when using the DSL model:

```bibtex
@inproceedings{li2026dsltopic,
  title     = {Improving Topic Modeling by Distilling Soft Labels from Language Models},
  author    = {Li, Raymond and Abaskohi, Amirhossein and Li, Chuyuan and Murray, Gabriel and Carenini, Giuseppe},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {65815--65836},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/li26a.html}
}

@article{
  Kardos2025,
  title = {Turftopic: Topic Modelling with Contextual Representations from Sentence Transformers},
  doi = {10.21105/joss.08183},
  url = {https://doi.org/10.21105/joss.08183},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {111},
  pages = {8183},
  author = {Kardos, Márton and Enevoldsen, Kenneth C. and Kostkan, Jan and Kristensen-McLachlan, Ross Deans and Rocca, Roberta},
  journal = {Journal of Open Source Software}
}
```

## API Reference

::: turftopic.models.dsl.DSL

::: turftopic.encoders.soft_labels.SoftLabelEncoder
