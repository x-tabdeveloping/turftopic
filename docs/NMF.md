# Nonnegative Matrix Factorization

NMF is a factor-based topic model of bag-of-words or tf-idf representations.
It treats topics as latent factors that explain variance in word counts/importances in documents.
Factors are strictly positive, hence the name.

!!! tip
    NMF is an older topic model with substantially less interpretable topics than modern approaches, and with no contextual understanding.
    If you want to use a model that is more powerful but similar to NMF, we recommend you try out [KeyNMF](KeyNMF.md) or [SensTopic](SensTopic.md).

<figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/f9/NMF.png?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=thumbnail_unscaled", title="", style="width:600px;padding:0px;border:none;"></img>
  <figcaption> Figure 1: Depiction of Nonnegative Matrix Factorization.<br> <i>By Qwertyus - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=29114677</i> </figcaption>
</figure>



## Usage

```python
from turftopic import NMF

documents = [...]

model = NMF(10)
doc_topic_matrix = model.fit_transform(documents)
model.print_topics()
```

!!! tip
    Since NMF is a BoW topic model, it does not handle stop words particularly well.
    We recommend that you use a vectorizer that removes stop words or all too frequent words, potentially with tf-idf weighting
    ```python
    from turftopic import LDA
    from sklearn.feature_extraction.text import TfidfVectorizer

    model = NMF(10, vectorizer=TfidfVectorizer(min_df=10, stop_words="english", max_df=0.8))
    ```

## Citation

Please cite the authors of the implementation, and Turftopic when using NMF:

```bibtex
@article{Cedric2011,
    author = {Févotte, Cédric and Idier, Jérôme},
    title = {Algorithms for Nonnegative Matrix Factorization with the β-Divergence},
    journal = {Neural Computation},
    volume = {23},
    number = {9},
    pages = {2421-2456},
    year = {2011},
    month = {09},
    abstract = {This letter describes algorithms for nonnegative matrix factorization (NMF) with the β-divergence (β-NMF). The β-divergence is a family of cost functions parameterized by a single shape parameter β that takes the Euclidean distance, the Kullback-Leibler divergence, and the Itakura-Saito divergence as special cases (β = 2, 1, 0 respectively). The proposed algorithms are based on a surrogate auxiliary function (a local majorization of the criterion function). We first describe a majorization-minimization algorithm that leads to multiplicative updates, which differ from standard heuristic multiplicative updates by a β-dependent power exponent. The monotonicity of the heuristic algorithm can, however, be proven for β ∈ (0, 1) using the proposed auxiliary function. Then we introduce the concept of the majorization-equalization (ME) algorithm, which produces updates that move along constant level sets of the auxiliary function and lead to larger steps than MM. Simulations on synthetic and real data illustrate the faster convergence of the ME approach. The letter also describes how the proposed algorithms can be adapted to two common variants of NMF: penalized NMF (when a penalty function of the factors is added to the criterion function) and convex NMF (when the dictionary is assumed to belong to a known subspace).},
    issn = {0899-7667},
    doi = {10.1162/NECO_a_00168},
    url = {https://doi.org/10.1162/NECO_a_00168},
    eprint = {https://direct.mit.edu/neco/article-pdf/23/9/2421/852282/neco_a_00168.pdf},
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

::: turftopic.models.bow.lda.LDA
