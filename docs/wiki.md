# Wiki Analyzer

Not all users have access to an LLM API or can afford to run LLMs on their own hardware.
This is why we've added a light-weight analyzer that is retrieval-based, rather than relying on text generation.
This allows the topic analyzer to work with the same language model that was used for fitting the topic model.

The `WikiAnalyzer` works in the following steps:

 1. It searches Wikipedia for articles using the top N keywords from a topic model.
 2. For each topic it produces a topic embedding from the average of top 10 keywords and documents.
 3. It retrieves the most similar articles to each topic.
 4. If the similarity crosses a certain threshold, it assigns the article's name to the topic.

```python
from sklearn.datasets import fetch_20newsgroups

from turftopic import SensTopic
from turftopic.analyzers.wiki import WikiAnalyzer

dataset = fetch_20newsgroups(subset="all", categories=["alt.atheism"])
corpus = dataset.data

t_model = SensTopic(
    random_state=42,
    encode_kwargs=dict(show_progress_bar=True),
    sparsity=5.0,
)
embeddings = t_model.encode_documents(corpus)
t_model.fit(corpus, embeddings=embeddings)

analyzer = WikiAnalyzer(t_model, similarity_threshold=0.3)
t_model.rename_topics(analyzer)
t_model.print_topics()
```

|    | Topic Name         | Highest Ranking                                                                                                     |
|---:|:-------------------|:--------------------------------------------------------------------------------------------------------------------|
|  0 | Omnipotence        | contradictions, contradiction, creationism, creation, omnipotent, belief, believing, contradictory, deity, believed |
|  1 | Capital punishment | genocide, punishments, murder, punishment, killing, punish, deaths, executed, kills, penalty                        |
|  2 | Morality           | morality, morals, moral, morally, ethical, immoral, societal, societally, objectively, justified                    |
|  3 |                 | amusing, responses, discussions, discussing, funny, disclaimer, newsgroups, policy, isn, offensive                  |
|  4 | Agnostic atheism   | atheism, atheist, atheists, atheistic, agnostics, agnostic, agnosticism, theists, secular, religious                |
|  5 | Gospel             | testament, gospel, theological, biblical, bible, verses, revelation, theology, christianity, verses_                |
|  6 | Quran              | islamic, muslim, islam, qur, muslims, koran, quran, allah, rushdie, rashid                                          |

## API Reference

:::turftopic.analyzers.wiki.WikiAnalyzer
