# Customer Dissatisfaction Analysis

When providing some service for customers as a company, it is valuable to have an overview of what customers are dissatisfied with.
A good place to start would be to analyze reviews of your product that have low scores.

In this tutorial we will use Turftopic to find topics in customer reviews of the Uber app on Google Play.
The dataset is [openly available on Kaggle](https://www.kaggle.com/datasets/kanchana1990/uber-customer-reviews-dataset-2024?resource=download), we will:

 - Build and train a Noun-phrase informed [KeyNMF](../KeyNMF.md) topic model on customer feedback
 - Learn about topic interpretation.
 - Investigate the prevalence of issues identified by customers.

## Installation

We will be using SpaCy for phrase extraction and Plotly for producing visualizations and pandas for data wrangling.
We will also have to install a SpaCy pipeline.

```python
pip install turftopic[spacy] plotly pandas
python -m spacy download en_core_web_sm
```

## Data Preparation

We will load the dataset, and extract the reviews with the lowest scores, to see what the most disappointed users take issue with.

```python
import pandas as pd

df = pd.read_csv("uber_reviews_without_reviewid.csv")
corpus = list(df["content"][df["score"] == 1])
```

Since the dataset contains reviews in Spanish, we will be using a multilingual encoder model:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = encoder.encode(corpus, show_progress_bar=True)
```

## Model Training

We will be using the [KeyNMF](../KeyNMF.md) model for topic discovery, as it is capable of hierarchically expanding topics if we need it.

!!! tip
    Using noun phrases, as it is done in this tutorial can greatly increase the interpretability, clarity and precision of your topic descriptions, but can make model training substantially slower, as sentences have to be parsed.
    See the [Vectorizers](../vectorizers.md) page for more detail.

```python
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

model = KeyNMF(
    # We train first with 10 topics, and can then expand the hierarchy if needed
    n_components=10,
    encoder=encoder,
    vectorizer=NounPhraseCountVectorizer("en_core_web_sm"),
    random_state=42,
)
doc_topic_matrix = model.fit_transform(corpus, embeddings=embeddings)
```

## Model Interpretation

Let us first examine the highest ranking phrases on each topic.

!!! tip
    For a more detailed discussion, see the [Model Interpretation](../model_interpretation.md) page in the documentation.

```python
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | driver, pickup, cancellation fee, car, money, fee, location, extra money, road, auto driver |
| 1 | app, apps, worst application, option, taxi, phone, support, ads, time, customer service |
| 2 | drivers, rides, company, time, trips, customer, riders, customers, car, apps |
| 3 | ride, time, rider, charge, destination, worst service, rides, emergency no rider, 2 rides, uber app |
| 4 | worst app, booking, customer support, waiting time, destination, rides, scam, worst customer support, senseless irresponsible app, traffic useless app |
| 5 | cab, cab driver, time, emergency, booking, auto, bad service, worst cab bookinv app, rapido, amount |
| 6 | account, money, customer support, refund, card, bank, permission, email, credit card, customers |
| 7 | price, prices, pricing, fees, traffic, predatory pricing, destination, higher price, discount, amount |
| 8 | trip, cash, amount, car, customer support, service, trips, bad experience, destination, cancelation fee |
| 9 | uber, service, payment, taxi, rides, customer service, worst service, uber app, customers, refund |

It seems that a substantial part of the bad reviews are related to the drivers, either because they are rude, people, or overcharge customers.
This indicates that the company should be wary of such behaviour and should implement measures to counteract these issues.
Other topics are related to customer support, or the quality of the app itself.


### Hierarchical Topic Expansion

Let us say that we are mostly interested in issues with customer service since we want to provide insights, about what could be improved.
We can investigate the topics related to this in more detail by expanding the model's hierarchy.

!!! tip
    For a more detailed discussion, see the [Hierarchical Modelling](../hierarchical.md) page in the documentation.

We can for instance investigate what subtopics can be found in Topic 9:
```python
# Accessing Topic 9 and dividing it to 3 subtopics
print(model.hierarchy[9].divide(3))
```
<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b style="color: magenta">9</b>: uber, service, payment, taxi, rides, customer service, worst service, uber app, customers, refund <br>
├── <b style="color: blue">0</b>: driver, ride, uber, payment, money, cab, cash, uber app, trip, refund <br>
├── <b style="color: blue">1</b>: app, taxi, the app prices journeys, airport, prices, inflated prices, scam, inflated price, customer service, uber <br>
└── <b style="color: blue">2</b>: drivers, uber, rides, service, fraud drivers, voucher payment, prices, frauds, uber wallet, payments <br>
</div>
</tt>

This tells us that issues with customer service are often related to refunds because of inflated prices or fraud.
We can also see that voucher payments and uber wallet are often associated with fraud.

### Topic Prevalence

We can check how prevalent each of these issues are by adding the occurrence of these topics up in the corpus, then plotting them.

```python
import plotly.express as px

topic_prevalence = doc_topic_matrix.sum(axis=0)
prev_df = pd.DataFrame({"prevalence": topic_prevalence, "name": model.topic_names})
fig = px.pie(prev_df, values="prevalence", names="name", title="Prevalence of topics in negative reviews.")
fig.show()
```

<center>
  <iframe src="../images/topic_pie.html", title="Plot of topic importance", style="height:500px;width:650px;padding:0px;border:none;"></iframe>
</center>



We can see that most complaints are made about their driver, as topics of this nature dominate most of the documents,
while the next most common issue is with the application.
Customer support is the least prevalent of issues in our dataset.
