# Topeax

Topeax is a probabilistic topic model based on the Peax clustering model, which finds topics based on peaks in point density in the embedding space. The model can recover the number of topics automatically.

In the following example I run a Topeax model on the BBC News corpus, and plot the steps of the algorithm to inspect how our documents have been clustered and why:

```python
# pip install datasets, plotly
from datasets import load_dataset
from turftopic import Topeax

ds = load_dataset("gopalkalpande/bbc-news-summary", split="train")
topeax = Topeax(random_state=42)
doc_topic = topeax.fit_transform(list(ds["Summaries"]))

fig = topeax.plot_steps(hover_text=[text[:200] for text in corpus])
fig.show()
```

<figure>
  <iframe src="../images/topeax_steps.html", title="", style="height:620px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 1: Steps in a Topeax model fitted on BBC News displayed on an interactive graph. </figcaption>
</figure>

```python
topeax.print_topics()
```


| Topic ID | Highest Ranking |
| - | - |
| 0 | mobile, microsoft, digital, technology, broadband, phones, devices, internet, mobiles, computer |
| 1 | economy, growth, economic, deficit, prices, gdp, inflation, currency, rates, exports |
| 2 | profits, shareholders, shares, takeover, shareholder, company, profit, merger, investors, financial |
| 3 | film, actor, oscar, films, actress, oscars, bafta, movie, awards, actors |
| 4 | band, album, song, singer, concert, rock, songs, rapper, rap, grammy |
| 5 | tory, blair, labour, ukip, mps, minister, election, tories, mr, ministers |
| 6 | olympic, tennis, iaaf, federer, wimbledon, doping, roddick, champion, athletics, olympics |
| 7 | rugby, liverpool, england, mourinho, chelsea, premiership, arsenal, gerrard, hodgson, gareth |

## How does Topeax work?

The Topeax algorithm, similar to clustering topic models consists of two consecutive steps.
One of them discovers the underlying clusters in the data, the other one estimates term importance scores for each topic in the corpus.

<br>
<figure>
  <img src="../images/peax.png" width="100%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Figure 2: Schematic overview of the steps of the Peax clustering algorithm</figcaption>
</figure>

### 1. Clustering


Documents embeddings first get projected into two-dimensional space using t-SNE.
In order to identify clusters, we first calculate a Kernel Density Estimate over the embedding space,
then find local maxima in the KDE by grid approximation.
When we discover local maxima (peaks), we assume these to be cluster means.
Cluster density is then approximated with a Gaussian Mixture, where we fix means to the density peaks and then use expectation-maximization to fit the rest of the parameters. (see Figure 2)
Documents are then assigned to the component with the highest responsibility:

$$\hat{z_d} = arg max_k (r_{kd}); r_{kd}=p(z_k=1 | \hat{x}_d)$$

where $z_d$ is the cluster label for document $d$, $r_{kd}$ is the responsibility of component $k$ for document $d$ and $\hat{x}_d$ is the 2D embedding of document $d$.

### 2. Term Importance Estimation

Topeax uses a combined semantic-lexical term importance, which is the geometric mean of the NPMI method (see [Clustering Topic Models](clustering.md) for more detail) and a slightly modified centroid-based method.
The modified centroids are calculated like so:

$$t_k = \frac{\sum_d r_{kd} \cdot x_d}{\sum_d r_{kd}}$$

where $t_k$ is the embedding of topic $k$ and $x_d$ is the embedding of document $d$.

## Visualization

Topeax has a number of plots available that can aid you when interpreting your results:

### Density Plots

One can plot the kernel density estimate on both a 2D and a 3D plot.

```python
topeax.plot_density()
```

<figure>
  <iframe src="../images/topeax_density.html", title="", style="height:620px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 2: Density contour plot of the Topeax model. </figcaption>
</figure>

```python
topeax.plot_density3d()
```

<figure>
  <iframe src="../images/topeax_density_3d.html", title="", style="height:620px;width:620px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 3: 3D Density Surface of the Topeax model. </figcaption>
</figure>

### Component Plots

You can also create a plot over the mixture components/clusters found by the model.

```python
topeax.plot_components()
```

<figure>
  <iframe src="../images/topeax_components.html", title="", style="height:620px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 4: Gaussian components estimated for the model. </figcaption>
</figure>

You can also create a datamapplot figure similar to clustering models:

```python
# pip install turftopic[datamapplot]
topeax.plot_components_datamapplot()
```

<figure>
  <iframe src="../images/topeax_components_datamapplot.html", title="", style="height:620px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 5: Datapoints colored by mixture components on a datamapplot. </figcaption>
</figure>

## API Reference

::: turftopic.models.topeax.Topeax

::: turftopic.models.topeax.Peax
