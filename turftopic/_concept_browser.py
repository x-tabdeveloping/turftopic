import tempfile
import time
import webbrowser
from random import shuffle
from typing import Literal

import numpy as np

from turftopic._figure import HTMLFigure
from turftopic.utils import sanitize_for_html


def open_html(html: str):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        url = "file://" + f.name
        f.write(html)
        time.sleep(1.0)
        webbrowser.open(url)


COLOR_PALETTE = [
    "rgba(116,221,201, 0.25)",
    "rgba(235,171,204, 0.25)",
    "rgba(181,217,160, 0.25)",
    "rgba(197,179,239, 0.25)",
    "rgba(222,202,142, 0.25)",
    "rgba(160,188,233, 0.25)",
    "rgba(234,177,158, 0.25)",
    "rgba(129,209,231, 0.25)",
    "rgba(207,198,216, 0.25)",
    "rgba(186,213,195, 0.25)",
]


def create_bar_plot(seed_id, topic_names, topic_sizes, colors: list[str]):
    topic_sizes = [int(size) for size in topic_sizes]
    plot_js = """
    const yValue{seed_id} = {topic_sizes};
    const data{seed_id} = [
        {{
            x: {topic_names},
            y: yValue{seed_id},
            type: "bar",
            hoverinfo: 'none',
            textposition: 'auto',
            text: yValue{seed_id}.map(n => `n=${{n}}`).map(String),
            marker: {{
                color: {colors},
                line: {{
                color: 'black',
                width: 1
                }}
            }}
        }}
    ];
    const layout{seed_id} = {{
        font: {{family: "Merriweather, sans-serif", size: 16}},
        margin: {{t: 0, l: 30, b:15, r: 30}},
        height: 200,
        width: 600,
        yaxis: {{
            showticklabels: false
        }},
        xaxis: {{
            showticklabels: false
        }},
    }};
    Plotly.newPlot("plot-{seed_id}", data{seed_id}, layout{seed_id})
    """.format(
        topic_names=str(topic_names),
        topic_sizes=str(topic_sizes),
        colors=str(colors),
        seed_id=str(seed_id),
    )
    res = """
    <div id="plot-{seed_id}" class="plot"></div>
    <script>
    {plot_js}
    </script>
    """.format(
        plot_js=plot_js,
        seed_id=seed_id,
    )
    return res


def render_cards(
    seed_id,
    topic_names,
    keywords,
    topic_descriptions,
    topic_sizes,
    colors: list[str],
):
    # Adding the ID if it doesn't already start like that
    topic_names = [
        f"{i} - {name}"
        for i, name in enumerate(topic_names)
        if not name.startswith(str(i))
    ]
    res = create_bar_plot(seed_id, topic_names, topic_sizes, colors)
    for i, (name, keys, desc, color) in enumerate(
        zip(
            topic_names,
            keywords,
            topic_descriptions,
            colors,
        ),
    ):
        desc = sanitize_for_html(desc)
        name = sanitize_for_html(name)
        res += """
        <div class="card" style="background-color: {color};", id="card-{seed_id}-{i}">
        <h3>{name}</h3>
        <p><b>Keywords: </b><i>{keywords}</i></p>
        <p><b>Description: </b>{description}<p>
        </div>
        """.format(
            name=name,
            keywords=", ".join(keys),
            description=desc,
            i=i,
            color=color,
            seed_id=seed_id,
        )
    return res


def prep_document(doc: str, max_chars: int = 900):
    if len(doc) > max_chars:
        doc = doc[: max_chars - 3] + "..."
    doc = sanitize_for_html(doc)
    return doc


def render_documents(top_documents: list[list[str]], colors: list[str]):
    documents = []
    labels = []
    # Flattening the array out
    for topic_id, _top in enumerate(top_documents):
        documents.extend(_top)
        labels.extend([topic_id] * len(_top))
    indices = list(range(len(documents)))
    shuffle(indices)
    res = ""
    for i_doc in indices:
        res += """
        <div class="document" style="background-color: {bgcolor};">
        {document}
        </div>
        """.format(
            document=prep_document(documents[i_doc]),
            bgcolor=colors[labels[i_doc]],
        )
    return res


def render_widget(
    seeds: list[str],
    topic_names: list[list[str]],
    keywords: list[list[list[str]]],
    topic_descriptions: list[list[str]],
    topic_sizes: list[np.ndarray],
    top_documents: list[list[str]],
) -> str:
    n_topics = 0
    for names in topic_names:
        n_topics += len(names)
    colors = list(COLOR_PALETTE)
    # Colors will loop back when we run out of them
    while len(colors) <= n_topics:
        colors.extend(COLOR_PALETTE)
    res = """<div class="column">\n"""
    res += """
    <div class="button-container">
    <div class="floating-label">Seed Phrases</div>
    """
    for seed_id, seed in enumerate(seeds):
        res += """
        <button id="button-{seed_id}" class="box model-switcher" style="background-color: {bgcolor};"><b>{seed}</b></button>
        """.format(
            bgcolor="rgba(160,188,233, 0.4)" if seed_id == 0 else "white",
            seed_id=seed_id,
            seed=str(seed),
        )
    res += "</div>\n"
    color_start = 0
    for seed_id, seed in enumerate(seeds):
        n_topics = len(topic_names[seed_id])
        seed_colors = colors[color_start : color_start + n_topics]
        color_start += n_topics
        container = """
        <div id="topic-container-{seed_id}" class="box topic-container" style="display: {visibility}">
        <div class="floating-label">Concepts</div>
            {content}
        </div>
        """.format(
            content=render_cards(
                seed_id,
                topic_names[seed_id],
                keywords[seed_id],
                topic_descriptions[seed_id],
                topic_sizes[seed_id],
                seed_colors,
            ),
            visibility="block" if seed_id == 0 else "none",
            seed_id=seed_id,
        )
        res += container
    res += "\n</div>\n"
    color_start = 0
    for seed_id, seed in enumerate(seeds):
        n_topics = len(topic_names[seed_id])
        seed_colors = colors[color_start : color_start + n_topics]
        color_start += n_topics
        res += """
        <div id="document-viewer-{seed_id}" class="box document-viewer" style="display: {visibility};">
        <div class="floating-label">Example Documents</div>
        """.format(
            seed_id=seed_id,
            visibility="flex" if seed_id == 0 else "none",
        )
        res += render_documents(top_documents[seed_id], seed_colors)
        res += "\n</div>\n"
    res += """
    <script>
    const modelSwitchers = Array.from(document.getElementsByClassName('model-switcher'))
    modelSwitchers.forEach(button => {
        console.log("Found this button")
        button.addEventListener('click', function(event) {
            const modelId = button.id.split("-")[1];
            const containers = Array.from(document.getElementsByClassName('topic-container'))
            containers.forEach(container => {
                const containerId = container.id.split("-")[2];
                if (containerId == modelId) {
                    container.style = "display: block";
                } else {
                    container.style = "display: none";
                }
            })
            const documentViewers = Array.from(document.getElementsByClassName('document-viewer'))
            documentViewers.forEach(container => {
                const containerId = container.id.split("-")[2];
                if (containerId == modelId) {
                    container.style = "display: flex";
                } else {
                    container.style = "display: none";
                }
            })
            modelSwitchers.forEach(button => {
                button.style = "background-color: white;";
            })
            button.style = "background-color: rgba(160,188,233, 0.4);";
        });
    });
    </script>
    """
    return res


STYLE = """
body {
    font-family: "Merriweather", Times New Roman;
}
.topic-container {
    max-width: 600px;
    overflow-y: auto;
    overflow-x: hidden;
}
.floating-label {
    padding: 10px;
    position: fixed;
    color: white;
    background-color: black;
    border-radius: 10px;
    z-index: 30;
    width: fit-content;
    margin-top: -30px;
    margin-left: -30px;
}
.card {
    padding-top: 2px;
    padding-bottom: 2px;
    padding-left: 15px;
    padding-right: 15px;
    margin: 5px;
    margin-top: 10px;
    margin-bottom: 10px;
    background-color: #E6F3FF;
    border: solid;
    box-shadow: 0px 0px 1px 1px rgba(0,0,0,0.1);
    border-color: #999999;
    border-width: 1px;
    border-radius: 5px;
    border-color: black;
}
.button-container {
    padding: 10px;
    margin: 30px;
    margin-bottom: 5px;
    background-color: "white";
    border-radius: 5px;
    box-shadow: 0px 0px 1px 1px rgba(0,0,0,0.1);
    align-self: stretch;
    flex-grow: 1;
    flex-shrink: 1;
    display: flex;
    flex-direction: row;
    max-width: 600px;
}
.box {
    padding: 10px;
    margin: 30px;
    background-color: "white";
    border-radius: 5px;
    box-shadow: 0px 0px 1px 1px rgba(0,0,0,0.1);
    align-self: stretch;
    flex-grow: 0;
    flex-shrink: 1;
}
.document {
    padding: 20px;
    margin: 5px;
    border-radius: 5px;
    border: solid;
    box-shadow: 0px 0px 1px 1px rgba(0,0,0,0.1);
    border-color: #999999;
    border-width: 1px;
    max-height: 150px;
    flex-shrink: 0;
    text-align: left;
    text-overflow: ellipsis;
    font-style: italic;
    overflow: hidden;
}
.model-switcher {
    font-size: 16px;
    font-family: "Merriweather", Times New Roman;
    display: flex;
    flex-grow: 1;
    align-items: center;
    align-content: center;
    border: solid;
    border-color: #999999;
    border-width: 1px;
    border-radius: 5px;
    background-color: white;
    margin: 5px;
    color: black;
    padding: 10px;
    justify-content: left;
    text-decoration: none;
}
.model-switcher:hover {
    border-color: black;
}
.document-viewer {
    flex-basis: 600px;
    display: flex;
    justify-content: flex-start;
    flex-direction: column;
    overflow-y: scroll;
    overflow-x: hidden;
}
#container {
    display: flex;
    flex-direction: row;
    flex-grow: 0;
    justify-content: center;
    align-items: stretch;
    align-content: stretch;
    max-height: 1000px;
}
.column {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex-basis: fit-content;
}
"""
HTML_WRAPPER = """
<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <script src="https://cdn.plot.ly/plotly-3.1.0.min.js" charset="utf-8"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Merriweather">
    <style>
    {style}
    </style>
  </head>
  <body>
    <div id="container">
    {body_content}
    </div>
  </body>
</html>
"""


def create_browser(
    seeds: list[str],
    topic_names: list[list[str]],
    keywords: list[list[list[str]]],
    topic_descriptions: list[list[str]],
    topic_sizes: np.ndarray,
    top_documents: list[list[str]],
) -> HTMLFigure:
    """Creates a concept browser figure with which you can investigate concepts related to different seeds.

    Parameters
    ----------
    seeds: list[str]
        Seed phrases used for the analysis.
    topic_names: list[list[str]]
        Names of the topics for each of the seed phrases.
    keywords: list[list[list[str]]]
        Keywords for each of the topics for each seed.
    topic_descriptions: list[list[str]]
        Descriptions of the topics for each of the seed phrases.
    topic_sizes: np.ndarray
        Sizes of the topics for each seed, preferably number of documents.
    top_documents: list[list[str]]
        Top documents for each of the topics for each seed.

    Returns
    -------
    HTMLFigure
        Interactive HTML figure that you can either display or save.
    """
    html = HTML_WRAPPER.format(
        style=STYLE,
        body_content=render_widget(
            seeds,
            topic_names,
            keywords,
            topic_descriptions,
            topic_sizes,
            top_documents,
        ),
    )
    return HTMLFigure(html)
