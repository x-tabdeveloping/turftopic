import re
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import scale

from turftopic.utils import sanitize_for_html

CUSTOM_CSS = """
.row {
    display : flex;
    align-items : center;
}
.box {
    height:10px;
    width:10px;
    border-radius:2px;
    margin-right:7px;
    margin-top:5px;
    margin-bottom:5px;
    padding:0px 0 1px 0;
    text-align:center;
    color: white;
    font-size: 14px;
    cursor: pointer;
}
.description {
    position: absolute;
    bottom: 0;
    left: 0;
    text-align: justify;
    padding: 20px;
    margin: 15px;
    max-width: 40%;
    background-color: #ffffff;
    border-radius:15px;
    box-shadow: 0px 0px 5px 5px rgba(0,0,0,0.05);
}
h3 {
    margin: 0px;
    margin-bottom: 4px;
    padding: 0px;
}
p {
    margin: 0px;
    padding: 0px;
}
#legend {
        position: absolute;
        top: 0;
        right: 0;
    }
#title-container {
    max-width: 75%;
}
.topic-tree-header {
    display: none;
}
.topic-tree-close-btn {
    display: none;
}
.bullet {
    padding: 0px;
}
#topic-tree-body > .nested {
    display: block;
    padding: 0px;
    margin-right: 10px;
}
#topic-tree-body > li {
    margin: 0px;
}
#topic-tree {
    padding: 5px;
}
#topic-tree-body {
    padding: 0px;
}
#topic-keywords {
    font-style: italic;
    margin-bottom: 2px;
    text-align: left;
}
#topic-percent {
    margin-top: 10px;
    margin-right: 10px;
}
#percent-message {
    margin-top: 10px;
}
"""


def _labels_to_indices(labels, classes):
    n_classes = len(classes)
    class_to_index = dict(zip(classes, np.arange(n_classes)))
    return np.array([class_to_index[label] for label in labels])


def build_datamapplot(
    coordinates: np.ndarray,
    topic_names: list[str],
    labels: np.ndarray,
    classes: np.ndarray,
    top_words: Optional[list[list[str]]] = None,
    topic_descriptions: Optional[list[str]] = None,
    font_family: str = "Merriweather",
    enable_topic_tree=True,
    topic_tree_kwds={
        "color_bullets": True,
    },
    cluster_boundary_polygons=False,
    cluster_boundary_line_width=6,
    polygon_alpha=2,
    **kwargs,
):
    """Builds a Turftopic interactive datamapplot.

    Parameters
    ----------
    coordinates: np.ndarray
        X and Y coordinates of datapoints.
    topic_names: list[str]
        Names of topics in the model.
    labels: np.ndarray
        Topic labels for each datapoint (topic_id for each point)
    classes: np.ndarray
        List of topic IDs in the model.
    top_words: list[list[str]], optional
        List of top keywords for each topic.
    topic_descriptions: list[str], optional
        List of descriptions for the given topics.
    """
    try:
        import datamapplot
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "You need to install datamapplot to be able to use plot_clusters_datamapplot()."
        ) from e
    coordinates = scale(coordinates) * 4
    indices = _labels_to_indices(labels, classes)
    labels = np.array(topic_names)[indices]
    if -1 in classes:
        i_outlier = np.where(classes == -1)[0][0]
        kwargs["noise_label"] = topic_names[i_outlier]
    # Calculating how much of the corpus is made up of a topic
    percentages = []
    for label in topic_names:
        percentages.append(100 * np.sum(labels == label) / len(labels))
    # Sanitizing the names so they don't mess up the HTML
    topic_names = [sanitize_for_html(name) for name in topic_names]
    custom_js = ""
    custom_js += "const nameToPercent = new Map();\n"
    for name, percent in zip(topic_names, percentages):
        custom_js += 'nameToPercent.set("{name}", {percent});\n'.format(
            name=name,
            percent=percent,
        )
    custom_js += "const nameToDesc = new Map();\n"
    if topic_descriptions is not None:
        topic_descriptions = [
            sanitize_for_html(desc) for desc in topic_descriptions
        ]
        for topic_id, name, desc in zip(
            classes, topic_names, topic_descriptions
        ):
            custom_js += 'nameToDesc.set("{name}", "{desc}");\n'.format(
                name=name,
                desc=desc,
            )
    custom_js += "const nameToKeywords = new Map();\n"
    if top_words is not None:
        for name, words in zip(topic_names, top_words):
            custom_js += (
                'nameToKeywords.set("{name}", "{keywords}");\n'.format(
                    name=name, keywords=", ".join(words)
                )
            )
    custom_html = ""
    custom_html += """
<div class="description">
    <h3 id="topic-name">{topic_name}</h3>
    <p id="topic-keywords">{topic_keywords}</p>
    <progress id="topic-percent" value="{percentage:.2f}" max="100"></progress>
    <b id="percent-message">{percentage:.2f}% of all documents</b>
    <hr>
    <p id="topic-description">{topic_description}</p>
</div>
    """.format(
        topic_name=topic_names[1],
        topic_description=(
            topic_descriptions[1] if topic_descriptions is not None else ""
        ),
        topic_keywords=(
            "Keywords: "
            + (", ".join(top_words[1]) if top_words is not None else "")
        ),
        percentage=percentages[1],
    )
    custom_js += """
setTimeout(function(){
    const labelNodes = Array.from(document.getElementsByClassName('topic-tree-label'))
    labelNodes.forEach(button => {
        console.log("Found this button")
        button.addEventListener('click', function(event) {
            const topicName = document.getElementById("topic-name");
            const topicDesc = document.getElementById("topic-description");
            const topicKeywords = document.getElementById("topic-keywords");
            const topicPercent = document.getElementById("topic-percent");
            const percentMessage = document.getElementById("percent-message");
            const name = button.textContent.replace(/[\\n\\r\\t]/gm, " ");
            topicName.textContent = name;
            const percent = nameToPercent.get(name);
            topicPercent.value = percent;
            percentMessage.textContent = percent.toFixed(2) + "% of all documents";
            const description = nameToDesc.get(name);
            console.log(description)
            if (description) {
                topicDesc.textContent = description;
            } else {
                topicDesc.textContent = "";
            }
            const keywords = nameToKeywords.get(name);
            console.log(keywords)
            if (keywords) {
                topicKeywords.textContent = "Keywords: " + keywords;
            } else {
                topicKeywords.textContent = "";
            }
        });
    });
}, 200);
    """
    plot = datamapplot.create_interactive_plot(
        coordinates,
        labels,
        font_family=font_family,
        logo="https://x-tabdeveloping.github.io/turftopic/images/logo.svg",
        logo_width=80,
        enable_topic_tree=enable_topic_tree,
        topic_tree_kwds=topic_tree_kwds,
        cluster_boundary_polygons=cluster_boundary_polygons,
        cluster_boundary_line_width=cluster_boundary_line_width,
        polygon_alpha=polygon_alpha,
        custom_css=CUSTOM_CSS,
        custom_html=custom_html,
        custom_js=custom_js,
        **kwargs,
    )

    def show_fig():
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir).joinpath("fig.html")
            plot.save(file_name)
            webbrowser.open("file://" + str(file_name.absolute()), new=2)
            time.sleep(2)

    plot.show = show_fig
    plot.write_html = plot.save
    return plot
