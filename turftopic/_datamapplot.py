import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import scale


def _labels_to_indices(labels, classes):
    n_classes = len(classes)
    class_to_index = dict(zip(classes, np.arange(n_classes)))
    return np.array([class_to_index[label] for label in labels])


def build_datamapplot(
    coordinates: np.ndarray,
    topic_names: list[str],
    labels: np.ndarray,
    classes: np.ndarray,
    topic_descriptions: Optional[list[str]] = None,
    font_family: str = "Merriweather",
    enable_topic_tree=True,
    topic_tree_kwds={
        "color_bullets": True,
    },
    cluster_boundary_polygons=True,
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
    if topic_descriptions is None:
        topic_descriptions = [""] * len(classes)

    # Sanitizing the names so they don't mess up the HTML
    topic_names = [name.replace('"', "'") for name in topic_names]
    topic_descriptions = [
        desc.replace('"', "'") for desc in topic_descriptions
    ]
    custom_css = """
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
    padding: 15px;
    margin: 10px;
    max-width: 40%;
    background-color: #ffffff;
    border-radius:15px;
    box-shadow: 0px 0px 5px 5px rgba(0,0,0,0.05);
}
h3 {
    margin: 0px;
    margin-bottom: 8px;
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
    """
    custom_html = ""
    custom_html += """
<div class="description">
    <h3 id="topic-name">{topic_name}</h3>
    <p id="topic-description">{topic_description}</p>
</div>
    """.format(
        topic_name=topic_names[1].replace('"', "'"),
        topic_description=topic_descriptions[1].replace('"', "'"),
    )
    custom_js = ""
    custom_js += """
const nameToDesc = new Map();
    """
    for topic_id, name, desc in zip(classes, topic_names, topic_descriptions):
        custom_js += """
nameToDesc.set("{name}", "{desc}");
        """.format(
            name=name,
            desc=desc,
        )
    custom_js += """
setTimeout(function(){
    const labelNodes = Array.from(document.getElementsByClassName('topic-tree-label'))
    labelNodes.forEach(button => {
        console.log("Found this button")
        button.addEventListener('click', function(event) {
            const topicName = document.getElementById("topic-name");
            const topicDesc = document.getElementById("topic-description");
            console.log(button.textContent);
            console.log(nameToDesc.get(button.textContent));
            const name = button.textContent.replace(/[\\n\\r\\t]/gm, " ")
            topicName.textContent = name;
            topicDesc.textContent = nameToDesc.get(name);
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
        custom_css=custom_css,
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
