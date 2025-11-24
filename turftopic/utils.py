import csv
import io

import numpy as np
from sklearn.preprocessing import label_binarize


def safe_binarize(y, classes) -> np.ndarray:
    """label_binarize, but its behaviour stays consistent when the labels are {0,1}"""
    if set(classes) == {0, 1}:
        binary_vec = np.squeeze(label_binarize(y, classes=classes))
        negative_vec = np.zeros(binary_vec.shape[0], dtype=binary_vec.dtype)
        negative_vec[binary_vec == 0] = 1
        # Concatenating
        onehot = np.stack((binary_vec, negative_vec))
        return onehot.T
    else:
        return label_binarize(y, classes=classes)


def export_table(
    table: list[list[str]],
    format="csv",
) -> str:
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        for row in table:
            writer.writerow(row)
        return output.getvalue()
    if format == "latex":
        columns, *rows = table
        n_columns = len(columns)
        latex_column_format = " ".join(["l"] * n_columns)
        latex_columns = " & ".join(columns) + "\\\\"
        latex_rows = [" & ".join(row) + "\\\\" for row in rows]
        latex_table = [
            "\\begin{center}",
            f"\\begin{{tabular}}{{ {latex_column_format} }}",
            latex_columns,
            "\\hline",
            *latex_rows,
            "\\end{tabular}",
            "\\end{center}",
        ]
        return "\n".join(latex_table)
    if format == "markdown":
        columns, *rows = table
        n_columns = len(columns)
        separator = ["-"] * n_columns
        markdown_rows = [
            "|" + "|".join([" " + value + " " for value in row]) + "|"
            for row in [columns, separator, *rows]
        ]
        return "\n".join(markdown_rows)
    raise ValueError(
        f"Format '{format}' not supported for tables, please use 'markdown', 'latex' or 'csv'"
    )


def sanitize_for_html(text: str) -> str:
    """Sanitizes strings so they can be put into JS or HTML strings"""
    # Escaping special characters
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
    # Removing unnecessary whitespace
    text = " ".join(text.split())
    return text


def confidence_ellipse(mean, cov, n_std=1, size=100):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )
    x_scale = np.sqrt(cov[0, 0]) * n_std
    y_scale = np.sqrt(cov[1, 1]) * n_std
    x_mean, y_mean = mean
    translation_matrix = np.tile(
        [x_mean, y_mean], (ellipse_coords.shape[0], 1)
    )
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix)
        + translation_matrix
    )
    path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    path += " Z"
    return path
