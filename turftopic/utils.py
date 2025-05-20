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
