import csv
import io


def export_table(
    table: list[list[str]],
    format="csv",
) -> str:
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        for row in table:
            writer.writerow(row)
        return output.read()
    if format == "latex":
        columns, *rows = table
        n_columns = len(columns)
        latex_column_format = " ".join(["c"] * n_columns)
        latex_columns = " & ".join(columns) + "\\\\"
        latex_rows = [" & ".join(row) + "\\\\" for row in rows]
        latex_table = [
            f"\\begin{{tabular}}{{ {latex_column_format} }}",
            latex_columns,
            "\\hline",
            *latex_rows,
        ]
        return "\n".join(latex_table)
    if format == "markdown":
        columns, *rows = table
        n_columns = len(columns)
        separator = [" - ", " - ", " - "]
        markdown_rows = [
            "|" + "|".join(row) + "|" for row in [columns, separator, *rows]
        ]
        return "\n".join(markdown_rows)
    raise ValueError(
        f"Format '{format}' not supported for tables, please use 'markdown', 'latex' or 'csv'"
    )
