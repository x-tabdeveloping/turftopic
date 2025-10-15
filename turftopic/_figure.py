import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Union


class HTMLFigure:
    def __init__(self, html: str):
        self.html = html

    def show(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = Path(temp_dir).joinpath("fig.html")
            self.write_html(file_name)
            webbrowser.open("file://" + str(file_name.absolute()), new=2)
            time.sleep(2)

    def write_html(self, path: Union[str, Path]):
        path = Path(path)
        with path.open("w") as out_file:
            out_file.write(self.html)

    def __repr_html__(self):
        return self.html
