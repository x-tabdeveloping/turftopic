"""I took this trick from embetter, it's for letting
the user know they have to install dependencies when they try
to import something that needs optional ones."""


class NotInstalled:
    """
    This object is used for optional dependencies.
    If a backend is not installed we replace the
    estimator with this object.
    """

    def __init__(self, tool, dep):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        msg += f"pip install {self.dep}\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)
