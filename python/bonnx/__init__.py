"""bonnx — view ONNX models in the browser.

    python -m bonnx model.onnx                 # serve on localhost and open the browser
    python -m bonnx model.onnx --out view.html # one self-contained HTML file

    import bonnx
    bonnx.show("model.onnx")                   # or bytes / onnx.ModelProto
    bonnx.save_html(model, "view.html")
    bonnx.display(model)                       # Jupyter
"""
from .api import display, save_html, show, stop_all

__version__ = "1.0.0"
__all__ = ["show", "save_html", "display", "stop_all", "__version__"]
