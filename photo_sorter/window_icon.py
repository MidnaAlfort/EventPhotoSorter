"""Set application icons, including Windows title-bar and child windows."""
import sys
import tkinter as tk

from .app_paths import bundled_resource


def set_window_icon(window):
    if sys.platform == "win32":
        try:
            window.iconbitmap(default=str(bundled_resource("assets/app_icon.ico")))
            return
        except (tk.TclError, OSError):
            pass
    # Pillow normalizes PNG formats that Tcl/Tk's PNG reader cannot decode.
    from PIL import Image, ImageTk
    try:
        with Image.open(bundled_resource("assets/app_icon.png")) as source:
            rgba = source.convert("RGBA")
            window._window_icons = [ImageTk.PhotoImage(rgba.resize((size, size)), master=window)
                                    for size in (16, 32, 48, 64)]
        window.iconphoto(True, *window._window_icons)
    except (tk.TclError, OSError):
        pass
