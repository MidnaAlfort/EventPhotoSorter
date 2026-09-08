"""Manual region review; never alters image pixels or automatically marks a proposal reviewed."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .image_utils import open_rgb, resized_for_model
from .regions import read_reviewed_regions, save_reviewed_regions


def edit_regions(parent, path: Path):
    from PIL import ImageTk
    try:
        source = open_rgb(path)
        preview = resized_for_model(source, 780)
        source.close()
    except (OSError, ValueError) as exc:
        messagebox.showerror("画像を開けません", str(exc), parent=parent)
        return
    dialog = tk.Toplevel(parent)
    dialog.title("人物領域の確認")
    dialog.transient(parent)
    dialog.grab_set()
    ttk.Label(dialog, text="頭・胴体が見える全員を1人ずつドラッグで囲んでください。鏡・ポスターは除外します。",
              wraplength=780, padding=10).pack()
    width, height = preview.size
    photo = ImageTk.PhotoImage(preview)
    canvas = tk.Canvas(dialog, width=width, height=height, highlightthickness=0)
    canvas.pack(padx=10)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo
    scene = read_reviewed_regions(path)
    boxes = list(scene.boxes) if scene else []
    state = {}
    status = tk.StringVar()
    def redraw():
        canvas.delete("box")
        for index, b in enumerate(boxes, 1):
            canvas.create_rectangle(b[0]*width, b[1]*height, b[2]*width, b[3]*height,
                                    outline="#00ff99", width=3, tags="box")
            canvas.create_text(b[0]*width+10, b[1]*height+10, text=str(index), fill="#00ff99", tags="box")
        status.set(f"指定した人物：{len(boxes)}人。0人で保存すると人物なしの確認結果になります。")
    def start(event):
        state["start"] = (max(0, min(width, event.x)), max(0, min(height, event.y)))
    def drag(event):
        if "start" in state:
            canvas.delete("draft")
            canvas.create_rectangle(*state["start"], event.x, event.y, outline="#ffff00", tags="draft")
    def end(event):
        if "start" not in state:
            return
        x0, y0 = state.pop("start")
        x1, y1 = max(0, min(width, event.x)), max(0, min(height, event.y))
        canvas.delete("draft")
        if abs(x1-x0) >= 5 and abs(y1-y0) >= 5:
            boxes.append([min(x0,x1)/width, min(y0,y1)/height, max(x0,x1)/width, max(y0,y1)/height])
        redraw()
    canvas.bind("<ButtonPress-1>", start)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", end)
    ttk.Label(dialog, textvariable=status, padding=8).pack()
    buttons = ttk.Frame(dialog, padding=8)
    buttons.pack(fill="x")
    def undo():
        if boxes:
            boxes.pop()
        redraw()
    def save():
        try:
            save_reviewed_regions(path, boxes)
        except (OSError, ValueError) as exc:
            messagebox.showerror("保存できません", str(exc), parent=dialog)
            return
        dialog.destroy()
    ttk.Button(buttons, text="最後の枠を取り消す", command=undo).pack(side="left")
    ttk.Button(buttons, text="全員の領域を確認して保存", command=save).pack(side="right")
    redraw()
