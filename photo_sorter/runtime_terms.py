"""One-time acceptance of terms for bundled Microsoft distributable code."""
import hashlib
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from .app_paths import bundled_resource


def accept_runtime_terms(base_dir: Path) -> bool:
    terms = bundled_resource("RUNTIME_TERMS.txt").read_text(encoding="utf-8")
    version = hashlib.sha256(terms.encode("utf-8")).hexdigest()
    record = base_dir / "アプリデータ/runtime_terms.json"
    try:
        if json.loads(record.read_text(encoding="utf-8")).get("accepted_sha256") == version:
            return True
    except (OSError, ValueError, AttributeError):
        pass
    accepted = False
    window = tk.Tk()
    window.title("EventPhotoSorter — 初回の利用条件確認")
    window.geometry("760x590")
    window.minsize(600, 450)
    ttk.Label(window, text="同梱Microsoftランタイムの利用条件", font=("Yu Gothic UI", 14, "bold")).pack(anchor="w", padx=16, pady=12)
    body = ScrolledText(window, wrap="word", font=("Yu Gothic UI", 10), padx=12, pady=8)
    body.pack(fill="both", expand=True, padx=16)
    body.insert("1.0", terms)
    body.configure(state="disabled")
    buttons = ttk.Frame(window, padding=16)
    buttons.pack(fill="x")

    def agree():
        nonlocal accepted
        accepted = True
        try:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"accepted_sha256": version}) + "\n", encoding="utf-8")
        except OSError:
            messagebox.showinfo("同意の保存", "今回の同意で起動します。保存できなかったため、次回も確認が表示されます。", parent=window)
        window.destroy()

    ttk.Button(buttons, text="同意して起動", command=agree).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="同意せず終了", command=window.destroy).pack(side="right")
    window.mainloop()
    return accepted
