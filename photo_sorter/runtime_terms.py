"""One-time acceptance of terms for bundled Microsoft distributable code."""
import hashlib
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from .app_paths import bundled_resource


def show_full_terms(parent, base_dir: Path, terms: str):
    """Read the conditions without recording or implying acceptance."""
    detail = tk.Toplevel(parent)
    detail.title("EventPhotoSorter — ライセンス全文")
    detail.geometry("760x560")
    detail.minsize(560, 380)
    detail.transient(parent)
    body = ScrolledText(detail, wrap="word", font=("Yu Gothic UI", 10), padx=12, pady=8)
    body.pack(fill="both", expand=True, padx=16, pady=(16, 0))
    documents = [terms]
    for relative in ("LICENSE", "THIRD_PARTY_NOTICES.md", "runtime/msvc/VS2022-LICENSE.txt"):
        candidates = (base_dir / "licenses" / relative, base_dir / relative)
        path = next((p for p in candidates if p.is_file()), None)
        if path:
            documents.append("\n\n" + "─" * 30 + "\n" + relative + "\n\n" + path.read_text(encoding="utf-8-sig"))
    body.insert("1.0", "".join(documents))
    body.configure(state="disabled")
    buttons = ttk.Frame(detail, padding=16)
    buttons.pack(fill="x")

    def open_licenses():
        try:
            os.startfile(str(base_dir / "licenses"))
        except OSError:
            messagebox.showinfo("ライセンス", "ライセンスフォルダを開けませんでした。\n" + str(base_dir / "licenses"), parent=detail)

    ttk.Button(buttons, text="各部品のライセンスフォルダを開く", command=open_licenses).pack(side="left")
    ttk.Button(buttons, text="閉じる", command=detail.destroy).pack(side="right")


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
    from .window_icon import set_window_icon
    set_window_icon(window)
    window.title("EventPhotoSorter — 初回の利用条件確認")
    window.resizable(False, False)
    content = ttk.Frame(window, padding=20)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="はじめに、ライセンスをご確認ください", font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(0, 12))
    ttk.Label(content, text="このアプリと同梱部品の利用条件に同意してご利用ください。\n"
              "アプリ本体はMITライセンス、同梱部品にはそれぞれの\n利用条件が適用されます。",
              font=("Yu Gothic UI", 10), justify="left", wraplength=460).pack(anchor="w")
    ttk.Label(content, text="全文は下のボタンから確認できます。次回からは表示しません。",
              font=("Yu Gothic UI", 9), foreground="#555555", wraplength=460).pack(anchor="w", pady=(12, 0))
    buttons = ttk.Frame(content)
    buttons.pack(fill="x", pady=(18, 0))

    def agree():
        nonlocal accepted
        accepted = True
        try:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"accepted_sha256": version}) + "\n", encoding="utf-8")
        except OSError:
            messagebox.showinfo("同意の保存", "今回の同意で起動します。保存できなかったため、次回も確認が表示されます。", parent=window)
        window.destroy()

    ttk.Button(buttons, text="全文を読む", command=lambda: show_full_terms(window, base_dir, terms)).pack(side="left")
    ttk.Button(buttons, text="同意して起動", command=agree).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="終了", command=window.destroy).pack(side="right")
    window.mainloop()
    return accepted
