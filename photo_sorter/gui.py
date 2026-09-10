from __future__ import annotations

import os
import json
import sys
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .app_paths import APP_NAME, APP_VERSION, application_dir, bundled_resource, load_app_environment, model_cache_dir, save_api_key
from .catalog import discover_reference_catalog, discover_reference_classes
from .image_utils import open_rgb, resized_for_model
from .reference_learning import add_confirmed_reference
from .costs import estimate_api_cost, format_cost_estimate
from .engine import PhotoSorter, validate_config
from .formatting import format_elapsed_time
from .models import ProgressEvent, SorterConfig


MODE_LABELS = {
    "ハイブリッド": "hybrid",
    "固定参考API": "api",
    "人物領域ハイブリッド": "regional",
    "ローカルのみ（人数判定なし）": "local",
}

SIMPLE_MODE_LABELS = {
    "ローカルのみ（人数判定なし）": "local",
    "人物領域ハイブリッド": "regional",
}
MODE_DESCRIPTIONS = {
    "local": "APIキー不要・API料金なし。参考画像と見比べて人物別に振り分けます。複数人数には対応しません。"
             "人数を判定しないため、複数人の写真が個人フォルダに入る場合があります。単独写真向けです。",
    "regional": "ChatGPTのAPIキーが必要です(各自でトークンを発行して使用してください)。人物領域を照合し、複数人／大勢・ピンぼけ・"
                "品質が怪しい画像を、それぞれ「複数人」「大勢」「ピンぼけ」「品質要確認」フォルダに分けます。"
                "既定では4人以上が大勢です。中央の主役が大きく、端に小さく写り込む程度なら個人に分類します。"
                "自動判定のため、結果をご確認ください。",
    "hybrid": "ローカルで候補を絞り、曖昧な画像をAPIで確認します。APIキーが必要です。"
              "ローカルで確定した画像は人数を確認しません。",
    "api": "全員の参考画像をAPIへ渡して人物と人数を判定します。APIキーが必要です。"
           "一括判定の枚数は「一括判定・人物領域・比較」で変更できます。",
}


class SorterApp(tk.Tk):
    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir.resolve()
        self.title(f"{APP_NAME} Ver{APP_VERSION}")
        self.geometry("940x940")
        self.minsize(820, 820)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None

        self._set_window_icon()
        self.work_var = tk.StringVar(value=str(self.project_dir / "作業フォルダ"))
        self.references_var = tk.StringVar(value=str(self.project_dir / "参考画像"))
        self.output_var = tk.StringVar(value=str(self.project_dir / "振り分け後"))
        self.class_var = tk.StringVar()
        self.class_names: list[str] = []
        self.mode_var = tk.StringVar(value="ローカルのみ（人数判定なし）")
        self.simple_mode_var = tk.StringVar(value="ローカルのみ（人数判定なし）")
        self.key_var = tk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        self.model_var = tk.StringVar(value="gpt-5.6-luna")
        self.top_k_var = tk.StringVar(value="4")
        self.local_confidence_var = tk.StringVar(value="0.78")
        self.local_margin_var = tk.StringVar(value="0.06")
        self.concurrency_var = tk.StringVar(value="4")
        self.batch_size_var = tk.StringVar(value="4")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.trust_scene_var = tk.BooleanVar(value=False)
        self.local_first_var = tk.BooleanVar(value=True)
        self.quality_var = tk.BooleanVar(value=True)
        self.quality_api_var = tk.BooleanVar(value=True)
        self.dominant_var = tk.BooleanVar(value=True)
        self.screen_first_var = tk.BooleanVar(value=False)
        self.reference_cache_var = tk.BooleanVar(value=True)
        self.burst_var = tk.BooleanVar(value=True)
        self.crowd_var = tk.StringVar(value="4")
        self.max_files_var = tk.StringVar(value="20")
        self.save_records_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="待機中")
        self._load_preferences()

        self._build_ui()
        self.bind("<Configure>", self._resize_window)
        self._refresh_classes()
        self.after(100, self._poll_events)

    def _set_window_icon(self) -> None:
        from .window_icon import set_window_icon
        set_window_icon(self)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        edition = " GPU版" if Path(sys.executable).stem.lower().endswith("_gpu") else ""
        header = ttk.Frame(outer)
        header.pack(fill="x")
        self.logo_label = ttk.Label(header)
        self.logo_label.pack(side="left", padx=(0, 12))
        self._set_header_logo()
        ttk.Label(header, text=f"{APP_NAME} Ver{APP_VERSION}{edition}", font=("Yu Gothic UI", 18, "bold")).pack(side="left")
        ttk.Label(
            outer,
            text="イベントで撮った写真を、写っている人物ごとのフォルダに自動で整理するツールです。\n"
                 "人物ごとの参考写真を用意し、整理したい写真を「作業フォルダ」に入れて実行します。\n"
                 "誰が写っているかを参考写真と見比べ、振り分けできた写真を「振り分け後」へ移動します。",
            justify="left",
        ).pack(anchor="w", pady=(6, 10))

        paths = ttk.LabelFrame(outer, text="フォルダ", padding=10)
        paths.pack(fill="x")
        self._path_row(paths, 0, "作業フォルダ", self.work_var)
        self._path_row(paths, 1, "参考画像", self.references_var)
        self._path_row(paths, 2, "振り分け後", self.output_var)

        self.class_frame = ttk.LabelFrame(outer, text="クラス選択", padding=10)
        self.class_frame.pack(fill="x", pady=(12, 0))
        self.class_buttons = ttk.Frame(self.class_frame)
        self.class_buttons.pack(side="left", fill="x", expand=True)
        ttk.Button(
            self.class_frame,
            text="再読込",
            command=lambda: self._refresh_classes(show_errors=True),
        ).pack(side="right", padx=(8, 0))
        ttk.Button(self.class_frame, text="確認済み写真を参考に追加", command=self._add_reference).pack(side="right")

        self.settings_tabs = ttk.Notebook(outer)
        self.settings_tabs.pack(fill="x", pady=12)
        self.simple_settings = ttk.Frame(self.settings_tabs, padding=10)
        self.detailed_settings = ttk.Frame(self.settings_tabs, padding=4)
        self.settings_tabs.add(self.simple_settings, text="シンプル設定")
        self.settings_tabs.add(self.detailed_settings, text="詳細設定")
        self._build_simple_settings()
        self.details_canvas = tk.Canvas(self.detailed_settings, height=340, highlightthickness=0)
        details_scrollbar = ttk.Scrollbar(self.detailed_settings, orient="vertical", command=self.details_canvas.yview)
        details_scrollbar.pack(side="right", fill="y")
        self.details_canvas.configure(yscrollcommand=details_scrollbar.set)
        self.details_canvas.pack(side="left", fill="both", expand=True)
        detail_content = ttk.Frame(self.details_canvas)
        detail_window = self.details_canvas.create_window((0, 0), window=detail_content, anchor="nw")
        detail_content.bind("<Configure>", lambda _event: self.details_canvas.configure(scrollregion=self.details_canvas.bbox("all")))
        self.details_canvas.bind("<Configure>", lambda event: self.details_canvas.itemconfigure(detail_window, width=event.width))
        self.bind("<MouseWheel>", self._scroll_details)
        tabs = ttk.Notebook(detail_content)
        tabs.pack(fill="x")
        settings = ttk.Frame(tabs, padding=10)
        advanced = ttk.Frame(tabs, padding=16)
        tabs.add(settings, text="判定設定")
        tabs.add(advanced, text="一括判定・人物領域・比較")
        economy = ttk.Frame(tabs, padding=16)
        tabs.add(economy, text="ローカル優先・ピンぼけ")
        ttk.Checkbutton(economy, text="人物領域でローカル判定を優先する", variable=self.local_first_var).pack(anchor="w", pady=6)
        ttk.Label(economy, text="切り抜きを変えた人物照合と、検出サイズを変えた人数確認が一致する写真はAPIを省略します。"
                  "鏡・スクリーン・未知の外見では誤る場合があります。OFFで従来の人数API確認に戻せます。", wraplength=740).pack(anchor="w")
        ttk.Checkbutton(economy, text="人物領域でピンぼけを分類する（二段階）", variable=self.quality_var).pack(anchor="w", pady=(16, 6))
        ttk.Checkbutton(economy, text="品質が曖昧な写真だけAPIで追加確認（人数も同時に確認）", variable=self.quality_api_var).pack(anchor="w", pady=6)
        ttk.Label(economy, text="鮮明な写真はローカルで通過。明確なピンぼけは「ピンぼけ」、判断できない写真は「品質要確認」へ。"
                  "追加確認をOFFにすると品質APIは使わず、候補を「品質要確認」へ入れます。人物領域モード専用です。", wraplength=740).pack(anchor="w")
        ttk.Button(economy, text="人物領域：従来／ローカル優先／ピンぼけ付きで比較（API課金あり）",
                   command=lambda: self._compare(['regional', 'regional_local', 'regional_quality'])).pack(anchor="w", pady=(16, 6))
        ttk.Button(economy, text="現在の人物領域設定で並列3／4／6を比較（API課金あり）",
                   command=lambda: self._compare(['regional_c3', 'regional_c4', 'regional_c6'])).pack(anchor="w", pady=6)
        ttk.Label(economy, text="比較は写真を移動しません。参考領域の再利用も一時OFFにし、時間・料金・API不要枚数・再試行数を記録します。"
                  "品質判定の前後で要確認枚数も比較してください。", wraplength=740).pack(anchor="w", pady=6)
        ttk.Label(advanced, text="固定参考API：1回に判定する写真の枚数").pack(anchor="w")
        ttk.Combobox(advanced, textvariable=self.batch_size_var, values=("1", "2", "4"),
                     state="readonly", width=8).pack(anchor="w", pady=6)
        ttk.Label(advanced, text="2・4枚では曖昧な写真だけ個別に再判定します。固定参考APIモードで有効です。",
                  wraplength=720).pack(anchor="w", pady=(0, 12))
        ttk.Checkbutton(advanced, text="判定だけ試す（写真を移動しません・API料金は発生します）",
                        variable=self.dry_run_var).pack(anchor="w", pady=6)
        ttk.Label(advanced, text="人物領域ハイブリッドでは、検出した人物を切り抜いてローカル照合します。"
                  "省略条件とピンぼけ分類は「ローカル優先・ピンぼけ」で設定します。モデル未配置の場合は初回に約690MBを取得します。",
                  wraplength=720).pack(anchor="w", pady=(12, 6))
        ttk.Checkbutton(advanced, text="人数もローカルで確定する（試験・鏡や写り込みの誤判定に注意）",
                        variable=self.trust_scene_var).pack(anchor="w", pady=6)
        ttk.Button(advanced, text="写真の人物領域を確認・指定", command=self._edit_regions).pack(anchor="w", pady=6)
        ttk.Label(advanced, text="人物領域は写真の横に保存します。参考画像にも指定できます。"
                  "写真内の全員を囲み、鏡・ポスターは囲まないでください。", wraplength=720).pack(anchor="w")
        ttk.Button(advanced, text="4方式を同じ写真で比較（写真移動なし・API課金あり）",
                   command=self._compare).pack(anchor="w", pady=(16, 6))
        ttk.Label(advanced, text="ハイブリッド／固定参考1枚・2枚・4枚を順に実行します。"
                  "最大枚数の設定を使用します。比較結果は出力先の「比較結果」に保存します。",
                  wraplength=720).pack(anchor="w")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        ttk.Label(settings, text="モード").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            settings,
            textvariable=self.mode_var,
            values=list(MODE_LABELS),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="APIモデル").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.model_var).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(settings, text="APIキー").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.key_var, show="●").grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(
            settings,
            text="入力して実行すると、EXEと同じ場所の .env に保存し、次回から利用できます。",
            foreground="#555555",
        ).grid(row=2, column=1, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(settings, text="ローカル候補数").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(settings, textvariable=self.top_k_var, from_=1, to=21, width=8).grid(
            row=3, column=1, sticky="w", pady=4
        )
        ttk.Label(settings, text="大勢とする人数").grid(row=3, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Spinbox(settings, textvariable=self.crowd_var, from_=3, to=30, width=8).grid(
            row=3, column=3, sticky="w", pady=4
        )

        ttk.Label(settings, text="ローカル類似度").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Spinbox(
            settings,
            textvariable=self.local_confidence_var,
            from_=0.0,
            to=1.0,
            increment=0.01,
            width=8,
        ).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Label(settings, text="1位と2位の最低差").grid(row=4, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Spinbox(
            settings,
            textvariable=self.local_margin_var,
            from_=0.0,
            to=1.0,
            increment=0.01,
            width=8,
        ).grid(row=4, column=3, sticky="w", pady=4)
        ttk.Label(
            settings,
            text="類似度は正解率ではありません。両方を満たすとAPIを省略します。",
            foreground="#555555",
        ).grid(row=5, column=1, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(settings, text="今回の最大枚数").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.max_files_var, width=10).grid(row=6, column=1, sticky="w", pady=4)
        ttk.Label(settings, text="API同時処理数").grid(row=6, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Spinbox(settings, textvariable=self.concurrency_var, from_=1, to=8, width=8).grid(
            row=6, column=3, sticky="w", pady=4
        )
        ttk.Checkbutton(
            settings,
            text="中央の主役が十分大きく、端が小さな写り込みならソロ扱い",
            variable=self.dominant_var,
        ).grid(row=7, column=1, columnspan=3, sticky="w", pady=(2, 0))
        ttk.Checkbutton(settings, text="集合写真向け：先に人数だけ判定（ソロではAPIが1回増えます）",
                        variable=self.screen_first_var).grid(row=8, column=1, columnspan=3, sticky="w")
        ttk.Checkbutton(settings, text="参考画像の人物領域だけを再利用（作業写真は毎回判定）",
                        variable=self.reference_cache_var).grid(row=9, column=1, columnspan=3, sticky="w")
        ttk.Checkbutton(economy, text="連写の人物照合を共有（3秒以内・人数と品質は写真ごとに確認）",
                        variable=self.burst_var).pack(anchor="w", pady=6)
        ttk.Button(economy, text="処理記録から参考画像の補強候補を確認", command=self._suggest_references).pack(anchor="w", pady=6)
        ttk.Label(settings, text="最大枚数は0 = 全件。最初は20枚推奨", foreground="#555555").grid(
            row=10, column=1, columnspan=3, sticky="w", pady=(0, 4)
        )
        ttk.Checkbutton(settings, text="処理記録を保存（CSV・集計JSON・再開用履歴）", variable=self.save_records_var).grid(
            row=11, column=1, columnspan=3, sticky="w", pady=4)
        ttk.Label(settings, text="標準は保存しません。比較を実行した場合は、比較結果と記録を保存します。",
                  foreground="#555555").grid(row=12, column=1, columnspan=3, sticky="w")
        self.advanced_mode_description = tk.StringVar()
        ttk.Label(settings, textvariable=self.advanced_mode_description, wraplength=740).grid(
            row=13, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.mode_var.trace_add("write", self._update_mode_descriptions)
        self.simple_mode_var.trace_add("write", self._update_mode_descriptions)
        self._update_mode_descriptions()
        self._settings_page = str(self.simple_settings)
        self.settings_tabs.bind("<<NotebookTabChanged>>", self._settings_tab_changed)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="振り分け開始", command=self._start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="停止", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.progress_bar = ttk.Progressbar(buttons, mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.initial_run_notice = ttk.Label(
            outer, text="初回実行時はモデルデータのダウンロード等で時間がかかります。", wraplength=850)
        self.initial_run_notice.pack(anchor="w", fill="x", pady=(10, 0))
        self.initial_run_notice.bind("<Configure>", lambda event: self.initial_run_notice.configure(wraplength=max(200, event.width)))
        self.api_transfer_notice = ttk.Label(
            outer, text="APIを使うモード・方式比較では、写真・切り抜き・参考画像と、区分名／人物名フォルダ由来のIDをOpenAIへ送信します。"
                        "必要な権利・許諾を確認してご利用ください。ローカルのみでは画像をAPIへ送りません。",
            wraplength=850, foreground="#555555")
        self.api_transfer_notice.pack(anchor="w", fill="x", pady=(4, 0))
        self.api_transfer_notice.bind("<Configure>", lambda event: self.api_transfer_notice.configure(wraplength=max(200, event.width)))
        status = ttk.Label(outer, textvariable=self.status_var, wraplength=850)
        status.pack(anchor="w", fill="x", pady=(4, 4))
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(200, event.width)))
        log_frame = ttk.LabelFrame(outer, text="処理ログ", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _set_header_logo(self) -> None:
        from PIL import Image, ImageOps, ImageTk
        try:
            with Image.open(bundled_resource("assets/app_icon.png")) as source:
                logo = ImageOps.contain(source.convert("RGBA"), (100, 100), Image.Resampling.LANCZOS)
            self._header_logo = ImageTk.PhotoImage(logo)
            self.logo_label.configure(image=self._header_logo)
        except (OSError, tk.TclError):
            pass

    def _build_simple_settings(self) -> None:
        panel = self.simple_settings
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="モード").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=4)
        ttk.Combobox(panel, textvariable=self.simple_mode_var, values=list(SIMPLE_MODE_LABELS),
                     state="readonly", width=32).grid(row=0, column=1, sticky="w", pady=4)
        self.simple_mode_description = tk.StringVar()
        ttk.Label(panel, textvariable=self.simple_mode_description, wraplength=740).grid(
            row=1, column=1, sticky="ew", pady=(2, 10))
        ttk.Label(panel, text="枚数設定").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.max_files_var, width=10).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="今回の最大枚数。0 = 全件。最初は20枚でお試しください。", foreground="#555555").grid(
            row=3, column=1, sticky="w", pady=(0, 8))
        ttk.Label(panel, text="APIキー").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.key_var, show="●").grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(panel, text="入力して実行すると .env に保存し、次回から利用できます。ローカルのみでは空欄で使えます。",
                  foreground="#555555", wraplength=740).grid(row=5, column=1, sticky="ew", pady=(0, 8))
        ttk.Label(panel, text="その他は既定値を使用します。細かい調整や記録の保存は「詳細設定」で実行してください。",
                  foreground="#555555", wraplength=800).grid(row=6, column=0, columnspan=2, sticky="ew")

    def _update_mode_descriptions(self, *_args) -> None:
        self.simple_mode_description.set(MODE_DESCRIPTIONS[SIMPLE_MODE_LABELS[self.simple_mode_var.get()]])
        self.advanced_mode_description.set(MODE_DESCRIPTIONS[MODE_LABELS[self.mode_var.get()]]
            + "（詳細設定で品質分類などをOFFにすると、その機能は使われません。）")
        self.after_idle(self._resize_settings)

    def _scroll_details(self, event) -> None:
        widget = event.widget
        while widget is not None:
            if widget is self.detailed_settings:
                self.details_canvas.yview_scroll(-int(event.delta / 120), "units")
                return
            widget = getattr(widget, "master", None)

    def _resize_settings(self) -> None:
        self.update_idletasks()
        selected = self.settings_tabs.select()
        self.settings_tabs.configure(height=self.nametowidget(selected).winfo_reqheight())

    def _resize_window(self, event) -> None:
        if event.widget is self:
            height = max(200, min(340, event.height - 625))
            if int(self.details_canvas.cget("height")) != height:
                self.details_canvas.configure(height=height)
                self.after_idle(self._resize_settings)

    def _settings_tab_changed(self, _event=None) -> None:
        selected = self.settings_tabs.select()
        if selected != self._settings_page:
            if selected == str(self.detailed_settings):
                self.mode_var.set(self.simple_mode_var.get())
            else:
                self.simple_mode_var.set(self.mode_var.get() if self.mode_var.get() in SIMPLE_MODE_LABELS
                                         else "ローカルのみ（人数判定なし）")
            self._settings_page = selected
        self.after_idle(self._resize_settings)

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if variable is self.references_var:
            entry.bind("<FocusOut>", lambda _event: self._refresh_classes())
        ttk.Button(parent, text="選択", command=lambda: self._choose_dir(variable)).grid(
            row=row, column=2, padx=(8, 0), pady=4
        )

    def _choose_dir(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(self.project_dir))
        if selected:
            variable.set(selected)
            if variable is self.references_var:
                self._refresh_classes(show_errors=True)

    def _refresh_classes(self, show_errors: bool = False) -> None:
        reference_root = Path(self.references_var.get()).expanduser()
        class_dirs = discover_reference_classes(reference_root)
        self.class_names = [path.name for path in class_dirs]

        for child in self.class_buttons.winfo_children():
            child.destroy()

        current = self.class_var.get()
        if current not in self.class_names:
            self.class_var.set(self.class_names[0] if self.class_names else "")

        self.class_frame.configure(text=f"クラス選択（{len(self.class_names)}件）")
        if not self.class_names:
            ttk.Label(
                self.class_buttons,
                text="参考画像フォルダ直下にクラスフォルダがありません。",
                foreground="#a00000",
            ).grid(row=0, column=0, sticky="w")
            if show_errors:
                messagebox.showwarning(
                    "クラスが見つかりません",
                    "参考画像フォルダ直下に、クラスごとのフォルダを作成してください。",
                )
            return

        for index, class_name in enumerate(self.class_names):
            ttk.Radiobutton(
                self.class_buttons,
                text=class_name,
                value=class_name,
                variable=self.class_var,
            ).grid(row=index // 5, column=index % 5, sticky="w", padx=(0, 18), pady=2)

    def _build_config(self) -> SorterConfig:
        simple = self.settings_tabs.select() == str(self.simple_settings)
        mode = SIMPLE_MODE_LABELS[self.simple_mode_var.get()] if simple else MODE_LABELS[self.mode_var.get()]
        reference_root = Path(self.references_var.get()).expanduser()
        available_classes = {
            path.name: path for path in discover_reference_classes(reference_root)
        }
        selected_class = self.class_var.get()
        if selected_class not in available_classes:
            raise ValueError("参考画像からクラスを再読込し、対象クラスを選択してください。")
        common = dict(
            work_dir=Path(self.work_var.get()), reference_dir=available_classes[selected_class],
            output_dir=Path(self.output_var.get()) / selected_class, mode=mode,
            api_key=self.key_var.get().strip() or os.getenv("OPENAI_API_KEY"),
            max_files=int(self.max_files_var.get()), cache_dir=model_cache_dir(self.project_dir),
        )
        if simple:
            return SorterConfig(**common, api_concurrency=4,
                                regional_local_first=mode == "regional", quality_filter=mode == "regional")
        return SorterConfig(
            **common,
            model=self.model_var.get().strip() or "gpt-5.6-luna",
            local_top_k=int(self.top_k_var.get()),
            local_confidence=float(self.local_confidence_var.get()),
            local_margin=float(self.local_margin_var.get()),
            api_concurrency=int(self.concurrency_var.get()),
            api_batch_size=int(self.batch_size_var.get()) if mode == "api" else 1,
            dry_run=self.dry_run_var.get(),
            trust_local_scene=self.trust_scene_var.get() if mode == "regional" else False,
            allow_dominant_subject=self.dominant_var.get(),
            crowd_threshold=int(self.crowd_var.get()),
            save_records=self.save_records_var.get(),
            screen_first=self.screen_first_var.get(),
            resume=False,
            reuse_api_results=False,
            reuse_reference_regions=self.reference_cache_var.get(),
            share_burst_identity=self.burst_var.get(),
            regional_local_first=self.local_first_var.get() if mode == 'regional' else False,
            quality_filter=self.quality_var.get() if mode == 'regional' else False,
            quality_api_review=self.quality_api_var.get(),
        )

    def _preference_vars(self):
        return {'mode': self.mode_var, 'simple_mode': self.simple_mode_var, 'concurrency': self.concurrency_var,
                'reuse_reference_regions': self.reference_cache_var, 'share_burst_identity': self.burst_var,
                'local_first': self.local_first_var,
                'quality_filter': self.quality_var, 'quality_api_review': self.quality_api_var}

    def _load_preferences(self):
        try:
            data = json.loads((self.project_dir / 'アプリデータ/settings.json').read_text(encoding='utf-8'))
            for key, variable in self._preference_vars().items():
                value = data.get(key)
                if isinstance(variable, tk.BooleanVar) and isinstance(value, bool):
                    variable.set(value)
                elif key == 'mode' and value == '人物領域ハイブリッド（試験）':
                    variable.set('人物領域ハイブリッド')
                elif key == 'mode' and value == '従来ハイブリッド':
                    variable.set('ハイブリッド')
                elif key == 'mode' and value in MODE_LABELS:
                    variable.set(value)
                elif key == 'simple_mode' and value in SIMPLE_MODE_LABELS:
                    variable.set(value)
                elif key == 'concurrency' and str(value) in [str(n) for n in range(1, 9)]:
                    variable.set(str(value))
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def _save_preferences(self):
        path = self.project_dir / 'アプリデータ/settings.json'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({key: value.get() for key, value in self._preference_vars().items()},
                                           ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
        except OSError as exc:
            self._append_log(f'設定保存をスキップ: {exc}')

    def _add_reference(self, selected=None) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("処理中", "振り分けが終了してから参考画像を追加してください。")
            return
        try:
            reference_dir = Path(self.references_var.get()) / self.class_var.get()
            identities = discover_reference_catalog(reference_dir)
        except ValueError as exc:
            messagebox.showerror("参考画像", str(exc))
            return
        selected = selected or filedialog.askopenfilename(
            title="人物を確認して参考に追加する写真を選択",
            initialdir=str(Path(self.output_var.get()) / self.class_var.get()),
            filetypes=[("画像", "*.png *.jpg *.jpeg *.webp *.bmp *.gif")],
        )
        if not selected:
            return
        source = Path(selected)
        try:
            from PIL import ImageTk
            preview = ImageTk.PhotoImage(resized_for_model(open_rgb(source), 640))
        except (OSError, ValueError) as exc:
            messagebox.showerror("画像を開けません", str(exc))
            return
        dialog = tk.Toplevel(self)
        dialog.title("参考画像に追加する人物の確認")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(dialog, text="写っている本人を選んでください。単独で、顔や服装が分かる写真を使います。",
                  padding=10).pack()
        label = ttk.Label(dialog, image=preview)
        label.image = preview
        label.pack(padx=10)
        ttk.Label(dialog, text=source.name, padding=6).pack()
        identity_var = tk.StringVar()
        names = [i.identity_id for i in identities]
        suggested = next((i.identity_id for i in identities
                          if source.parent.name == i.name and source.parent.parent.name == i.category), "")
        identity_var.set(suggested)
        ttk.Combobox(dialog, textvariable=identity_var, values=names, state="readonly", width=55).pack(padx=10, pady=8)
        local_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="ローカル専用に追加（APIへ送る参考画像を増やしません）", variable=local_only_var).pack(pady=6)

        def save() -> None:
            try:
                destination, added = add_confirmed_reference(reference_dir, identity_var.get(), source, local_only=local_only_var.get())
            except (OSError, ValueError) as exc:
                messagebox.showerror("参考画像の追加", str(exc), parent=dialog)
                return
            self._append_log(f"参考画像{'を追加' if added else 'は登録済み'}: {destination}")
            self.status_var.set("参考画像を追加しました。次回のローカル判定から利用します。" if added else "この写真は登録済みです。")
            dialog.destroy()

        ttk.Button(dialog, text="この人物の参考画像に追加（写真はコピー）", command=save).pack(pady=(4, 12))

    def _suggest_references(self):
        if self.worker and self.worker.is_alive():
            return
        manifest = filedialog.askopenfilename(title='参考画像の補強に使う処理記録を選択',
            initialdir=str(Path(self.output_var.get()) / self.class_var.get() / '処理記録'),
            filetypes=[('処理記録', '*.csv')])
        if not manifest:
            return
        try:
            from .reference_suggestions import suggest_references
            suggestions = suggest_references(Path(manifest), Path(self.references_var.get()) / self.class_var.get())
        except (OSError, ValueError) as exc:
            messagebox.showerror('補強候補', str(exc))
            return
        if not suggestions:
            messagebox.showinfo('補強候補', '未登録の補強候補はありません。品質が良好な単独写真で、人物照合APIを使った写真が対象です。')
            return
        dialog = tk.Toplevel(self)
        dialog.title('ローカル参考画像の補強候補')
        ttk.Label(dialog, text='1人につき最大2枚。自動判定の人物名は候補です。写真を開き、本人を確認してから追加してください。',
                  wraplength=740, padding=10).pack()
        listing = tk.Listbox(dialog, width=100, height=min(18, len(suggestions)), exportselection=False)
        listing.pack(fill='both', expand=True, padx=10)
        for item in suggestions:
            listing.insert('end', f"{item['identity_id']}  |  {item['reason']}  |  {Path(item['source']).name}")
        def review():
            selection = listing.curselection()
            if selection:
                self._add_reference(suggestions[selection[0]]['source'])
        ttk.Button(dialog, text='写真と人物を確認して追加', command=review).pack(pady=10)

    def _edit_regions(self):
        if self.worker and self.worker.is_alive():
            return
        from .region_editor import edit_regions
        selected = filedialog.askopenfilename(title="人物領域を指定する写真", initialdir=self.references_var.get(),
            filetypes=[("画像", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if selected:
            edit_regions(self, Path(selected))

    def _compare(self, strategies=None):
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._build_config()
            if not config.api_key:
                raise ValueError("比較にはAPIキーが必要です")
            validate_config(config)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("比較設定", str(exc))
            return
        if not self._persist_api_key():
            return
        from datetime import datetime
        from .benchmark import compare
        strategies = strategies or ['hybrid', 'api1', 'api2', 'api4']
        output = config.output_dir / "比較結果" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.cancel_event.clear()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("同じ写真で設定を比較しています。写真は移動しません。")
        def run():
            try:
                result = compare(config, output, strategies,
                    progress=lambda text: self.events.put(("log", text)), cancel_event=self.cancel_event)
                self.events.put(("comparison_done", (result, output)))
            except Exception as exc:
                self.events.put(("error", exc))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._build_config()
            validate_config(config)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("設定エラー", f"数値または設定を確認してください。\n{exc}")
            return
        if config.mode in {"hybrid", "api", "regional"} and not config.api_key:
            messagebox.showerror("APIキーが必要です", "APIキーを入力するかOPENAI_API_KEYを設定してください。")
            return

        if not self._persist_api_key():
            return

        self.cancel_event.clear()
        self._save_preferences()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_bar["value"] = 0
        self.status_var.set("準備中… 初回のローカルモデル読込には時間がかかります。")
        self._append_log(
            f"クラス「{self.class_var.get()}」を対象に処理を開始します。"
            + ("判定だけ実行し、写真は移動しません。" if config.dry_run else "成功した画像は振り分け先へ移動します。")
        )

        def run() -> None:
            try:
                sorter = PhotoSorter(config, progress=self._on_progress, cancel_event=self.cancel_event)
                records = sorter.run()
                self.events.put(("done", (records, config.model, sorter.elapsed_seconds, config.dry_run)))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _persist_api_key(self) -> bool:
        try:
            save_api_key(self.project_dir, self.key_var.get())
        except (OSError, ValueError):
            messagebox.showerror("APIキーを保存できません", ".env に保存できませんでした。APIキーの改行や、アプリフォルダの書き込み権限を確認してください。")
            return False
        return True

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("実行中の判定が終了してから停止します（一括判定は最大4枚）…")

    def _on_progress(self, event: ProgressEvent) -> None:
        self.events.put(("progress", event))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    event = payload
                    assert isinstance(event, ProgressEvent)
                    self.progress_bar["maximum"] = max(1, event.total)
                    self.progress_bar["value"] = event.current
                    self.status_var.set(
                        f"{event.current}/{event.total}  {event.filename}  "
                        f"→ {event.message}  API累計{event.api_calls}回"
                    )
                    self._append_log(self.status_var.get())
                elif kind == "done":
                    records, model, elapsed_seconds, dry_run = payload  # type: ignore[misc]
                    errors = sum(bool(record.error) for record in records)  # type: ignore[union-attr]
                    api_calls = sum(record.api_calls for record in records)  # type: ignore[union-attr]
                    screen_calls = sum(record.screen_calls for record in records)  # type: ignore[union-attr]
                    input_tokens = sum(record.input_tokens for record in records)  # type: ignore[union-attr]
                    cached_tokens = sum(record.cached_input_tokens for record in records)  # type: ignore[union-attr]
                    cache_write_tokens = sum(record.cache_write_input_tokens for record in records)  # type: ignore[union-attr]
                    output_tokens = sum(record.output_tokens for record in records)  # type: ignore[union-attr]
                    local_count = sum(record.decision_source in {"local", "exact", "regional_local"} and record.local_gate != "no_candidates"
                                      and record.route not in {"要確認", "エラー"} for record in records)
                    cache_hits = sum(record.api_cache_hits for record in records)
                    cost_text = format_cost_estimate(
                        estimate_api_cost(
                            model,
                            input_tokens,
                            cached_tokens,
                            cache_write_tokens,
                            output_tokens,
                        ),
                        model,
                    )
                    self._finish_ui()
                    message = (
                        f"{'判定のみ完了（写真移動なし）' if dry_run else '振り分け完了'}。今回の処理 {len(records)}枚、エラー {errors}枚、"
                        f"API {api_calls}回（事前人数 {screen_calls}、人物照合 {api_calls - screen_calls}）、"
                        f"ローカル確定 {local_count}枚、連写照合共有 {sum(bool(r.burst_shared_from) for r in records)}枚、"
                        f"参考領域の再利用 {sum(r.reference_cache_hits for r in records)}回、"
                        f"API不要 {sum(r.api_calls - r.reference_setup_calls == 0 and not r.error for r in records)}枚、"
                        f"参考領域の初回確認 {sum(r.reference_setup_calls for r in records)}回（人数確認の内数）、"
                        f"ピンぼけ {sum(r.route == 'ピンぼけ' for r in records)}枚、品質要確認 {sum(r.route == '品質要確認' for r in records)}枚、"
                        f"入力 {input_tokens:,} tokens（キャッシュ {cached_tokens:,}、"
                        f"書込 {cache_write_tokens:,}）、出力 {output_tokens:,} tokens、"
                        f"{cost_text}、所要時間 {format_elapsed_time(elapsed_seconds)}。"
                    )
                    self.status_var.set(message)
                    self._append_log(message)
                    messagebox.showinfo("振り分け完了", message)
                elif kind == "log":
                    self.status_var.set(str(payload))
                    self._append_log(str(payload))
                elif kind == "comparison_done":
                    result, output = payload
                    self._finish_ui()
                    lines = []
                    for name, summary in result["strategies"].items():
                        cost = summary["estimated_cost"]
                        price = f"約{cost['jpy']:.2f}円" if cost else "単価未登録"
                        lines.append(f"{name}: {format_elapsed_time(summary['elapsed_seconds'])}・{price}・エラー{summary['errors']}枚")
                    text = "\n".join(lines) + f"\n保存先: {output}\n正解ラベルなしの比較です。分類結果も確認してください。"
                    self._append_log(text)
                    self.status_var.set("比較結果を保存しました。写真は移動していません。")
                    messagebox.showinfo("方式の比較", text)
                elif kind == "error":
                    self._finish_ui()
                    self.status_var.set("処理を開始できませんでした。")
                    self._append_log(f"エラー: {payload}")
                    messagebox.showerror("処理エラー", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _finish_ui(self) -> None:
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    project_dir = application_dir()
    load_app_environment(project_dir)
    app = SorterApp(project_dir)
    app.mainloop()
