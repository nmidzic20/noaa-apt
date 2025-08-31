# ui_skin.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Optional image preview + pill backgrounds for buttons
try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Import functional core
import gui  # the core-only module from above

# Emoji "icons" per decoder (no external files needed)
DECODER_EMOJI = {
    "abs_val": "📈",
    "cosine": "🧮",
    "hilbertFIR": "🧱",
    "hilbertFFT": "⚡",
    "contrast": "🌗",
    "falsecolour": "🌈",
    "pseudocolour1": "🎨",
    "pseudocolour2": "🧬",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NOAA APT Demodulator and Decoder")
        self.geometry("1180x780")

        # Core logic
        self.core = gui.DecoderCore()

        # Theming
        self.style = ttk.Style(self)
        self._current_theme = "light"
        self._apply_theme(self._current_theme)

        # Default output folder shown in field
        default_out = gui.get_default_output_dir()

        # State
        self.selected_decoder = None
        self.decoder_buttons = {}
        self._pill_images_normal = {}
        self._pill_images_hover_normal = {}
        self._pill_images_selected = {}
        self._pill_images_hover_selected = {}

        # ---------- Layout: left controls / right preview ----------
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_col = ttk.Frame(paned)
        right_col = ttk.Frame(paned)
        paned.add(left_col, weight=1)     # controls
        paned.add(right_col, weight=2)    # preview

        # ---------- Left side ----------
        # Top bar with theme toggle
        topbar = ttk.Frame(left_col)
        topbar.pack(fill="x", padx=12, pady=(12, 6))

        title_lbl = ttk.Label(topbar, text="NOAA APT Demodulator and Decoder", font=("Segoe UI", 14, "bold"))
        title_lbl.pack(side="left")

        self.theme_btn = ttk.Button(topbar, text="Dark mode", command=self.toggle_theme)
        self.theme_btn.pack(side="right")

        # Group: Input / Output
        file_frame = ttk.LabelFrame(left_col, text="Input / Output")
        file_frame.pack(fill="x", padx=12, pady=(6, 6))

        # WAV picker
        self.wav_var = tk.StringVar()
        wav_row = ttk.Frame(file_frame)
        wav_row.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(wav_row, text="Input WAV:").pack(side="left", padx=(0, 8))
        self.wav_entry = ttk.Entry(wav_row, textvariable=self.wav_var)
        self.wav_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(wav_row, text="Browse…", command=self.pick_wav).pack(side="left", padx=8)

        # Output folder
        self.outdir_var = tk.StringVar(value=default_out)
        out_row = ttk.Frame(file_frame)
        out_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(out_row, text="Output folder:").pack(side="left", padx=(0, 8))
        self.out_entry = ttk.Entry(out_row, textvariable=self.outdir_var)
        self.out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse…", command=self.pick_outdir).pack(side="left", padx=8)

        # Group: Options
        opts_frame = ttk.LabelFrame(left_col, text="Options")
        opts_frame.pack(fill="x", padx=12, pady=(0, 6))

        # Width (numeric)
        self.width_var = tk.StringVar(value="1200")
        width_row = ttk.Frame(opts_frame)
        width_row.pack(fill="x", padx=10, pady=(10, 10))
        ttk.Label(width_row, text="Width:").pack(side="left", padx=(0, 8))
        vcmd = (self.register(self._validate_int), "%P")
        self.width_entry = ttk.Entry(width_row, textvariable=self.width_var, width=12, validate="key", validatecommand=vcmd)
        self.width_entry.pack(side="left")

        # Mode / Color (Comboboxes)
        adv_row = ttk.Frame(opts_frame)
        adv_row.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(adv_row, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.mode_var = tk.StringVar()
        self.mode_combo = ttk.Combobox(adv_row, textvariable=self.mode_var, state="disabled", width=14)
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(adv_row, text="Color:").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.color_var = tk.StringVar()
        self.color_combo = ttk.Combobox(adv_row, textvariable=self.color_var, state="disabled", width=18)
        self.color_combo.grid(row=0, column=3, sticky="w")

        # Group: Decoder selection
        dec_frame = ttk.LabelFrame(left_col, text="Choose decoder")
        dec_frame.pack(fill="x", padx=12, pady=(0, 6))
        btn_grid = ttk.Frame(dec_frame)
        btn_grid.pack(fill="x", padx=10, pady=10)

        # Create decoder buttons (with pill backgrounds if Pillow available)
        for i, (label, exe) in enumerate(gui.DECODERS):
            text = f"{DECODER_EMOJI.get(label, '🛰️')}  {label}"
            b = ttk.Button(
                btn_grid,
                text=text,
                style="Decoder.TButton",
                command=lambda lbl=label: self.select_decoder(lbl)
            )
            r, c = divmod(i, 4)
            b.grid(row=r, column=c, sticky="ew", padx=6, pady=6)
            btn_grid.grid_columnconfigure(c, weight=1)
            self.decoder_buttons[label] = b
            self._add_hover(b)

        # Actions
        actions = ttk.Frame(left_col)
        actions.pack(fill="x", padx=12, pady=10)
        self.run_btn = ttk.Button(actions, text="Run", style="Action.TButton", command=self.run_clicked)
        self.run_btn.pack(side="left")

        # Spinner (animated text) shown during processing
        self._spinner_var = tk.StringVar(value="")
        self.spinner_label = ttk.Label(actions, textvariable=self._spinner_var)
        # pack but hide initially
        self.spinner_label.pack(side="left", padx=12)
        self.spinner_label.pack_forget()

        ttk.Button(actions, text="Quit", command=self.destroy).pack(side="left", padx=8)

        # Spinner state
        self._spinning = False
        self._spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_idx = 0

        # Log
        log_frame = ttk.LabelFrame(left_col, text="Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log = tk.Text(log_frame, height=12, wrap="word", relief="flat", bd=0)
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

        # ---------- Right side (Preview) ----------
        preview_frame = ttk.LabelFrame(right_col, text="Preview")
        preview_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Zoom toolbar
        zoombar = ttk.Frame(preview_frame)
        zoombar.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 4))
        ttk.Button(zoombar, text="＋", width=3, command=lambda: self._zoom_step(1.25)).pack(side="left")
        ttk.Button(zoombar, text="－", width=3, command=lambda: self._zoom_step(1/1.25)).pack(side="left", padx=(6, 0))
        ttk.Button(zoombar, text="100%", command=self._zoom_100).pack(side="left", padx=(10, 0))
        ttk.Button(zoombar, text="Fit", command=self._zoom_fit).pack(side="left", padx=(6, 0))

        # Scrollable canvas that displays the image (zoomable)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            highlightthickness=0,
            bd=0,
            background=self._palette.get("bg", "#FFFFFF")
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        yscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        xscroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll.grid(row=2, column=0, sticky="ew")

        # Mouse wheel zoom with Ctrl, pan with drag
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_mousewheel_zoom)
        self.preview_canvas.bind("<ButtonPress-1>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B1-Motion>",     lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))

        # Internal image/zoom state
        self._tk_img_preview = None
        self._img_id = None
        self._base_pil = None   # original PIL image
        self._zoom = 1.0        # current zoom factor
        self._min_zoom = 0.1
        self._max_zoom = 8.0

        # Generate pill images if possible
        self._build_pill_images()
        self._refresh_decoder_button_looks()

    # ---------- Theming & styles ----------

    def toggle_theme(self):
        self._current_theme = "dark" if self._current_theme == "light" else "light"
        self._apply_theme(self._current_theme)
        self._build_pill_images()
        self._refresh_decoder_button_looks()
        self.theme_btn.configure(text="Light mode" if self._current_theme == "dark" else "Dark mode")
        # Update preview canvas background to match theme
        self.preview_canvas.configure(background=self._palette.get("bg", "#FFFFFF"))

    def _apply_theme(self, mode: str):
        base = "clam"  # respects custom colors better than "vista"
        if base in self.style.theme_names():
            self.style.theme_use(base)

        # Adjusted palette (dark-mode hover is darker, not light)
        if mode == "light":
            bg = "#F7F7F8"
            fg = "#111827"
            entry_bg = "#FFFFFF"
            btn_bg = "#E5E7EB"         # neutral
            btn_hover = "#D1D5DB"      # slight contrast
            primary = "#2563EB"        # selected
            primary_hover_selected = "#1D4ED8"  # darker on hover when selected
            selected_fg = "#FFFFFF"
            outline = "#D1D5DB"
        else:
            bg = "#0F172A"
            fg = "#E5E7EB"
            entry_bg = "#1E293B"
            btn_bg = "#334155"         # neutral dark
            btn_hover = "#475569"      # slightly lighter but still dark
            primary = "#3B82F6"        # selected
            primary_hover_selected = "#1E3A8A"  # darker hover to keep text readable
            selected_fg = "#FFFFFF"
            outline = "#475569"

        # Window background
        self.configure(bg=bg)

        # Style ttk widgets
        self.style.configure(".", background=bg, foreground=fg, font=("Segoe UI", 10))
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TLabelframe", background=bg, foreground=fg)
        self.style.configure("TLabelframe.Label", background=bg, foreground=fg, font=("Segoe UI", 10, "bold"))
        self.style.configure("TFrame", background=bg)
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=fg)
        self.style.configure("TCombobox", fieldbackground=entry_bg, foreground=fg, background=bg)
        self.style.map("TCombobox", fieldbackground=[("readonly", entry_bg)])

        # Action buttons
        self.style.configure("Action.TButton", padding=(12, 8), relief="flat", background=primary, foreground=selected_fg)
        self.style.map("Action.TButton",
                       background=[("active", primary_hover_selected)],
                       foreground=[("active", selected_fg)])

        # Decoder buttons (text-only fallback if no Pillow)
        self.style.configure("Decoder.TButton", padding=(12, 10), relief="flat", background=btn_bg, foreground=fg)
        self.style.map("Decoder.TButton",
                       background=[("active", btn_hover)],
                       foreground=[("active", fg)])

        self.style.configure("DecoderSelected.TButton", padding=(12, 10), relief="flat",
                             background=primary, foreground=selected_fg)
        self.style.map("DecoderSelected.TButton",
                       background=[("active", primary_hover_selected)],
                       foreground=[("active", selected_fg)])

        self._palette = {
            "bg": bg,
            "fg": fg,
            "entry_bg": entry_bg,
            "btn_bg": btn_bg,
            "btn_hover": btn_hover,
            "primary": primary,
            "primary_hover_selected": primary_hover_selected,
            "selected_fg": selected_fg,
            "outline": outline,
        }

    def _build_pill_images(self):
        """Build rounded pill background images for normal/hover(selected & normal)/selected states."""
        self._pill_images_normal.clear()
        self._pill_images_hover_normal.clear()
        self._pill_images_selected.clear()
        self._pill_images_hover_selected.clear()

        if not PIL_AVAILABLE:
            return

        # Sizes for the background image
        w, h, r = 200, 42, 18
        pad = 2

        def rounded_rect(color_fill, outline, shadow_alpha=0):
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            if shadow_alpha > 0:
                s = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                sd = ImageDraw.Draw(s)
                sd.rounded_rectangle([(pad, pad), (w - pad, h - pad)], radius=r, fill=(0, 0, 0, shadow_alpha))
                img = Image.alpha_composite(img, s)
            draw.rounded_rectangle([(pad, pad), (w - pad, h - pad)], radius=r,
                                   fill=color_fill, outline=self._palette["outline"])
            return ImageTk.PhotoImage(img)

        btn_bg = self._palette["btn_bg"]
        btn_hover = self._palette["btn_hover"]
        primary = self._palette["primary"]
        primary_hover_selected = self._palette["primary_hover_selected"]

        normal = rounded_rect(btn_bg, self._palette["outline"], shadow_alpha=30 if self._current_theme == "dark" else 0)
        hover_normal = rounded_rect(btn_hover, self._palette["outline"], shadow_alpha=45 if self._current_theme == "dark" else 0)
        selected = rounded_rect(primary, self._palette["outline"], shadow_alpha=50 if self._current_theme == "dark" else 0)
        hover_selected = rounded_rect(primary_hover_selected, self._palette["outline"], shadow_alpha=60 if self._current_theme == "dark" else 0)

        for label, _ in gui.DECODERS:
            self._pill_images_normal[label] = normal
            self._pill_images_hover_normal[label] = hover_normal
            self._pill_images_selected[label] = selected
            self._pill_images_hover_selected[label] = hover_selected

    def _refresh_decoder_button_looks(self):
        """Apply appropriate style/image to decoder buttons according to theme + selection."""
        for label, btn in self.decoder_buttons.items():
            if PIL_AVAILABLE:
                if self.selected_decoder == label:
                    btn.configure(style="DecoderSelected.TButton", image=self._pill_images_selected[label], compound="center")
                else:
                    btn.configure(style="Decoder.TButton", image=self._pill_images_normal[label], compound="center")
            else:
                btn.configure(style="DecoderSelected.TButton" if self.selected_decoder == label else "Decoder.TButton")

    # ---------- UI helpers ----------

    def _validate_int(self, proposed: str) -> bool:
        if proposed == "":
            return True
        return proposed.isdigit()

    def _set_combo(self, combo: ttk.Combobox, values, enabled: bool, default_idx: int = 0):
        combo.configure(values=values)
        if values:
            combo.current(default_idx)
        combo.configure(state="readonly" if enabled else "disabled")

    def _add_hover(self, btn: ttk.Button):
        # Hover cursor + image swap distinct for normal/selected if Pillow is available
        def on_enter(_e):
            btn.configure(cursor="hand2")
            if PIL_AVAILABLE:
                for label, b in self.decoder_buttons.items():
                    if b is btn:
                        if self.selected_decoder == label:
                            btn.configure(image=self._pill_images_hover_selected[label])
                        else:
                            btn.configure(image=self._pill_images_hover_normal[label])
                        break

        def on_leave(_e):
            btn.configure(cursor="")
            if PIL_AVAILABLE:
                for label, b in self.decoder_buttons.items():
                    if b is btn:
                        if self.selected_decoder == label:
                            btn.configure(image=self._pill_images_selected[label])
                        else:
                            btn.configure(image=self._pill_images_normal[label])
                        break

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def select_decoder(self, label: str):
        self.selected_decoder = label

        # Update styles/images for all buttons
        self._refresh_decoder_button_looks()

        # Enable/disable Mode/Color according to decoder
        if label in ("pseudocolour1", "pseudocolour2"):
            self._set_combo(self.mode_combo, ["manual"], enabled=True, default_idx=0)
            self._set_combo(self.color_combo, ["pseudo"], enabled=True, default_idx=0)
        elif label == "falsecolour":
            self._set_combo(self.mode_combo, gui.FALSECOLOUR_MODES, enabled=True, default_idx=0)
            self._set_combo(self.color_combo, gui.FALSECOLOUR_COLORS, enabled=True, default_idx=0)
        else:
            # Not applicable for the first five decoders
            self._set_combo(self.mode_combo, [], enabled=False)
            self._set_combo(self.color_combo, [], enabled=False)

    def pick_wav(self):
        path = filedialog.askopenfilename(
            title="Select WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.wav_var.set(path)

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.outdir_var.set(path)

    # ---------- Run logic ----------

    def run_clicked(self):
        if not self.selected_decoder:
            messagebox.showerror("Error", "Please choose a decoder.")
            return

        wav = self.wav_var.get().strip()
        if not wav:
            messagebox.showerror("Error", "Please select a WAV file.")
            return

        width_str = self.width_var.get().strip()
        if not width_str or not width_str.isdigit():
            messagebox.showerror("Error", "Please enter a numeric Width.")
            return

        # Build command using core
        try:
            cmd, out_png = self.core.build_command(
                decoder_key=self.selected_decoder,
                wav_path=wav,
                output_dir=self.outdir_var.get().strip(),
                width=width_str,
                mode=(self.mode_var.get() or None),
                color=(self.color_var.get() or None),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # UI prep
        self.log_delete()
        self._clear_preview()

        # Disable Run and show spinner
        self.run_btn.state(["disabled"])
        self._start_spinner("Processing")

        # Run via core
        self.core.run_async(
            cmd,
            on_stdout=lambda line: self.log_append(line),
            on_complete=lambda err: self.after(0, self._on_process_done, err, out_png),
        )

    def _on_process_done(self, err, out_png_path: str):
        try:
            if not err:
                self._display_image_native(out_png_path)
        finally:
            # Always stop spinner and re-enable Run
            self._stop_spinner()
            self.run_btn.state(["!disabled"])

    # ---------- Preview (right side) ----------

    def _clear_preview(self):
        if getattr(self, "_img_id", None) is not None:
            try:
                self.preview_canvas.delete(self._img_id)
            except tk.TclError:
                pass
            self._img_id = None
        self._tk_img_preview = None
        # reset scrollregion
        self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))

    def _display_image_native(self, path):
        try:
            if os.path.exists(path):
                if PIL_AVAILABLE:
                    self._base_pil = Image.open(path)  # keep original
                    self._zoom = 1.0  # reset to 100%
                    self._render_zoomed(center=None)
                    self.log_append("Done.")
                else:
                    # Fallback: display simple text if Pillow not available
                    self.preview_canvas.create_text(
                        10, 10, anchor="nw",
                        text=f"Image saved: {path}",
                        fill=self._palette.get("fg", "#111827")
                    )
                    self.log_append("Done.")
            else:
                self.log_append("Finished, but image not found.")
        except Exception as e:
            self.log_append(f"Image preview failed: {e}\nImage saved at: {path}")

    def _render_zoomed(self, center=None):
        """Render the base image at current zoom. Optionally keep a canvas point centered."""
        if not (PIL_AVAILABLE and self._base_pil):
            return

        # Resize with high-quality resampling
        w0, h0 = self._base_pil.width, self._base_pil.height
        w = max(1, int(w0 * self._zoom))
        h = max(1, int(h0 * self._zoom))

        img = self._base_pil.resize((w, h), Image.LANCZOS)
        self._tk_img_preview = ImageTk.PhotoImage(img)

        # First image draw or update existing
        if getattr(self, "_img_id", None) is None:
            self._img_id = self.preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_img_preview)
        else:
            self.preview_canvas.itemconfigure(self._img_id, image=self._tk_img_preview)

        # Update scroll region
        self.preview_canvas.configure(scrollregion=(0, 0, w, h))

        # Keep a point centered if requested
        if center is not None:
            cx, cy = center
            vx = self.preview_canvas.winfo_width()
            vy = self.preview_canvas.winfo_height()
            self.preview_canvas.xview_moveto(max(0, (cx - vx / 2) / max(1, w)))
            self.preview_canvas.yview_moveto(max(0, (cy - vy / 2) / max(1, h)))

    def _zoom_step(self, factor):
        """Zoom relative to the current center of viewport."""
        if not (PIL_AVAILABLE and self._base_pil):
            return
        old_zoom = self._zoom
        new_zoom = min(self._max_zoom, max(self._min_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        # Determine canvas center in image coords before resizing
        vx = self.preview_canvas.winfo_width()
        vy = self.preview_canvas.winfo_height()
        cx = self.preview_canvas.canvasx(vx / 2)
        cy = self.preview_canvas.canvasy(vy / 2)

        scale = new_zoom / (old_zoom or 1.0)
        cx *= scale
        cy *= scale
        self._zoom = new_zoom
        self._render_zoomed(center=(cx, cy))

    def _zoom_100(self):
        if not (PIL_AVAILABLE and self._base_pil):
            return
        self._zoom = 1.0
        self._render_zoomed(center=None)

    def _zoom_fit(self):
        """Fit the image into the visible canvas area."""
        if not (PIL_AVAILABLE and self._base_pil):
            return
        w0, h0 = self._base_pil.width, self._base_pil.height
        # Current viewport size
        vx = max(1, self.preview_canvas.winfo_width())
        vy = max(1, self.preview_canvas.winfo_height())
        # Fit scale
        sx = vx / w0
        sy = vy / h0
        new_zoom = max(self._min_zoom, min(self._max_zoom, min(sx, sy)))
        self._zoom = new_zoom
        self._render_zoomed(center=None)

    def _on_mousewheel_zoom(self, event):
        """Ctrl+MouseWheel to zoom."""
        if not (PIL_AVAILABLE and self._base_pil):
            return
        factor = 1.1 if event.delta > 0 else (1/1.1)
        self._zoom_step(factor)

    # ---------- Spinner ----------
    def _start_spinner(self, text_prefix="Processing"):
        if getattr(self, "_spinning", False):
            return
        self._spinning = True
        self._spinner_idx = 0
        self._spinner_var.set(f"{text_prefix} ⠋")
        try:
            self.spinner_label.pack_info()
        except tk.TclError:
            pass
        self.spinner_label.pack(side="left", padx=12)
        self._tick_spinner()

    def _tick_spinner(self):
        if not getattr(self, "_spinning", False):
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self._spinner_var.set(f"Processing {self._spinner_frames[self._spinner_idx]}")
        self.after(90, self._tick_spinner)

    def _stop_spinner(self):
        if not getattr(self, "_spinning", False):
            return
        self._spinning = False
        self._spinner_var.set("")
        try:
            self.spinner_label.pack_forget()
        except tk.TclError:
            pass

    # ---------- log helpers ----------
    def log_append(self, text):
        def _do():
            self.log.insert("end", text + "\n")
            self.log.see("end")
        self.after(0, _do)

    def log_delete(self):
        self.log.delete("1.0", "end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
