import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from sr import super_resolve, super_resolve_srcnn


PREVIEW_MAX_W = 480
PREVIEW_MAX_H = 480

MODE_SRCNN = "SRCNN (x2/x3)"
MODE_TRADITIONAL = "传统增强"

BG = "#f4f6fb"
PANEL = "#ffffff"
BORDER = "#e5e7eb"
TEXT = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#0f766e"
ACCENT_DARK = "#115e59"
STATUS_BG = "#eef1f6"


def fit_image(img, max_w=PREVIEW_MAX_W, max_h=PREVIEW_MAX_H):
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.BICUBIC)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("HighImage 图像超分辨率")
        self.root.geometry("1280x780")
        self.root.minsize(1120, 700)

        self.original = None
        self.output = None
        self._tk_orig = None
        self._tk_out = None

        self.scale_var = tk.IntVar(value=2)
        self.iters_var = tk.IntVar(value=10)
        self.sharp_amount_var = tk.DoubleVar(value=0.9)
        self.sharp_radius_var = tk.DoubleVar(value=1.3)
        self.detail_boost_var = tk.DoubleVar(value=0.25)
        self.ibp_lambda_var = tk.DoubleVar(value=1.0)
        self.mode_var = tk.StringVar(value=MODE_SRCNN)
        default_weights = os.path.join(
            os.path.dirname(__file__),
            "SRCNN-pytorch-master",
            "best.pth",
        )
        self.weights_var = tk.StringVar(value=default_weights)
        self.match_save_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪。")

        self._init_style()
        self._build_ui()

    def _init_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Microsoft YaHei UI", 10)
        header_font = ("Microsoft YaHei UI", 16, "bold")
        section_font = ("Microsoft YaHei UI", 10, "bold")

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)

        style.configure("TLabel", background=BG, foreground=TEXT, font=base_font)
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=header_font)
        style.configure("Subheader.TLabel", background=BG, foreground=MUTED, font=base_font)
        style.configure("CardTitle.TLabel", background=PANEL, foreground=TEXT, font=section_font)

        style.configure(
            "Card.TLabelframe",
            background=PANEL,
            relief="solid",
            bordercolor=BORDER,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=PANEL,
            foreground=TEXT,
            font=section_font,
        )

        style.configure(
            "TButton",
            background=ACCENT,
            foreground="white",
            padding=(12, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", ACCENT_DARK), ("!disabled", ACCENT)],
            foreground=[("!disabled", "white")],
        )

        style.configure(
            "TEntry",
            fieldbackground=PANEL,
            background=PANEL,
            foreground=TEXT,
        )
        style.configure(
            "TSpinbox",
            fieldbackground=PANEL,
            background=PANEL,
            foreground=TEXT,
            padding=2,
        )
        style.configure("TSeparator", background=BORDER)

        style.configure("Status.TFrame", background=STATUS_BG)
        style.configure("Status.TLabel", background=STATUS_BG, foreground=MUTED, font=base_font)

    def _build_ui(self):
        self.root.configure(bg=BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=20)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(header, text="HighImage", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="图像超分辨率与细节增强",
            style="Subheader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        sidebar = ttk.Frame(main, style="Card.TFrame", padding=12)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        sidebar.columnconfigure(0, weight=1)

        actions = ttk.Labelframe(sidebar, text="操作", style="Card.TLabelframe", padding=12)
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)

        ttk.Button(actions, text="加载图片", command=self.load_image).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(actions, text="开始超分", command=self.run_sr).grid(
            row=1, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(actions, text="保存结果", command=self.save_output).grid(
            row=2, column=0, sticky="ew"
        )
        ttk.Checkbutton(
            actions,
            text="保存为原图尺寸",
            variable=self.match_save_var,
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))

        params = ttk.Labelframe(sidebar, text="参数", style="Card.TLabelframe", padding=12)
        params.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        params.columnconfigure(0, weight=0)
        params.columnconfigure(1, weight=1)
        params.columnconfigure(2, weight=0)

        self._labeled_combo(
            params,
            "模式",
            self.mode_var,
            [MODE_SRCNN, MODE_TRADITIONAL],
            0,
        )
        self._labeled_path(params, "模型权重", self.weights_var, self._browse_weights, 1)
        self._labeled_spinbox(params, "放大倍数", self.scale_var, 2, 4, 1, 2)
        self._labeled_spinbox(params, "IBP 迭代", self.iters_var, 0, 30, 1, 3)
        self._labeled_entry(params, "IBP 强度", self.ibp_lambda_var, 4)
        self._labeled_entry(params, "锐化强度", self.sharp_amount_var, 5)
        self._labeled_entry(params, "锐化半径", self.sharp_radius_var, 6)
        self._labeled_entry(params, "细节增强", self.detail_boost_var, 7)

        preview = ttk.Frame(main, style="Card.TFrame", padding=12)
        preview.grid(row=1, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(1, weight=1)

        ttk.Label(preview, text="预览", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        orig_group = ttk.Labelframe(preview, text="输入图片", style="Card.TLabelframe", padding=8)
        orig_group.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        out_group = ttk.Labelframe(preview, text="输出图片", style="Card.TLabelframe", padding=8)
        out_group.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        self.orig_canvas = ttk.Label(orig_group, anchor=tk.CENTER)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        self.out_canvas = ttk.Label(out_group, anchor=tk.CENTER)
        self.out_canvas.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(self.root, style="Status.TFrame")
        status.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            status,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor=tk.W,
            padding=(12, 6),
        ).grid(row=0, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)

    def _labeled_spinbox(self, parent, label, var, min_v, max_v, inc, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        spin = ttk.Spinbox(
            parent,
            from_=min_v,
            to=max_v,
            increment=inc,
            textvariable=var,
            width=10,
            justify="center",
        )
        spin.grid(row=row, column=1, sticky="e", pady=4)

    def _labeled_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=10, justify="center")
        entry.grid(row=row, column=1, sticky="e", pady=4)
        return entry

    def _labeled_combo(self, parent, label, var, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=20)
        combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        return combo

    def _labeled_path(self, parent, label, var, browse_cmd, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4, padx=(0, 6))
        ttk.Button(parent, text="浏览", command=browse_cmd).grid(row=row, column=2, sticky="e", pady=4)
        return entry

    def _browse_weights(self):
        path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch 权重", "*.pth"), ("所有文件", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self.original = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return

        self.output = None
        self._show_original()
        self._show_output(clear=True)
        self.status_var.set(f"已加载：{self.original.size[0]}x{self.original.size[1]}。")

    def run_sr(self):
        if self.original is None:
            messagebox.showinfo("未加载图片", "请先加载一张图片。")
            return

        try:
            scale = int(self.scale_var.get())
            iters = int(self.iters_var.get())
            ibp_lambda = float(self.ibp_lambda_var.get())
            sharp_amount = float(self.sharp_amount_var.get())
            sharp_radius = float(self.sharp_radius_var.get())
            detail_boost = float(self.detail_boost_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "请检查数值参数是否填写正确。")
            return

        if scale < 2:
            messagebox.showerror("倍率错误", "放大倍数必须大于等于 2。")
            return

        mode = self.mode_var.get()
        if mode == MODE_SRCNN:
            self.status_var.set("正在运行 SRCNN (x2/x3)...")
        else:
            self.status_var.set("正在运行传统增强超分...")
        self.root.update_idletasks()

        try:
            if mode == MODE_SRCNN:
                weights_path = self.weights_var.get().strip()
                if not weights_path:
                    messagebox.showerror("缺少权重", "请选择 .pth 模型权重文件。")
                    self.status_var.set("就绪。")
                    return
                self.output = super_resolve_srcnn(
                    self.original,
                    scale=scale,
                    weights_path=weights_path,
                )
            else:
                self.output = super_resolve(
                    self.original,
                    scale=scale,
                    sharpen_amount=sharp_amount,
                    sharpen_radius=sharp_radius,
                    detail_boost=detail_boost,
                    ibp_iters=iters,
                    ibp_lambda=ibp_lambda,
                )
        except Exception as exc:
            messagebox.showerror("超分失败", str(exc))
            self.status_var.set("失败。")
            return

        self._show_output()
        self.status_var.set(
            f"完成：输出尺寸 {self.output.size[0]}x{self.output.size[1]}。"
        )

    def save_output(self):
        if self.output is None:
            messagebox.showinfo("没有可保存结果", "请先运行超分。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg;*.jpeg"), ("TIFF 图片", "*.tif;*.tiff"), ("BMP 图片", "*.bmp")],
        )
        if not path:
            return
        try:
            to_save = self.output
            if self.match_save_var.get() and self.original is not None:
                to_save = self.output.resize(self.original.size, Image.LANCZOS)
            to_save.save(path)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        if self.match_save_var.get() and self.original is not None:
            self.status_var.set(f"已保存到 {path}（已匹配原图尺寸）。")
        else:
            self.status_var.set(f"已保存到 {path}。")

    def _show_original(self):
        preview = fit_image(self.original)
        self._tk_orig = ImageTk.PhotoImage(preview)
        self.orig_canvas.configure(image=self._tk_orig)

    def _show_output(self, clear=False):
        if clear:
            self.out_canvas.configure(image="")
            self._tk_out = None
            return
        display_img = self.output
        if self.original is not None and self.output.size != self.original.size:
            # Keep algorithm output intact; resize only for side-by-side GUI comparison.
            display_img = self.output.resize(self.original.size, Image.LANCZOS)
        preview = fit_image(display_img)
        self._tk_out = ImageTk.PhotoImage(preview)
        self.out_canvas.configure(image=self._tk_out)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
