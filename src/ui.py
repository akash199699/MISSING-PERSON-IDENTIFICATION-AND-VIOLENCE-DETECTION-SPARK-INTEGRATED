import platform
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import torch
from PIL import Image, ImageTk
import numpy as np
import time
import gc
import sys

from missing_person_detection import setup_missing_person_detection, process_video
from violence_detection import load_violence_detection_model, detect_violence_in_video
from report_generation import export_to_pdf, export_violence_report, generate_combined_report
from spark_processing import initialize_spark, run_distributed_pipeline
from stats import stats_monitor, calculate_performance_stats
from config import config

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

class ColorTheme:
    def __init__(self, is_dark=True):
        theme = {
            'dark': {
                'BG_DARK': "#121212", 'BG_MEDIUM': "#1E1E1E", 'BG_LIGHT': "#2D2D2D",
                'TEXT': "#FFFFFF", 'ACCENT': "#7E57C2", 'SUCCESS': "#4CAF50",
                'ERROR': "#F44336", 'WARNING': "#FF9800", 'INFO': "#2196F3"
            },
            'light': {
                'BG_DARK': "#F5F5F5", 'BG_MEDIUM': "#E0E0E0", 'BG_LIGHT': "#FFFFFF",
                'TEXT': "#212121", 'ACCENT': "#673AB7", 'SUCCESS': "#4CAF50",
                'ERROR': "#F44336", 'WARNING': "#FF9800", 'INFO': "#2196F3"
            }
        }
        selected = theme['dark' if is_dark else 'light']
        for k, v in selected.items():
            setattr(self, k, v)

class MissingPersonDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.theme = ColorTheme(is_dark=True)
        self.models = {'missing_person': None, 'violence': None}
        self.setup_variables()
        self.apply_theme_style()
        self.create_ui()
        
    def setup_window(self):
        self.root.title("Missing Person & Violence Detection")
        self.root.geometry("1100x800")
        self.setup_high_dpi()
        
    def setup_high_dpi(self):
        if sys.platform.startswith('win'):
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass
        elif sys.platform.startswith('linux'):
            try:
                os.environ['GDK_SCALE'] = '2'
            except:
                pass

    def setup_variables(self):
        self.ref_files = []
        self.video_files = []
        self.running = False
        self.start_time = None
        self.num_detections = 0
        self.is_dark_theme = tk.BooleanVar(value=True)
        
        self.metrics_vars = {
            'accuracy': tk.StringVar(value="Accuracy: -"),
            'precision': tk.StringVar(value="Precision: -"),
            'recall': tk.StringVar(value="Recall: -"),
            'fps': tk.StringVar(value="FPS: -"),
            'detections': tk.StringVar(value="Detections: TP=0 FP=0"),
            'total_time': tk.StringVar(value="Total Time: -"),
            'total_frames': tk.StringVar(value="Frames: -")
        }
        
        self.detection_threshold = tk.DoubleVar(value=0.65)
        self.frame_interval = tk.IntVar(value=60)
        self.preload_models = tk.BooleanVar(value=False)
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.status_text = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.recommendation_var = tk.StringVar(value="Recommendations: -")
        self.display_video = tk.BooleanVar(value=True)
        self.batch_processing = tk.BooleanVar(value=bool(torch.cuda.is_available()))
        self.preview_image = None
        
        # Spark config
        spark_conf = config.SPARK_CONF
        self.spark_master = tk.StringVar(value=spark_conf.get("master", "local[*]"))
        self.spark_executor_memory = tk.StringVar(value=spark_conf.get("executor.memory", "4g"))
        self.spark_executor_cores = tk.IntVar(value=int(spark_conf.get("executor.cores", 2)))
        self.spark_executor_instances = tk.IntVar(value=int(spark_conf.get("spark.executor.instances", 2)))

    def apply_theme_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TCombobox', 
                      fieldbackground=self.theme.BG_LIGHT,
                      background=self.theme.BG_LIGHT,
                      foreground=self.theme.TEXT,
                      arrowcolor=self.theme.ACCENT)
        
        style.configure("TProgressbar", 
                       troughcolor=self.theme.BG_LIGHT,
                       background=self.theme.ACCENT)
        
        style.configure("TNotebook", 
                       background=self.theme.BG_DARK, 
                       tabmargins=[2, 5, 2, 0])
        
        style.configure("TNotebook.Tab", 
                       background=self.theme.BG_LIGHT,
                       foreground=self.theme.TEXT,
                       padding=[10, 4],
                       font=('Arial', 10))
        
        style.map("TNotebook.Tab",
                background=[("selected", self.theme.ACCENT)],
                foreground=[("selected", self.theme.TEXT)])

    def create_ui(self):
        self.create_menu()
        self.create_header()
        self.create_main_frame()
        self.create_status_bar()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select Reference Images", command=self.select_reference_images)
        file_menu.add_command(label="Select Video Files", command=self.select_video_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_checkbutton(label="Dark Theme", variable=self.is_dark_theme, command=self.toggle_theme)
        settings_menu.add_checkbutton(label="Preload Models", variable=self.preload_models, command=self.handle_preload_models)
        settings_menu.add_checkbutton(label="Batch Processing", variable=self.batch_processing)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=lambda: self.open_url("https://example.com/documentation"))
        
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def toggle_theme(self):
        self.theme = ColorTheme(is_dark=self.is_dark_theme.get())
        self.root.configure(bg=self.theme.BG_DARK)
        self.update_widget_theme(self.root)

    def update_widget_theme(self, widget):
        try:
            if isinstance(widget, (tk.Frame, tk.LabelFrame)):
                widget.configure(bg=self.theme.BG_DARK)
            elif isinstance(widget, tk.Label):
                widget.configure(bg=widget.master['bg'], fg=self.theme.TEXT)
            elif isinstance(widget, tk.Button) and not isinstance(widget, ttk.Button):
                widget.configure(bg=self.theme.BG_LIGHT, fg=self.theme.TEXT)
            elif isinstance(widget, tk.Checkbutton):
                widget.configure(bg=widget.master['bg'], fg=self.theme.TEXT, selectcolor=self.theme.BG_DARK)
            elif isinstance(widget, tk.Scale):
                widget.configure(bg=self.theme.BG_LIGHT, fg=self.theme.TEXT, troughcolor=self.theme.BG_DARK)
        except:
            pass
            
        for child in widget.winfo_children():
            self.update_widget_theme(child)

    def create_header(self):
        header_frame = tk.Frame(self.root, bg=self.theme.BG_MEDIUM, height=70)
        header_frame.pack(fill=tk.X)
        
        # Logo and title
        tk.Label(header_frame, text="🔍", font=("Arial", 24),
                fg=self.theme.ACCENT, bg=self.theme.BG_MEDIUM, padx=20).pack(side=tk.LEFT)
        
        tk.Label(header_frame, text="Advanced Detection System", 
                font=("Arial", 22, "bold"), fg=self.theme.TEXT,
                bg=self.theme.BG_MEDIUM, pady=15).pack(side=tk.LEFT)
        
        tk.Label(header_frame, text="PRID", font=("Arial", 12),
                fg="#AAAAAA", bg=self.theme.BG_MEDIUM, pady=15, padx=10).pack(side=tk.LEFT)
        
        # System info
        info_frame = tk.Frame(header_frame, bg=self.theme.BG_MEDIUM)
        info_frame.pack(side=tk.RIGHT, padx=15)
        
        gpu_status = "Available ✓" if torch.cuda.is_available() else "Not Available ✗"
        tk.Label(info_frame, text=f"GPU: {gpu_status}", 
                fg=self.theme.SUCCESS if torch.cuda.is_available() else self.theme.ERROR,
                bg=self.theme.BG_MEDIUM, font=("Arial", 10)).pack(anchor=tk.E)
        
        if torch.cuda.is_available():
            tk.Label(info_frame, text=f"CUDA: {torch.cuda.get_device_name(0)}", 
                    fg=self.theme.INFO, bg=self.theme.BG_MEDIUM, 
                    font=("Arial", 10)).pack(anchor=tk.E)

    def create_main_frame(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detection tab
        main_tab = tk.Frame(self.notebook, bg=self.theme.BG_DARK)
        self.notebook.add(main_tab, text="Detection")
        
        # Settings tab
        settings_tab = tk.Frame(self.notebook, bg=self.theme.BG_DARK)
        self.notebook.add(settings_tab, text="Advanced Settings")
        
        # Main tab content
        main_frame = tk.Frame(main_tab, bg=self.theme.BG_DARK, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        settings_frame = self.create_labeled_frame(main_frame, "Settings", 0.4)
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.create_settings_components(settings_frame)
        
        # Right panel
        files_frame = self.create_labeled_frame(main_frame, "Files", 0.6)
        files_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.create_files_components(files_frame)
        
        # Advanced settings
        self.create_advanced_settings(settings_tab)

    def create_labeled_frame(self, parent, title, weight=1):
        frame = tk.Frame(parent, bg=self.theme.BG_DARK, padx=5, pady=5)
        
        title_bar = tk.Frame(frame, bg=self.theme.ACCENT, height=30)
        title_bar.pack(fill=tk.X)
        
        tk.Label(title_bar, text=title, font=("Arial", 12, "bold"),
               fg=self.theme.TEXT, bg=self.theme.ACCENT, pady=5).pack(side=tk.LEFT, padx=10)
        
        content_frame = tk.Frame(frame, bg=self.theme.BG_MEDIUM, padx=10, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        frame.content = content_frame
        
        return frame

    def create_settings_components(self, parent):
        content = parent.content
        
        # Detection mode
        detection_frame = tk.Frame(content, bg=self.theme.BG_MEDIUM, pady=5)
        detection_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(detection_frame, text="Detection Mode:", 
                bg=self.theme.BG_MEDIUM, fg=self.theme.TEXT, font=("Arial", 10)).pack(side=tk.LEFT)
        
        modes = ["Full Pipeline", "Missing Person Only", "Violence Only"]
        if SPARK_AVAILABLE:
            modes.append("Spark Distributed")
            
        self.detection_mode = tk.StringVar(value="Missing Person Only")
        mode_combo = ttk.Combobox(detection_frame, textvariable=self.detection_mode, 
                                values=modes, state="readonly", width=15)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        # GPU and display options
        option_frame = tk.Frame(content, bg=self.theme.BG_MEDIUM, pady=5)
        option_frame.pack(fill=tk.X, pady=5)
        
        gpu_state = tk.NORMAL if torch.cuda.is_available() else tk.DISABLED
        tk.Checkbutton(option_frame, text=f"Use GPU {'(Available)' if torch.cuda.is_available() else '(Not Available)'}",
                     variable=self.use_gpu, bg=self.theme.BG_MEDIUM, fg=self.theme.TEXT,
                     selectcolor=self.theme.BG_DARK, state=gpu_state).pack(side=tk.LEFT)
        
        tk.Checkbutton(option_frame, text="Display Real-Time Video", variable=self.display_video,
                     bg=self.theme.BG_MEDIUM, fg=self.theme.TEXT,
                     selectcolor=self.theme.BG_DARK).pack(side=tk.RIGHT, padx=5)

        # Threshold and interval sliders
        for label, var, from_, to, res in [
            ("Detection Threshold:", self.detection_threshold, 0.5, 0.95, 0.01),
            ("Frame Interval:", self.frame_interval, 1, 600, 10)
        ]:
            frame = tk.Frame(content, bg=self.theme.BG_MEDIUM, pady=5)
            frame.pack(fill=tk.X, pady=5)
            
            header_frame = tk.Frame(frame, bg=self.theme.BG_MEDIUM)
            header_frame.pack(fill=tk.X)
            
            tk.Label(header_frame, text=label, bg=self.theme.BG_MEDIUM,
                    fg=self.theme.TEXT, font=("Arial", 10)).pack(side=tk.LEFT)
            
            value_label = tk.Label(header_frame, text=f"{var.get():.2f}" if res < 1 else f"{var.get():.0f}", 
                                 bg=self.theme.BG_MEDIUM, fg=self.theme.ACCENT)
            value_label.pack(side=tk.RIGHT)
            
            slider = tk.Scale(frame, variable=var, from_=from_, to=to, resolution=res,
                           orient=tk.HORIZONTAL, bg=self.theme.BG_LIGHT, fg=self.theme.TEXT,
                           highlightthickness=0, troughcolor=self.theme.BG_DARK,
                           activebackground=self.theme.ACCENT)
            slider.pack(fill=tk.X)
            
            slider.config(command=lambda val, label=value_label, r=res: 
                        label.config(text=f"{float(val):.2f}" if r < 1 else f"{float(val):.0f}"))

        # Action buttons
        action_frame = tk.Frame(content, bg=self.theme.BG_MEDIUM, pady=10)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.run_button = tk.Button(action_frame, text="Run Detection", command=self.run_detection,
                                 bg=self.theme.SUCCESS, fg=self.theme.TEXT, font=("Arial", 12, "bold"),
                                 pady=10, relief=tk.FLAT, cursor="hand2")
        self.run_button.pack(fill=tk.X, pady=5)
        
        tk.Button(action_frame, text="Exit", command=self.root.destroy,
                bg=self.theme.ERROR, fg=self.theme.TEXT, font=("Arial", 12),
                pady=5, relief=tk.FLAT, cursor="hand2").pack(fill=tk.X, pady=5)

    def create_files_components(self, parent):
        content = parent.content
        
        # Reference images
        self.create_file_selector(content, "Reference Images:", "Select Reference Images",
                                self.select_reference_images, self.ref_files, "ref_label")
        
        # Video files
        self.create_file_selector(content, "Video Files:", "Select Video Files",
                                self.select_video_files, self.video_files, "video_label")
        
        # Preview area
        preview_frame = self.create_labeled_frame(content, "Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        preview_content = preview_frame.content
        
        # Preview controls
        controls_frame = tk.Frame(preview_content, bg=self.theme.BG_MEDIUM)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.preview_index = 0
        self.preview_files = []
        
        nav_frame = tk.Frame(controls_frame, bg=self.theme.BG_MEDIUM)
        nav_frame.pack(side=tk.RIGHT)
        
        self.prev_btn = tk.Button(nav_frame, text="◀", command=self.prev_preview,
                               bg=self.theme.BG_LIGHT, fg=self.theme.TEXT, width=3,
                               state=tk.DISABLED, cursor="hand2")
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.preview_counter = tk.Label(nav_frame, text="0/0", bg=self.theme.BG_MEDIUM,
                                     fg=self.theme.TEXT, width=5)
        self.preview_counter.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(nav_frame, text="▶", command=self.next_preview,
                               bg=self.theme.BG_LIGHT, fg=self.theme.TEXT, width=3,
                               state=tk.DISABLED, cursor="hand2")
        self.next_btn.pack(side=tk.LEFT, padx=2)
        
        # Canvas for preview
        preview_canvas_frame = tk.Frame(preview_content, bg=self.theme.BG_DARK, bd=1, relief=tk.SUNKEN)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_canvas_frame, bg=self.theme.BG_DARK, 
                                     highlightthickness=0, bd=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = tk.Label(self.preview_canvas, text="Select files to see preview",
                                   bg=self.theme.BG_DARK, fg=self.theme.TEXT, height=10)
        self.preview_canvas.create_window((0, 0), window=self.preview_label, anchor='nw')
        
        # Performance panel
        self.create_performance_panel(content)

    def create_file_selector(self, parent, label_text, button_text, command, files_var, label_attr):
        frame = tk.Frame(parent, bg=self.theme.BG_MEDIUM)
        frame.pack(fill=tk.X, pady=5)
        
        tk.Label(frame, text=label_text, bg=self.theme.BG_MEDIUM,
                fg=self.theme.TEXT, font=("Arial", 10)).pack(anchor=tk.W)
        
        label_frame = tk.Frame(frame, bg=self.theme.BG_LIGHT, relief=tk.FLAT, padx=5, pady=5)
        label_frame.pack(fill=tk.X, pady=2)
        
        label = tk.Label(label_frame, text="No files selected", bg=self.theme.BG_LIGHT,
                       fg=self.theme.TEXT, anchor=tk.W)
        label.pack(fill=tk.X)
        setattr(self, label_attr, label)
        
        button_frame = tk.Frame(frame, bg=self.theme.BG_MEDIUM)
        button_frame.pack(fill=tk.X)
        
        tk.Button(button_frame, text=button_text, command=command, cursor="hand2",
                bg=self.theme.INFO, fg=self.theme.TEXT, relief=tk.FLAT).pack(side=tk.LEFT)
        
        tk.Button(button_frame, text="Clear", command=lambda: self.clear_files(files_var, label_attr),
                bg=self.theme.WARNING, fg=self.theme.TEXT, relief=tk.FLAT,
                cursor="hand2").pack(side=tk.LEFT, padx=5)

    def clear_files(self, files_var, label_attr):
        if isinstance(files_var, list):
            files_var.clear()
        
        label = getattr(self, label_attr)
        label.config(text="No files selected")
        
        if label_attr == "ref_label":
            self.preview_files = []
            self.preview_index = 0
            self.update_preview_controls()
            self.preview_label.config(image="", text="No reference images selected")
            self.preview_image = None
        
    def select_reference_images(self):
        files = filedialog.askopenfilenames(title="Select Reference Images", 
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.ref_files = list(files)
            self.update_file_label(self.ref_label, files)
            
            # Update preview
            self.preview_files = self.ref_files
            self.preview_index = 0
            self.show_preview(files[0])
            self.update_preview_controls()

    def select_video_files(self):
        files = filedialog.askopenfilenames(title="Select Video Files",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if files:
            self.video_files = list(files)
            self.update_file_label(self.video_label, files)

    def update_file_label(self, label, files):
        label.config(text=os.path.basename(files[0])) if len(files) == 1 else label.config(text=f"{len(files)} files selected")

    def show_preview(self, file_path):
        try:
            img = Image.open(file_path)
            
            # Calculate size to fit in canvas
            canvas_width = self.preview_canvas.winfo_width() or 300
            canvas_height = self.preview_canvas.winfo_height() or 200
            
            img_width, img_height = img.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=self.preview_image, text="", compound=tk.TOP,
                                   width=new_size[0], height=new_size[1])
            
            # Show file info
            file_info = f"{os.path.basename(file_path)} - {img_width}x{img_height}"
            self.preview_label.config(text=file_info)
            
        except Exception as e:
            print(f"Error loading preview: {e}")
            self.preview_label.config(image="", text=f"Error loading image: {e}")

    def prev_preview(self):
        if self.preview_files and self.preview_index > 0:
            self.preview_index -= 1
            self.show_preview(self.preview_files[self.preview_index])
            self.update_preview_controls()

    def next_preview(self):
        if self.preview_files and self.preview_index < len(self.preview_files) - 1:
            self.preview_index += 1
            self.show_preview(self.preview_files[self.preview_index])
            self.update_preview_controls()

    def update_preview_controls(self):
        if not self.preview_files:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.preview_counter.config(text="0/0")
            return
            
        self.preview_counter.config(text=f"{self.preview_index + 1}/{len(self.preview_files)}")
        self.prev_btn.config(state=tk.NORMAL if self.preview_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.preview_index < len(self.preview_files) - 1 else tk.DISABLED)

    def create_advanced_settings(self, parent):
        advanced_frame = tk.Frame(parent, bg=self.theme.BG_DARK, padx=20, pady=20)
        advanced_frame.pack(fill=tk.BOTH, expand=True)
        
        # Performance Optimization
        perf_frame = self.create_labeled_frame(advanced_frame, "Performance Optimization")
        perf_frame.pack(fill=tk.X, pady=10)
        
        # Batch size option if GPU available
        if torch.cuda.is_available():
            batch_frame = tk.Frame(perf_frame.content, bg=self.theme.BG_MEDIUM)
            batch_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(batch_frame, text="Batch Size:", bg=self.theme.BG_MEDIUM,
                   fg=self.theme.TEXT).pack(side=tk.LEFT, padx=5)
            
            self.batch_size = tk.IntVar(value=4)
            ttk.Combobox(batch_frame, textvariable=self.batch_size,
                        values=[1, 2, 4, 8, 16, 32], width=5).pack(side=tk.LEFT, padx=5)
            
            tk.Checkbutton(batch_frame, text="Enable Batch Processing", 
                         variable=self.batch_processing,
                         bg=self.theme.BG_MEDIUM, fg=self.theme.TEXT,
                         selectcolor=self.theme.BG_DARK).pack(side=tk.LEFT, padx=10)
        
        # Memory management
        memory_frame = tk.Frame(perf_frame.content, bg=self.theme.BG_MEDIUM)
        memory_frame.pack(fill=tk.X, pady=5)
        
        self.clear_cuda_memory = tk.BooleanVar(value=True)
        tk.Checkbutton(memory_frame, text="Clear CUDA memory after processing",
                     variable=self.clear_cuda_memory, bg=self.theme.BG_MEDIUM,
                     fg=self.theme.TEXT, selectcolor=self.theme.BG_DARK).pack(side=tk.LEFT)
        
        self.clear_cache = tk.BooleanVar(value=True)
        tk.Checkbutton(memory_frame, text="Clear cache after processing",
                     variable=self.clear_cache, bg=self.theme.BG_MEDIUM,
                     fg=self.theme.TEXT, selectcolor=self.theme.BG_DARK).pack(side=tk.LEFT, padx=10)
        
        # Model preloading
        model_frame = tk.Frame(perf_frame.content, bg=self.theme.BG_MEDIUM)
        model_frame.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(model_frame, text="Preload models at startup",
                      variable=self.preload_models, bg=self.theme.BG_MEDIUM,
                      fg=self.theme.TEXT, selectcolor=self.theme.BG_DARK).pack(side=tk.LEFT)
        
        tk.Button(model_frame, text="Force Load Models Now",
                 command=self.preload_all_models, bg=self.theme.INFO,
                 fg=self.theme.TEXT, relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT, padx=10)
        
        # Distributed processing (Spark)
        if SPARK_AVAILABLE:
            spark_frame = self.create_labeled_frame(advanced_frame, "Distributed Processing (Spark)")
            spark_frame.pack(fill=tk.X, pady=10)
            
            grid_frame = tk.Frame(spark_frame.content, bg=self.theme.BG_MEDIUM)
            grid_frame.pack(fill=tk.X)
            
            # Create a grid for Spark settings
            labels = ["Master URL:", "Executor Memory:", "Executor Cores:", "Executor Instances:"]
            vars = [self.spark_master, self.spark_executor_memory, 
                   self.spark_executor_cores, self.spark_executor_instances]
            
            for i, (label, var) in enumerate(zip(labels, vars)):
                tk.Label(grid_frame, text=label, bg=self.theme.BG_MEDIUM,
                       fg=self.theme.TEXT).grid(row=i, column=0, sticky=tk.W, pady=5, padx=5)
                
                if isinstance(var, tk.IntVar):
                    tk.Spinbox(grid_frame, from_=1, to=16, textvariable=var, width=10).grid(row=i, column=1, sticky=tk.W, padx=5)
                else:
                    tk.Entry(grid_frame, textvariable=var, width=20).grid(row=i, column=1, sticky=tk.W, padx=5)
            
            # Test Spark connection button
            tk.Button(spark_frame.content, text="Test Spark Connection",
                    command=self.test_spark_connection, bg=self.theme.INFO,
                    fg=self.theme.TEXT, relief=tk.FLAT, cursor="hand2").pack(anchor=tk.W, pady=10)
        
        # Export Settings
        export_frame = self.create_labeled_frame(advanced_frame, "Export Settings")
        export_frame.pack(fill=tk.X, pady=10)
        
        export_content = export_frame.content
        
        self.auto_export = tk.BooleanVar(value=True)
        tk.Checkbutton(export_content, text="Auto export results after processing",
                     variable=self.auto_export, bg=self.theme.BG_MEDIUM,
                     fg=self.theme.TEXT, selectcolor=self.theme.BG_DARK).pack(anchor=tk.W)
        
        format_frame = tk.Frame(export_content, bg=self.theme.BG_MEDIUM, pady=5)
        format_frame.pack(fill=tk.X)
        
        tk.Label(format_frame, text="Export Format:", bg=self.theme.BG_MEDIUM,
               fg=self.theme.TEXT).pack(side=tk.LEFT)
        
        self.export_format = tk.StringVar(value="PDF")
        ttk.Combobox(format_frame, textvariable=self.export_format,
                    values=["PDF", "CSV", "JSON"], width=10, state="readonly").pack(side=tk.LEFT, padx=5)
        
        export_dir_frame = tk.Frame(export_content, bg=self.theme.BG_MEDIUM, pady=5)
        export_dir_frame.pack(fill=tk.X)
        
        tk.Label(export_dir_frame, text="Export Directory:", bg=self.theme.BG_MEDIUM,
               fg=self.theme.TEXT).pack(side=tk.LEFT)
        
        self.export_dir = tk.StringVar(value=os.path.join(os.getcwd(), "reports"))
        tk.Entry(export_dir_frame, textvariable=self.export_dir, width=30).pack(side=tk.LEFT, padx=5)
        
        tk.Button(export_dir_frame, text="Browse", command=self.select_export_dir,
                bg=self.theme.INFO, fg=self.theme.TEXT, relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT)

    def select_export_dir(self):
        directory = filedialog.askdirectory(title="Select Export Directory")
        if directory:
            self.export_dir.set(directory)

    def test_spark_connection(self):
        try:
            spark = SparkSession.builder \
                .master(self.spark_master.get()) \
                .appName("ConnectionTest") \
                .config("spark.executor.memory", self.spark_executor_memory.get()) \
                .config("spark.executor.cores", str(self.spark_executor_cores.get())) \
                .config("spark.executor.instances", str(self.spark_executor_instances.get())) \
                .getOrCreate()
            
            test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
            count = test_df.count()
            spark.stop()
            
            messagebox.showinfo("Spark Connection", f"Connection successful! Test dataframe has {count} row(s).")
        except Exception as e:
            messagebox.showerror("Spark Connection Error", f"Failed to connect to Spark cluster.\nError: {str(e)}")

    def create_performance_panel(self, parent):
        metrics_frame = self.create_labeled_frame(parent, "Performance Metrics")
        metrics_frame.pack(fill=tk.X, pady=5)
        
        content = metrics_frame.content
        
        # Create a 2x3 grid for metrics
        grid_frame = tk.Frame(content, bg=self.theme.BG_MEDIUM)
        grid_frame.pack(fill=tk.X, pady=5)
        
        metrics = [
            ("accuracy", "Accuracy: -", 0, 0),
            ("precision", "Precision: -", 0, 1),
            ("recall", "Recall: -", 0, 2),
            ("fps", "FPS: -", 1, 0),
            ("total_time", "Total Time: -", 1, 1),
            ("total_frames", "Frames: -", 1, 2)
        ]
        
        for var_name, default, row, col in metrics:
            self.metrics_vars[var_name] = tk.StringVar(value=default)
            tk.Label(grid_frame, textvariable=self.metrics_vars[var_name],
                   bg=self.theme.BG_MEDIUM, fg=self.theme.INFO, padx=5, pady=5).grid(row=row, column=col, sticky=tk.W)
        
        # Detection specific metrics
        detection_frame = tk.Frame(content, bg=self.theme.BG_MEDIUM)
        detection_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(detection_frame, textvariable=self.metrics_vars["detections"],
               bg=self.theme.BG_MEDIUM, fg=self.theme.INFO).pack(side=tk.LEFT, padx=5)
        
        # Recommendations
        tk.Label(content, textvariable=self.recommendation_var,
               bg=self.theme.BG_MEDIUM, fg=self.theme.WARNING, wraplength=400).pack(pady=5)

    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg=self.theme.BG_MEDIUM, height=25)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.progress = ttk.Progressbar(status_frame, variable=self.progress_var,
                                      mode='determinate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)
        
        status_label = tk.Label(status_frame, textvariable=self.status_text, 
                              bg=self.theme.BG_MEDIUM, fg=self.theme.TEXT, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=5)
        
        # Memory usage indicator
        self.memory_var = tk.StringVar(value="Mem: - / -")
        tk.Label(status_frame, textvariable=self.memory_var,
               bg=self.theme.BG_MEDIUM, fg=self.theme.INFO).pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Start memory monitoring
        self.monitor_resources()

    def monitor_resources(self):
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                self.memory_var.set(f"CUDA: {allocated:.2f}GB / {reserved:.2f}GB")
            else:
                import psutil
                vm = psutil.virtual_memory()
                self.memory_var.set(f"Mem: {vm.percent}%")
        except Exception as e:
            print(f"Error monitoring resources: {e}")
        
        self.root.after(2000, self.monitor_resources)

    def handle_preload_models(self):
        if self.preload_models.get() and messagebox.askyesno(
            "Load Models", 
            "Do you want to load the models now?\nThis may take a few minutes."
        ):
            self.preload_all_models()

    def preload_all_models(self):
        threading.Thread(target=self._preload_models_task, daemon=True).start()

    def _preload_models_task(self):
        self.status_text.set("Loading models...")
        self.progress_var.set(0)
        
        try:
            if not self.models['missing_person']:
                self.status_text.set("Loading missing person detection model...")
                self.models['missing_person'] = setup_missing_person_detection(
                    use_gpu=self.use_gpu.get()
                )
                self.progress_var.set(50)
            
            if not self.models['violence']:
                self.status_text.set("Loading violence detection model...")
                use_gpu=self.use_gpu.get()
                device = torch.device('cuda' if use_gpu else 'cpu')
                self.models['violence'] = load_violence_detection_model(device)
                self.progress_var.set(100)
            
            self.status_text.set("All models loaded successfully!")
            messagebox.showinfo("Models Loaded", "All models have been loaded successfully!")
        except Exception as e:
            self.status_text.set(f"Error loading models: {str(e)}")
            messagebox.showerror("Loading Error", f"Failed to load models:\n{str(e)}")
        finally:
            self.progress_var.set(0)
            self.status_text.set("Ready")

    def on_mode_change(self, event=None):
        mode = self.detection_mode.get()
        
        if mode == "Spark Distributed" and SPARK_AVAILABLE:
            self.notebook.select(1)
            messagebox.showinfo("Spark Mode", 
                              "Please configure Spark settings in the Advanced Settings tab.")

    def run_detection(self):
        # Validation
        mode = self.detection_mode.get()
        
        if mode in ["Full Pipeline", "Missing Person Only"] and not self.ref_files:
            messagebox.showerror("Error", "Please select reference images for missing person detection.")
            return
            
        if not self.video_files:
            messagebox.showerror("Error", "Please select video files to analyze.")
            return
        
        if self.running:
            messagebox.showinfo("Info", "Detection is already running.")
            return
        
        # Check for Spark configuration if using distributed mode
        if mode == "Spark Distributed" and not SPARK_AVAILABLE:
            messagebox.showerror("Error", "PySpark is not installed. Please install it to use distributed processing.")
            return

        # Save the current settings to persist across mode changes
        self.last_used_ref_files = self.ref_files
        self.last_used_video_files = self.video_files

        # Start detection in a separate thread
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_text.set("Processing...")
        
        detection_thread = threading.Thread(target=self.execute_detection)
        detection_thread.daemon = True
        detection_thread.start()
    
    def execute_detection(self):
        try:
            self.start_time = time.time()  # Record the start time
            self.num_detections = 0  # Reset the detection counter

            mode = self.detection_mode.get()
            threshold = self.detection_threshold.get()
            frame_interval = self.frame_interval.get()
            use_gpu = self.use_gpu.get()
            display_video = self.display_video.get()
        
            # Set device based on GPU selection
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            
            if mode == "Spark Distributed" and SPARK_AVAILABLE:
                spark = initialize_spark()
                try:
                    # Initial run for missing person detection
                    missing_detections, _, ref_filenames = run_distributed_pipeline(
                        spark, self.video_files, self.ref_files, run_violence=False
                    )
                    
                    output_dir = "Output"
                    # Generate missing person report
                    if missing_detections:
                        pdf_path = os.path.join(output_dir, "missing_person_report.pdf")
                        export_to_pdf(missing_detections, ref_filenames=ref_filenames, pdf_filename=pdf_path)
                    
                    # Prompt for violence detection
                    self.root.after(0, self.prompt_violence_detection, missing_detections)
                    
                except Exception as e:
                    self.root.after(0, self.detection_error, str(e))
                finally:
                    spark.stop()
                return

            # Initialize detection models first
            self.root.after(0, lambda: self.status_text.set("Loading models..."))
            device, mtcnn, resnet = setup_missing_person_detection()

            output_dir = "Output"
            os.makedirs(output_dir, exist_ok=True)

            if mode in ["Full Pipeline", "Missing Person Only"]:
                # Process reference images
                ref_embeddings = []
                self.root.after(0, lambda: self.status_text.set("Processing reference images..."))
            
                for ref_filename in self.ref_files:
                    ref_img = Image.open(ref_filename).convert("RGB")
                    faces, probs = mtcnn(ref_img, return_prob=True)
                    if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
                        continue
                    
                    # Use highest probability face if multiple are detected
                    ref_face = faces[int(np.argmax(probs))] if faces.ndim == 4 else faces
                    with torch.no_grad():
                        emb = resnet(
                            ref_face.unsqueeze(0).to(device).half() if device.type=='cuda'
                            else ref_face.unsqueeze(0).to(device)
                        )
                    ref_embeddings.append(emb)
            
                if not ref_embeddings:
                    raise Exception("No valid faces detected in the reference images.")
                
                # Convert reference embeddings to half precision if on GPU
                if device.type == 'cuda':
                    ref_embeddings = [emb.half() for emb in ref_embeddings]
                
                # Process each video
                all_detections = []
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Processing video: {os.path.basename(v)}..."))
                
                    # Call process_video function with parameters
                    detections = process_video(
                        video_file, 
                        mtcnn, 
                        resnet, 
                        device, 
                        ref_embeddings, 
                        frame_interval=frame_interval,
                        detection_threshold=threshold
                    )
                    all_detections.extend(detections)
                    self.num_detections += len(detections)  # Update the detection count
                
                # Export results    
                if all_detections:
                    self.root.after(0, lambda: self.status_text.set("Generating report..."))
                    all_detections.sort(key=lambda x: x['similarity'], reverse=True)
                    export_to_pdf(all_detections, ref_filenames=self.ref_files)
                
            if mode in ["Full Pipeline", "Violence Only"]:
                # Violence detection part
                self.root.after(0, lambda: self.status_text.set("Detecting violence..."))
            
                # Load violence model
                model = load_violence_detection_model(device)
            
                # Store all violence detections for potential combined report
                all_violence_detections = []

                # Process each video
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Checking violence in: {os.path.basename(v)}..."))
                    violence_detections = detect_violence_in_video(
                        video_file, 
                        model, 
                        device, 
                        threshold=threshold
                    )
                    self.num_detections += len(violence_detections)  # Update the detection count
                    all_violence_detections.extend(violence_detections)
                
                # Generate a combined violence report if needed
                if all_violence_detections and len(self.video_files) > 1:
                    combined_violence_path = os.path.join(output_dir, "violence_combined_report.pdf")
                    export_violence_report(all_violence_detections, "Multiple Videos", pdf_filename=combined_violence_path)

            # If both types of detections were run, consider generating a combined report
            if mode == "Full Pipeline" and (all_detections or all_violence_detections):
                combined_path = os.path.join(output_dir, "combined_analysis_report.pdf")
                generate_combined_report(all_detections if 'all_detections' in locals() else [], 
                                    all_violence_detections if 'all_violence_detections' in locals() else [], 
                                    self.ref_files, output_dir)        
            self.root.after(0, self.detection_complete)
            
        except Exception as e:
            err_msg=str(e)
            self.root.after(0, lambda: self.detection_error(err_msg))
    
    def prompt_violence_detection(self, missing_detections):
        """Prompt user to run violence detection after missing person results"""
        if missing_detections:
            msg = "Missing persons detected. Run violence detection on these videos?"
        else:
            msg = "Run violence detection on all videos?"
        
        response = messagebox.askyesno("Violence Detection", msg)
        if response:
            # Determine target videos
            target_videos = list(set(d['video_path'] for d in missing_detections)) if missing_detections else self.video_files
            
            # Run violence detection in a new thread
            violence_thread = threading.Thread(target=self.run_violence_detection, args=(target_videos,))
            violence_thread.daemon = True
            violence_thread.start()
        else:
            self.root.after(0, self.detection_complete)
    
    def run_violence_detection(self, target_videos):
        """Run violence detection on selected videos"""
        try:
            spark = initialize_spark()
            try:
                # Run violence detection
                _, violence_detections, _ = run_distributed_pipeline(
                    spark, target_videos, run_violence=True
                )
                
                # Generate reports
                if violence_detections:
                    output_dir = "Output"
                    from report_generation import export_violence_report, generate_combined_report
                    
                    # Violence report
                    pdf_path = os.path.join(output_dir, "violence_report.pdf")
                    export_violence_report(violence_detections, "Violence Detection", pdf_filename=pdf_path)
                    
                    # Combined report
                    combined_path = os.path.join(output_dir, "combined_report.pdf")
                    generate_combined_report([], violence_detections, [], output_dir)
                    
                    self.root.after(0, self.detection_complete)
                else:
                    self.root.after(0, self.detection_complete)
            except Exception as e:
                self.root.after(0, self.detection_error, str(e))
            finally:
                spark.stop()
        except Exception as e:
            self.root.after(0, self.detection_error, str(e))

    def detection_complete(self):
        self.progress.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Detection completed")

         # Calculate the time taken
        time_taken = time.time() - self.start_time
        time_taken_str = f"{time_taken:.2f} seconds"
    
        # Check if PDF files were generated
        pdf_files = []
        output_dir = "Output"
        for file in os.listdir(output_dir):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(output_dir, file))
    
        if pdf_files:
            # Create a dialog with buttons to open the PDFs
            result = messagebox.askquestion("Detection Complete", 
                                        f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}\nWould you like to view the results?")
            if result == "yes":
                self.show_pdf_viewer(pdf_files)
        else:
            messagebox.showinfo("Complete", f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}")

    def show_pdf_viewer(self, pdf_files):
        """Display a window with buttons to open available PDF reports"""
        pdf_window = tk.Toplevel(self.root)
        pdf_window.title("Detection Reports")
        pdf_window.geometry("400x300")
        pdf_window.configure(bg="#f5f5f5")
    
        # Header
        tk.Label(
            pdf_window, 
            text="Available Detection Reports",
            font=("Arial", 14, "bold"),
            bg="#f5f5f5",
            pady=10
        ).pack(fill=tk.X)
    
        # Create a frame for the PDF list
        list_frame = tk.Frame(pdf_window, bg="#f5f5f5", padx=20, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
    
        # Add a button for each PDF file
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
        
            # Determine file type for icon/color
            if "violence" in filename.lower():
                bg_color = "#e74c3c"  # Red for violence
                prefix = "🔍 "
            else:
                bg_color = "#3498db"  # Blue for missing person
                prefix = "👤 "
            
            button_frame = tk.Frame(list_frame, bg="#f5f5f5", pady=5)
            button_frame.pack(fill=tk.X)
        
            # Button to open the PDF
            open_button = tk.Button(
                button_frame,
                text=f"{prefix} Open {filename}",
                command=lambda f=pdf_file: self.open_pdf(f),
                bg=bg_color,
                fg="white",
                font=("Arial", 11),
                pady=8
            )
            open_button.pack(fill=tk.X)
    
        # Close button
        tk.Button(
            pdf_window,
            text="Close",
            command=pdf_window.destroy,
            bg="#7f8c8d",
            fg="white",
            font=("Arial", 11),
            pady=5
        ).pack(fill=tk.X, padx=20, pady=10)

    def open_pdf(self, pdf_path):
        """Open the PDF file with the default viewer"""
        try:
            import platform
            import subprocess
        
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', pdf_path))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(pdf_path)
            else:  # Linux
                subprocess.call(('xdg-open', pdf_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")
    
    def detection_error(self, error_msg):
        self.progress.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Error")
        messagebox.showerror("Error", f"An error occurred during detection:\n{error_msg}")

    def show_about(self):
        about_text = (
            "Missing Person & Violence Detection System\n"
            "Version 1.2.0\n\n"
            "This application provides advanced video analysis capabilities:\n"
            "• Missing person detection using reference images\n"
            "• Violence detection in video content\n"
            "• Distributed processing with Apache Spark\n\n"
            "GPU Acceleration: " + ("Available" if torch.cuda.is_available() else "Not Available") + "\n"
            "Spark Support: " + ("Available" if SPARK_AVAILABLE else "Not Available") + "\n\n"
            "© 2025 PRID Systems"
        )
        
        messagebox.showinfo("About", about_text)

    def open_url(self, url):
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open URL: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("Output", exist_ok=True)
    
    # Initialize the app
    root = tk.Tk()
    app = MissingPersonDetectionApp(root)
    
    # Center window
    window_width = 1100
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int((screen_width - window_width) / 2)
    center_y = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    
    def on_close():
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()