#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 服务器（基于 llama-server + Flask 代理）
完整端点：
- GET /v1/models
- GET /v1/models/{model}
- POST /v1/chat/completions （支持流式 SSE）
- POST /v1/completions （支持流式 SSE）
- POST /v1/embeddings
- GET /health, GET /v1/health（含资源监控 + llama-server 进程显存占用）
- POST /v1/chat/completions_with_file （文件上传）
- GET /v1/models/files （获取所有 GGUF 文件列表）
- POST /v1/models/switch （动态切换模型）
- GET / 和 GET /docs （API 文档页面，含实时资源监控和配置展示）

系统托盘控制服务启停、可视化配置、打开软件自动启动服务
增加 CUDA / Vulkan / CPU 后端选择（移除了 SYCL）
增加资源保护（内存/显存阈值、并发限制）
增强：访问日志、资源仪表板、优雅关闭、速率限制、流式断连检测、
      多GPU监控适配、llama-server 进程显存占用显示、超时记录
新增：设置窗体左右分栏，左栏设置，右栏监控+日志，日志实时显示
      设置窗体位置和大小保存到配置文件，首次居中
新增：智能参数推荐（合并硬件推荐和模型采样加载，一键完成）
新增：Web 界面实时资源监控和配置展示，并提供带转义的单行 cmd 命令示例
增强异常反馈：所有操作均提供详细的错误信息及解决建议
"""

import os
import sys
import json
import time
import threading
import logging
import configparser
import subprocess
import socket
import tempfile
import base64
import mimetypes
import hashlib
import math
import signal
import atexit
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO
from datetime import datetime

# PyQt5 托盘相关
from PyQt5.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QAction, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QSpinBox, QSlider, QLineEdit,
    QGroupBox, QScrollArea, QPushButton, QMessageBox, QDesktopWidget,
    QTextEdit, QDoubleSpinBox, QDialog, QPlainTextEdit, QListWidget,
    QListWidgetItem, QProgressBar, QFormLayout, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QPoint, QSize, QObject
from PyQt5.QtGui import QIcon, QPainter, QColor, QBrush, QPixmap, QFont, QPen, QTextCursor

# 请求库
import requests

# Flask 相关
try:
    from flask import Flask, request, jsonify, Response, g, stream_with_context
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("警告: Flask 未安装，代理功能不可用。请运行: pip install Flask flask_cors")

# 图片处理
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil 未安装，资源保护功能将受限。请运行: pip install psutil")

# 尝试导入 NVIDIA 管理库（优先使用新版 nvidia-ml-py）
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None
    logger = logging.getLogger("OpenAIServer")
    logger.warning("未安装 nvidia-ml-py，GPU 显存监控不可用。请执行: pip install nvidia-ml-py")

# ==================== 路径和配置 ====================
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_PATH, "config.ini")
LLAMA_SERVER_PATH = os.path.join(BASE_PATH, "Llama", "llama-server.exe")

# ==================== 日志配置（仅控制台） ====================
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OpenAIServer")

# ==================== 速率限制常量 ====================
RATE_LIMIT_REQUESTS = 10      # 每秒允许的请求数（每个IP）
RATE_LIMIT_BURST = 20        # 突发容量

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    "model_file": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "mmproj_file": "Qwen2.5-VL-3B-mmproj-model-f16.gguf",
    "context_size": 4096,
    "cpu_threads": 8,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.0,
    "max_tokens": 512,
    "system_prompt": "你是一个精通中文的AI助手。你的所有回复都必须使用中文,包括解释、说明和回答问题。",
    "backend_type": "auto",          # auto, cuda, vulkan, cpu
    "gpu_device": "0",               # 设备 ID，例如 "0" 或 "0,1"
    "omp_threads": 4,
    "ngl": 0,
    "batch_size": 512,
    "ubatch_size": 512,
    "flash_attn": "auto",
    "llama_server_port": 9999,
    "proxy_port": 8081,
    "server_start_timeout": 300,
    "request_timeout": 120,
    "server_ready_timeout": 2.0,
    # ----- 资源保护配置 -----
    "memory_threshold_mb": 512,
    "gpu_memory_threshold_mb": 512,
    "intel_shared_reserve_mb": 1024,
    "max_concurrent_requests": 50,
    # ----- 设置窗体位置大小 -----
    "settings_window_x": -1,
    "settings_window_y": -1,
    "settings_window_width": -1,
    "settings_window_height": -1,
}

# ==================== 配置数据类 ====================
@dataclass
class AppConfig:
    model_file: str = DEFAULT_CONFIG["model_file"]
    mmproj_file: str = DEFAULT_CONFIG["mmproj_file"]
    context_size: int = DEFAULT_CONFIG["context_size"]
    cpu_threads: int = DEFAULT_CONFIG["cpu_threads"]
    temperature: float = DEFAULT_CONFIG["temperature"]
    top_p: float = DEFAULT_CONFIG["top_p"]
    top_k: int = DEFAULT_CONFIG["top_k"]
    repeat_penalty: float = DEFAULT_CONFIG["repeat_penalty"]
    presence_penalty: float = DEFAULT_CONFIG["presence_penalty"]
    max_tokens: int = DEFAULT_CONFIG["max_tokens"]
    system_prompt: str = DEFAULT_CONFIG["system_prompt"]
    backend_type: str = DEFAULT_CONFIG["backend_type"]
    gpu_device: str = DEFAULT_CONFIG["gpu_device"]
    omp_threads: int = DEFAULT_CONFIG["omp_threads"]
    ngl: int = DEFAULT_CONFIG["ngl"]
    batch_size: int = DEFAULT_CONFIG["batch_size"]
    ubatch_size: int = DEFAULT_CONFIG["ubatch_size"]
    flash_attn: str = DEFAULT_CONFIG["flash_attn"]
    llama_server_port: int = DEFAULT_CONFIG["llama_server_port"]
    proxy_port: int = DEFAULT_CONFIG["proxy_port"]
    server_start_timeout: int = DEFAULT_CONFIG["server_start_timeout"]
    request_timeout: int = DEFAULT_CONFIG["request_timeout"]
    server_ready_timeout: float = DEFAULT_CONFIG["server_ready_timeout"]
    memory_threshold_mb: int = DEFAULT_CONFIG["memory_threshold_mb"]
    gpu_memory_threshold_mb: int = DEFAULT_CONFIG["gpu_memory_threshold_mb"]
    intel_shared_reserve_mb: int = DEFAULT_CONFIG["intel_shared_reserve_mb"]
    max_concurrent_requests: int = DEFAULT_CONFIG["max_concurrent_requests"]
    settings_window_x: int = DEFAULT_CONFIG["settings_window_x"]
    settings_window_y: int = DEFAULT_CONFIG["settings_window_y"]
    settings_window_width: int = DEFAULT_CONFIG["settings_window_width"]
    settings_window_height: int = DEFAULT_CONFIG["settings_window_height"]

    @classmethod
    def load(cls) -> "AppConfig":
        if not os.path.exists(CONFIG_FILE):
            cfg = cls()
            cfg.save()
            return cfg
        config_dict = DEFAULT_CONFIG.copy()
        try:
            cfg_parser = configparser.ConfigParser()
            cfg_parser.read(CONFIG_FILE, encoding='utf-8')
            sections = {
                'Model': ['model_file', 'mmproj_file', 'context_size', 'cpu_threads'],
                'AI': ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'presence_penalty', 'max_tokens', 'system_prompt'],
                'Hardware': ['backend_type', 'gpu_device', 'omp_threads', 'ngl', 'batch_size', 'ubatch_size', 'flash_attn', 'llama_server_port', 'server_start_timeout'],
                'Advanced': ['request_timeout', 'server_ready_timeout'],
                'Proxy': ['proxy_port'],
                'Resource': ['memory_threshold_mb', 'gpu_memory_threshold_mb', 'intel_shared_reserve_mb', 'max_concurrent_requests'],
                'GUI': ['settings_window_x', 'settings_window_y', 'settings_window_width', 'settings_window_height'],
            }
            for section, keys in sections.items():
                if section not in cfg_parser:
                    continue
                for key in keys:
                    if key not in config_dict:
                        continue
                    str_val = cfg_parser.get(section, key, fallback=str(config_dict[key]))
                    orig_type = type(config_dict[key])
                    if orig_type == int:
                        config_dict[key] = int(str_val)
                    elif orig_type == float:
                        config_dict[key] = float(str_val)
                    else:
                        config_dict[key] = str_val
        except Exception as e:
            # 配置文件损坏，向用户反馈
            logger.error(f"加载配置失败: {e}")
            QMessageBox.warning(
                None, 
                "配置加载警告",
                f"配置文件 `config.ini` 解析失败，将使用默认配置。\n错误详情：{str(e)}\n\n"
                "建议：检查配置文件格式，或删除后重新生成（程序将自动创建默认配置）。"
            )
        return cls(**config_dict)

    def save(self) -> Tuple[bool, str]:
        """保存配置，返回 (成功, 错误信息)"""
        try:
            cfg = configparser.ConfigParser()
            cfg['Model'] = {k: str(getattr(self, k)) for k in ['model_file', 'mmproj_file', 'context_size', 'cpu_threads']}
            cfg['AI'] = {k: str(getattr(self, k)) for k in ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'presence_penalty', 'max_tokens', 'system_prompt']}
            cfg['Hardware'] = {k: str(getattr(self, k)) for k in ['backend_type', 'gpu_device', 'omp_threads', 'ngl', 'batch_size', 'ubatch_size', 'flash_attn', 'llama_server_port', 'server_start_timeout']}
            cfg['Advanced'] = {k: str(getattr(self, k)) for k in ['request_timeout', 'server_ready_timeout']}
            cfg['Proxy'] = {'proxy_port': str(self.proxy_port)}
            cfg['Resource'] = {k: str(getattr(self, k)) for k in ['memory_threshold_mb', 'gpu_memory_threshold_mb', 'intel_shared_reserve_mb', 'max_concurrent_requests']}
            cfg['GUI'] = {k: str(getattr(self, k)) for k in ['settings_window_x', 'settings_window_y', 'settings_window_width', 'settings_window_height']}
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                cfg.write(f)
            return True, ""
        except PermissionError:
            return False, "没有写入权限，请以管理员身份运行或检查文件属性。"
        except OSError as e:
            return False, f"磁盘写入失败：{str(e)}，请检查磁盘空间或文件是否被占用。"
        except Exception as e:
            return False, f"保存配置异常：{str(e)}"

# ==================== LlamaServer 管理器 ====================
class LlamaServerManager:
    _instance = None
    _process = None
    _port = None
    pid = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def start(self, config: AppConfig) -> Tuple[bool, str, str]:
        """
        启动 llama-server
        返回: (成功, 错误信息, 解决建议)
        """
        # 先停止已有实例
        self.stop()

        port = config.llama_server_port
        # 检查端口是否被占用
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                # 端口被占用，尝试检测是否为已有llama-server
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/", timeout=2)
                    if r.status_code == 200:
                        content = r.text.lower()
                        if 'llama' in content or 'openai' in content or 'api' in content:
                            logger.info(f"端口 {port} 已有llama-server服务运行，复用")
                            self._port = port
                            return True, "", ""
                        else:
                            return False, f"端口 {port} 被其他服务占用（非llama-server），无法启动。", "请更改后端端口或关闭占用该端口的程序。"
                except Exception:
                    return False, f"端口 {port} 被占用且无法确认是否为llama-server，请检查。", "请更换端口或结束占用进程。"
        finally:
            sock.close()

        # 资源预检
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil 未安装，跳过资源预检")
        else:
            try:
                mem = psutil.virtual_memory()
                available_mb = mem.available // (1024 * 1024)
                if available_mb < config.memory_threshold_mb:
                    return False, f"系统可用内存不足（{available_mb}MB < {config.memory_threshold_mb}MB），无法启动。", "关闭其他程序释放内存，或降低内存阈值（设置→资源保护）。"

                if config.backend_type == "vulkan":
                    required = config.memory_threshold_mb + config.intel_shared_reserve_mb
                    if available_mb < required:
                        return False, f"Vulkan 后端需为核显预留额外内存，当前可用 {available_mb}MB，需 {required}MB。", "增加预留值或改用其他后端（CUDA/CPU）。"

                if config.backend_type == "cuda" and NVML_AVAILABLE:
                    try:
                        pynvml.nvmlInit()
                        for i in range(pynvml.nvmlDeviceGetCount()):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            free_mb = info.free // (1024 * 1024)
                            if free_mb < config.gpu_memory_threshold_mb:
                                return False, f"GPU {i} 可用显存不足（{free_mb}MB < {config.gpu_memory_threshold_mb}MB）。", "减少其他GPU任务，或降低显存阈值（设置→资源保护）。"
                    except Exception as e:
                        logger.warning(f"无法检查 NVIDIA 显存: {e}，继续启动")
                    finally:
                        try:
                            pynvml.nvmlShutdown()
                        except:
                            pass
            except Exception as e:
                logger.warning(f"启动前资源预检异常: {e}，继续启动")

        # 检查可执行文件
        if not os.path.exists(LLAMA_SERVER_PATH):
            return False, f"llama-server.exe 未找到: {LLAMA_SERVER_PATH}", "请将 llama-server.exe 放入 Llama 目录。"

        model_path = os.path.join(BASE_PATH, "Model", config.model_file)
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}", "请将模型文件放入 Model 目录，或在设置中选择正确的模型。"

        mmproj_path = None
        if config.mmproj_file and config.mmproj_file.strip():
            mmproj_path = os.path.join(BASE_PATH, "Model", config.mmproj_file.strip())
            if not os.path.exists(mmproj_path):
                return False, f"视觉模型文件不存在: {mmproj_path}", "请检查 mmproj 文件是否存在，或清除该配置。"

        # 准备环境变量
        env = os.environ.copy()
        backend = config.backend_type.lower()
        if backend == "cuda":
            if config.gpu_device:
                env["CUDA_VISIBLE_DEVICES"] = config.gpu_device
            env["GGML_CUDA_ENABLE"] = "1"
        elif backend == "vulkan":
            if config.gpu_device:
                env["GGML_VK_VISIBLE_DEVICES"] = config.gpu_device
        elif backend == "cpu":
            # 清除所有 GPU 相关环境变量
            for key in ["CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "GGML_CUDA_ENABLE", "GGML_SYCL", "SYCL_DEVICE_FILTER", "ONEAPI_DEVICE_SELECTOR"]:
                env.pop(key, None)
        # auto 模式不做特殊处理

        if config.omp_threads:
            env["OMP_NUM_THREADS"] = str(config.omp_threads)

        cmd = [
            LLAMA_SERVER_PATH,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(config.context_size),
            "-ngl", str(config.ngl),
            "-t", str(config.cpu_threads),
            "-b", str(config.batch_size),
            "-ub", str(config.ubatch_size),
            "--top-k", str(config.top_k),
            "--repeat-penalty", str(config.repeat_penalty),
            "--presence-penalty", str(config.presence_penalty),
            "--mlock", "--no-mmap",
            "--api-key", "",
            "--cont-batching",
        ]
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])
        if config.flash_attn in ["auto", "on", "off"]:
            cmd.extend(["--flash-attn", config.flash_attn])

        logger.info(f"启动 llama-server，完整命令: {' '.join(cmd)}")

        startupinfo = None
        creationflags = 0
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
                env=env
            )
            self.pid = self._process.pid
            self._port = port
        except FileNotFoundError:
            return False, f"无法执行 llama-server.exe，可能缺少运行库或路径错误。", "请安装 Visual C++ Redistributable 或以管理员身份运行。"
        except PermissionError:
            return False, f"没有权限执行 llama-server.exe", "请以管理员身份运行程序。"
        except Exception as e:
            return False, f"启动进程失败：{str(e)}", "检查系统环境或尝试重启。"

        # 等待服务就绪
        start_time = time.time()
        while time.time() - start_time < config.server_start_timeout:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/", timeout=config.server_ready_timeout)
                if r.status_code == 200:
                    logger.info(f"llama-server 启动成功，端口 {port}")
                    # 预热
                    self._warmup_model(config)
                    return True, "", ""
            except:
                pass
            time.sleep(1)
        # 超时
        return False, f"服务器启动超时（{config.server_start_timeout}秒），可能模型过大或配置不当。", "增加启动超时时间（设置→超时设置），或检查模型是否损坏。"

    def _warmup_model(self, config: AppConfig):
        try:
            warmup_start = time.time()
            warmup_url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
            warmup_data = {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False
            }
            r = requests.post(warmup_url, json=warmup_data, timeout=30)
            if r.status_code == 200:
                warmup_time = time.time() - warmup_start
                logger.info(f"模型预热完成，耗时 {warmup_time:.2f} 秒")
            else:
                logger.warning(f"模型预热失败: {r.status_code}")
        except Exception as e:
            logger.warning(f"模型预热异常: {e}")

    def stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._port = None
            self.pid = None
        logger.info("llama-server 已停止")

    def is_running(self) -> bool:
        if self._port is None:
            return False
        try:
            r = requests.get(f"http://127.0.0.1:{self._port}/", timeout=1)
            return r.status_code == 200
        except:
            return False

    def get_port(self) -> int:
        return self._port if self._port else 8080

# ==================== 嵌入向量生成器 ====================
class SimpleEmbedder:
    @staticmethod
    def get_embedding(text: str, dimension: int = 768) -> List[float]:
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        digest = hash_obj.digest()
        seed = int.from_bytes(digest[:8], 'little')
        vec = []
        rng_state = seed
        for _ in range(dimension):
            rng_state = (rng_state * 1103515245 + 12345) & 0xFFFFFFFF
            rand = rng_state / 4294967296.0
            vec.append(rand * 2.0 - 1.0)
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

# ==================== 日志处理器（Qt 信号） ====================
class LogHandler(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

# ==================== Flask 代理服务器（完整增强版） ====================
class ProxyServer:
    def __init__(self, config: AppConfig, switch_signal=None):
        self.config = config
        self.switch_signal = switch_signal
        self.app = Flask(__name__)
        CORS(self.app)
        self.thread = None
        self.running = False
        self.embedder = SimpleEmbedder()
        self._server = None

        self.memory_threshold = config.memory_threshold_mb * 1024 * 1024
        self.gpu_threshold = config.gpu_memory_threshold_mb * 1024 * 1024
        self.intel_reserve = config.intel_shared_reserve_mb * 1024 * 1024
        self.max_concurrent = config.max_concurrent_requests
        self.semaphore = threading.BoundedSemaphore(self.max_concurrent)

        self.active_requests = 0
        self.active_lock = threading.Lock()
        self.all_done = threading.Condition(threading.Lock())

        self.rate_limit_lock = threading.Lock()
        self.rate_limit_buckets = defaultdict(lambda: {
            "tokens": RATE_LIMIT_BURST,
            "last_time": time.time()
        })

        self.nvml_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.nvml_available = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"NVML 初始化成功，检测到 {self.device_count} 个 NVIDIA GPU")
            except Exception as e:
                logger.error(f"NVML 初始化失败: {e}")
                self.nvml_available = False

        self.llama_pid = None

        self._register_routes()

    def set_llama_pid(self, pid):
        self.llama_pid = pid

    # ---------- 资源检查与获取/释放 ----------
    def _check_resources(self) -> Tuple[bool, str, str]:
        """检查资源是否充足，返回 (充足?, 错误信息, 建议)"""
        if not PSUTIL_AVAILABLE:
            return True, "", ""
        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available // (1024 * 1024)
            if available_mb < self.memory_threshold // (1024 * 1024):
                return False, f"系统可用内存不足（{available_mb}MB < {self.memory_threshold//(1024*1024)}MB）", "关闭其他程序或降低内存阈值。"
            if self.config.backend_type == "vulkan":
                required = self.memory_threshold + self.intel_reserve
                if mem.available < required:
                    return False, f"Vulkan 后端需额外预留内存，当前 {available_mb}MB，需 {required//(1024*1024)}MB", "增加预留值或切换后端。"
            if self.config.backend_type == "cuda" and self.nvml_available:
                for i in range(self.device_count):
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                    info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                    free_mb = info.free // (1024 * 1024)
                    if free_mb < self.gpu_threshold // (1024 * 1024):
                        return False, f"GPU {i} 显存不足（{free_mb}MB < {self.gpu_threshold//(1024*1024)}MB）", "减少其他GPU任务或降低显存阈值。"
            return True, "", ""
        except Exception as e:
            logger.error(f"资源检查异常: {e}")
            return True, "", ""  # 检查异常时不阻止

    def _acquire_resource(self) -> Tuple[bool, str, str]:
        if not self.semaphore.acquire(blocking=False):
            return False, f"并发请求数已达上限（{self.max_concurrent}），请稍后重试。", "降低请求频率或增加最大并发数（设置→资源保护）。"
        with self.active_lock:
            self.active_requests += 1
        ok, err, sug = self._check_resources()
        if not ok:
            self.semaphore.release()
            with self.active_lock:
                self.active_requests -= 1
            return False, err, sug
        return True, "", ""

    def _release_resource(self):
        with self.active_lock:
            self.active_requests -= 1
            if self.active_requests == 0:
                with self.all_done:
                    self.all_done.notify_all()
        self.semaphore.release()

    # ---------- 速率限制 ----------
    def _check_rate_limit(self, ip: str) -> Tuple[bool, str, str]:
        now = time.time()
        with self.rate_limit_lock:
            bucket = self.rate_limit_buckets[ip]
            elapsed = now - bucket["last_time"]
            new_tokens = elapsed * RATE_LIMIT_REQUESTS
            bucket["tokens"] = min(bucket["tokens"] + new_tokens, RATE_LIMIT_BURST)
            bucket["last_time"] = now
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, "", ""
            return False, "请求过于频繁，请稍后重试。", "降低请求频率。"

    # ---------- 请求日志（含超时判断） ----------
    def _log_request(self):
        duration = time.time() - g.start_time
        ip = request.remote_addr
        endpoint = request.path
        status = g.get('response_status', 0)
        timeout = self.config.request_timeout
        if duration > timeout:
            logger.info(f"API_LOG {ip} {request.method} {endpoint} {status} {duration:.4f}s [TIMEOUT]")
        else:
            logger.info(f"API_LOG {ip} {request.method} {endpoint} {status} {duration:.4f}s")

    # ---------- 资源监控数据（增强版：增加进程 RAM 和 GPU 总显存） ----------
    def _get_resource_info(self):
        backend = self.config.backend_type.lower()
        info = {
            "memory_available_mb": 0,
            "memory_total_mb": 0,
            "memory_percent": 0,
            "gpu_info": [],
            "gpu_message": "",
            "active_requests": 0,
            "max_concurrent": self.max_concurrent,
            "backend": backend,
            "gpu_device": self.config.gpu_device,
            "llama_process_gpu_mb": 0,
            "llama_process_ram_mb": 0,
            "gpu_total_used_mb": 0,
        }
        with self.active_lock:
            info["active_requests"] = self.active_requests

        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            info["memory_available_mb"] = mem.available // (1024 * 1024)
            info["memory_total_mb"] = mem.total // (1024 * 1024)
            info["memory_percent"] = mem.percent

        # 采集 llama-server 进程的系统内存
        if self.llama_pid is not None and PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(self.llama_pid)
                mem_info = proc.memory_info()
                info["llama_process_ram_mb"] = mem_info.rss // (1024 * 1024)
            except Exception as e:
                logger.debug(f"无法获取 llama-server 进程内存信息: {e}")

        # 采集 NVIDIA GPU 信息
        if self.nvml_available:
            for i in range(self.device_count):
                try:
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb = mem_info.total // (1024 * 1024)
                    used_mb = mem_info.used // (1024 * 1024)
                    free_mb = mem_info.free // (1024 * 1024)
                    gpu_data = {
                        "device_id": i,
                        "total_mb": total_mb,
                        "used_mb": used_mb,
                        "free_mb": free_mb,
                    }
                    info["gpu_info"].append(gpu_data)
                    info["gpu_total_used_mb"] += used_mb

                    if self.llama_pid is not None:
                        try:
                            compute_procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                            graphics_procs = self.nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                            all_procs = list(compute_procs) + list(graphics_procs)
                            found = False
                            for proc in all_procs:
                                if proc.pid == self.llama_pid:
                                    used = getattr(proc, 'usedGpuMemory', 0) or 0
                                    info["llama_process_gpu_mb"] += used // (1024 * 1024)
                                    found = True
                                    break
                            if not found:
                                logger.debug(f"GPU {i} 未找到 PID {self.llama_pid} 的进程")
                        except Exception as e:
                            logger.warning(f"获取 GPU {i} 进程信息失败: {e}")
                except Exception as e:
                    logger.warning(f"获取 GPU {i} 基本信息失败: {e}")
        else:
            if backend == "cuda":
                info["gpu_message"] = "CUDA 后端已启用，但 NVML 初始化失败（请检查 NVIDIA 驱动是否安装）。"
            elif backend == "vulkan":
                info["gpu_message"] = "使用 Vulkan 后端（支持集成显卡）。"
            elif backend == "cpu":
                info["gpu_message"] = "CPU 模式无显存。"
            else:
                info["gpu_message"] = "未检测到 NVIDIA GPU 或驱动未安装。"

        return info

    def _get_config_info(self):
        config = self.config
        return {
            "model_file": config.model_file,
            "mmproj_file": config.mmproj_file,
            "context_size": config.context_size,
            "cpu_threads": config.cpu_threads,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "presence_penalty": config.presence_penalty,
            "max_tokens": config.max_tokens,
            "system_prompt": config.system_prompt,
            "backend_type": config.backend_type,
            "gpu_device": config.gpu_device,
            "omp_threads": config.omp_threads,
            "ngl": config.ngl,
            "batch_size": config.batch_size,
            "ubatch_size": config.ubatch_size,
            "flash_attn": config.flash_attn,
            "llama_server_port": config.llama_server_port,
            "proxy_port": config.proxy_port,
            "server_start_timeout": config.server_start_timeout,
            "request_timeout": config.request_timeout,
            "memory_threshold_mb": config.memory_threshold_mb,
            "gpu_memory_threshold_mb": config.gpu_memory_threshold_mb,
            "intel_shared_reserve_mb": config.intel_shared_reserve_mb,
            "max_concurrent_requests": config.max_concurrent_requests,
        }

    # ---------- 错误响应辅助 ----------
    def _make_error_response(self, message, error_type, detail=None, suggestion=None, status=400):
        resp = {
            "error": {
                "message": message,
                "type": error_type,
            }
        }
        if detail:
            resp["error"]["detail"] = detail
        if suggestion:
            resp["error"]["suggestion"] = suggestion
        return jsonify(resp), status

    # ---------- 路由注册 ----------
    def _register_routes(self):
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            ip = request.remote_addr
            logger.info(f"API_REQ START {ip} {request.method} {request.path}")
            if request.path not in ('/health', '/v1/health', '/', '/docs', '/api/status'):
                ok, msg, sug = self._check_rate_limit(ip)
                if not ok:
                    return self._make_error_response(msg, "rate_limit_exceeded", suggestion=sug, status=429)

        @self.app.after_request
        def after_request(response):
            g.response_status = response.status_code
            self._log_request()
            return response

        @self.app.route('/health', methods=['GET'])
        @self.app.route('/v1/health', methods=['GET'])
        def health():
            backend_ok = self._is_backend_ready()
            resource_info = self._get_resource_info()
            return jsonify({
                "status": "healthy" if backend_ok else "degraded",
                "backend": "running" if backend_ok else "not ready",
                "proxy": "running",
                "version": "1.0.0",
                "resources": resource_info
            }), 200 if backend_ok else 503

        @self.app.route('/api/status', methods=['GET'])
        def api_status():
            config_info = self._get_config_info()
            resource_info = self._get_resource_info()
            return jsonify({
                "config": config_info,
                "resources": resource_info,
                "timestamp": time.time()
            })

        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            model_path = os.path.join(BASE_PATH, "Model", self.config.model_file)
            created = int(os.path.getctime(model_path)) if os.path.exists(model_path) else int(time.time())
            return jsonify({
                "object": "list",
                "data": [{
                    "id": self.config.model_file,
                    "object": "model",
                    "created": created,
                    "owned_by": "local"
                }]
            })

        @self.app.route('/v1/models/<model_id>', methods=['GET'])
        def get_model(model_id):
            if model_id != self.config.model_file:
                return self._make_error_response(f"Model '{model_id}' not found", "invalid_request_error", status=404)
            model_path = os.path.join(BASE_PATH, "Model", self.config.model_file)
            created = int(os.path.getctime(model_path)) if os.path.exists(model_path) else int(time.time())
            return jsonify({
                "id": self.config.model_file,
                "object": "model",
                "created": created,
                "owned_by": "local",
                "permission": []
            })

        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            ok, err, sug = self._acquire_resource()
            if not ok:
                return self._make_error_response(err, "resource_exhausted", suggestion=sug, status=503)
            try:
                if not self._is_backend_ready():
                    self._release_resource()
                    return self._make_error_response("llama-server 未准备好，请确保服务已启动。", "server_error", status=503)
                backend_url = f"http://127.0.0.1:{self.config.llama_server_port}/v1/chat/completions"
                req_data = request.json
                if req_data is None:
                    self._release_resource()
                    return self._make_error_response("请求体必须是有效的 JSON", "invalid_request_error", status=400)
                stream = req_data.get('stream', False)
                messages = req_data.get('messages')
                if not messages:
                    self._release_resource()
                    return self._make_error_response("缺少 'messages' 字段", "invalid_request_error", status=400)
                if messages and messages[0]['role'] != 'system' and self.config.system_prompt:
                    req_data['messages'] = [{"role": "system", "content": self.config.system_prompt}] + messages
                resp = requests.post(backend_url, json=req_data, stream=stream, timeout=self.config.request_timeout)
                if stream:
                    @stream_with_context
                    def generate():
                        try:
                            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=False):
                                if chunk:
                                    yield chunk
                        except GeneratorExit:
                            resp.close()
                            logger.info("客户端断开流式连接")
                        except Exception as e:
                            logger.error(f"流式转发异常: {e}")
                            error_msg = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                            yield f"data: {error_msg}\n\n".encode('utf-8')
                        finally:
                            resp.close()
                            self._release_resource()
                    headers = {
                        'Content-Type': resp.headers.get('Content-Type', 'text/event-stream'),
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                    }
                    return Response(generate(), headers=headers, status=resp.status_code)
                else:
                    content = resp.content
                    resp.close()
                    self._release_resource()
                    return Response(content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
            except requests.exceptions.Timeout:
                self._release_resource()
                return self._make_error_response(f"请求超时（{self.config.request_timeout}秒），后端可能繁忙。", "server_timeout", suggestion="增加请求超时时间或减少 max_tokens。", status=504)
            except requests.exceptions.ConnectionError:
                self._release_resource()
                return self._make_error_response("无法连接到 llama-server，请检查服务是否运行。", "server_connection_error", status=503)
            except Exception as e:
                logger.error(f"转发失败: {e}")
                self._release_resource()
                return self._make_error_response(f"内部错误：{str(e)}", "api_error", status=500)

        @self.app.route('/v1/completions', methods=['POST'])
        def completions():
            ok, err, sug = self._acquire_resource()
            if not ok:
                return self._make_error_response(err, "resource_exhausted", suggestion=sug, status=503)
            try:
                if not self._is_backend_ready():
                    self._release_resource()
                    return self._make_error_response("llama-server 未准备好。", "server_error", status=503)
                backend_url = f"http://127.0.0.1:{self.config.llama_server_port}/v1/completions"
                req_data = request.json
                if req_data is None:
                    self._release_resource()
                    return self._make_error_response("请求体必须是有效的 JSON", "invalid_request_error", status=400)
                stream = req_data.get('stream', False)
                resp = requests.post(backend_url, json=req_data, stream=stream, timeout=self.config.request_timeout)
                if stream:
                    @stream_with_context
                    def generate():
                        try:
                            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=False):
                                if chunk:
                                    yield chunk
                        except GeneratorExit:
                            resp.close()
                            logger.info("客户端断开流式连接")
                        except Exception as e:
                            logger.error(f"流式转发异常: {e}")
                            error_msg = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                            yield f"data: {error_msg}\n\n".encode('utf-8')
                        finally:
                            resp.close()
                            self._release_resource()
                    headers = {
                        'Content-Type': resp.headers.get('Content-Type', 'text/event-stream'),
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                    }
                    return Response(generate(), headers=headers, status=resp.status_code)
                else:
                    content = resp.content
                    resp.close()
                    self._release_resource()
                    return Response(content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
            except requests.exceptions.Timeout:
                self._release_resource()
                return self._make_error_response(f"请求超时（{self.config.request_timeout}秒）", "server_timeout", status=504)
            except requests.exceptions.ConnectionError:
                self._release_resource()
                return self._make_error_response("无法连接到 llama-server", "server_connection_error", status=503)
            except Exception as e:
                logger.error(f"转发失败: {e}")
                self._release_resource()
                return self._make_error_response(f"内部错误：{str(e)}", "api_error", status=500)

        @self.app.route('/v1/embeddings', methods=['POST'])
        def embeddings():
            ok, err, sug = self._acquire_resource()
            if not ok:
                return self._make_error_response(err, "resource_exhausted", suggestion=sug, status=503)
            try:
                req_data = request.json
                if req_data is None:
                    self._release_resource()
                    return self._make_error_response("请求体必须是有效的 JSON", "invalid_request_error", status=400)
                model = req_data.get('model', self.config.model_file)
                input_text = req_data.get('input')
                if not input_text:
                    self._release_resource()
                    return self._make_error_response("缺少 'input' 字段", "invalid_request_error", status=400)
                if isinstance(input_text, str):
                    texts = [input_text]
                else:
                    texts = input_text
                embeddings_list = []
                for idx, text in enumerate(texts):
                    vec = self.embedder.get_embedding(text, dimension=768)
                    embeddings_list.append({
                        "object": "embedding",
                        "embedding": vec,
                        "index": idx
                    })
                return jsonify({
                    "object": "list",
                    "data": embeddings_list,
                    "model": model,
                    "usage": {
                        "prompt_tokens": sum(len(t) for t in texts),
                        "total_tokens": sum(len(t) for t in texts)
                    }
                })
            except Exception as e:
                logger.error(f"嵌入生成失败: {e}")
                return self._make_error_response(f"生成嵌入失败：{str(e)}", "api_error", status=500)
            finally:
                self._release_resource()

        @self.app.route('/v1/chat/completions_with_file', methods=['POST'])
        def chat_with_file():
            ok, err, sug = self._acquire_resource()
            if not ok:
                return self._make_error_response(err, "resource_exhausted", suggestion=sug, status=503)
            try:
                if not self._is_backend_ready():
                    self._release_resource()
                    return self._make_error_response("llama-server 未准备好。", "server_error", status=503)
                uploaded_file = request.files.get('file')
                text = request.form.get('text', '')
                stream = request.form.get('stream', 'false').lower() == 'true'
                if not uploaded_file:
                    self._release_resource()
                    return self._make_error_response("缺少文件，请使用 'file' 字段上传文件。", "invalid_request_error", status=400)
                file_bytes = uploaded_file.read()
                content_type = uploaded_file.content_type or mimetypes.guess_type(uploaded_file.filename)[0] or 'application/octet-stream'
                is_image = content_type.startswith('image/')
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                user_content = []
                if is_image:
                    if not PIL_AVAILABLE:
                        self._release_resource()
                        return self._make_error_response("图片处理需要 Pillow 库，但未安装。", "dependency_missing", suggestion="pip install Pillow", status=500)
                    try:
                        img = Image.open(BytesIO(file_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((640, 640))
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG", quality=75)
                        b64 = base64.b64encode(buffered.getvalue()).decode()
                        data_url = f"data:image/jpeg;base64,{b64}"
                        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
                    except Exception as e:
                        self._release_resource()
                        return self._make_error_response(f"图片处理失败：{str(e)}。请确保上传有效的图片文件。", "invalid_request_error", status=400)
                else:
                    try:
                        file_text = file_bytes.decode('utf-8')
                        user_content.append({"type": "text", "text": f"文件内容:\n{file_text}"})
                    except UnicodeDecodeError:
                        self._release_resource()
                        return self._make_error_response("文件不是有效的 UTF-8 文本文件，请上传文本文件或图片。", "invalid_request_error", status=400)
                if text:
                    user_content.append({"type": "text", "text": text})
                if len(user_content) == 1 and user_content[0]["type"] == "text":
                    messages.append({"role": "user", "content": user_content[0]["text"]})
                else:
                    messages.append({"role": "user", "content": user_content})
                payload = {
                    "messages": messages,
                    "stream": stream,
                    "temperature": float(request.form.get('temperature', self.config.temperature)),
                    "top_p": float(request.form.get('top_p', self.config.top_p)),
                    "top_k": int(request.form.get('top_k', self.config.top_k)),
                    "repeat_penalty": float(request.form.get('repeat_penalty', self.config.repeat_penalty)),
                    "presence_penalty": float(request.form.get('presence_penalty', self.config.presence_penalty)),
                    "max_tokens": int(request.form.get('max_tokens', self.config.max_tokens)),
                }
                backend_url = f"http://127.0.0.1:{self.config.llama_server_port}/v1/chat/completions"
                resp = requests.post(backend_url, json=payload, stream=stream, timeout=self.config.request_timeout)
                if stream:
                    @stream_with_context
                    def generate():
                        try:
                            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=False):
                                if chunk:
                                    yield chunk
                        except GeneratorExit:
                            resp.close()
                            logger.info("客户端断开流式连接")
                        except Exception as e:
                            logger.error(f"流式转发异常: {e}")
                            error_msg = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                            yield f"data: {error_msg}\n\n".encode('utf-8')
                        finally:
                            resp.close()
                            self._release_resource()
                    headers = {
                        'Content-Type': resp.headers.get('Content-Type', 'text/event-stream'),
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                    }
                    return Response(generate(), headers=headers, status=resp.status_code)
                else:
                    content = resp.content
                    resp.close()
                    self._release_resource()
                    return Response(content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
            except requests.exceptions.Timeout:
                self._release_resource()
                return self._make_error_response(f"请求超时（{self.config.request_timeout}秒）", "server_timeout", status=504)
            except requests.exceptions.ConnectionError:
                self._release_resource()
                return self._make_error_response("无法连接到 llama-server", "server_connection_error", status=503)
            except Exception as e:
                logger.error(f"转发失败: {e}")
                self._release_resource()
                return self._make_error_response(f"内部错误：{str(e)}", "api_error", status=500)

        @self.app.route('/v1/models/files', methods=['GET'])
        def list_model_files():
            model_dir = os.path.join(BASE_PATH, "Model")
            if not os.path.exists(model_dir):
                return self._make_error_response("Model 目录不存在", "server_error", status=500)
            models = []
            mmproj_files = []
            try:
                for f in os.listdir(model_dir):
                    if f.endswith('.gguf'):
                        if 'mmproj' in f.lower():
                            mmproj_files.append(f)
                        else:
                            models.append(f)
            except Exception as e:
                return self._make_error_response(f"读取目录失败：{str(e)}", "server_error", status=500)
            return jsonify({
                "models": models,
                "mmproj_files": mmproj_files
            })

        @self.app.route('/v1/models/switch', methods=['POST'])
        def switch_model():
            data = request.get_json()
            if not data or 'model' not in data:
                return self._make_error_response("缺少 'model' 字段", "invalid_request_error", status=400)
            new_model = data['model'].strip()
            new_mmproj = data.get('mmproj', '').strip()
            save_config = data.get('save_config', False)
            model_path = os.path.join(BASE_PATH, "Model", new_model)
            if not os.path.exists(model_path):
                return self._make_error_response(f"模型文件不存在: {new_model}", "invalid_request_error", status=404)
            if new_mmproj:
                mmproj_path = os.path.join(BASE_PATH, "Model", new_mmproj)
                if not os.path.exists(mmproj_path):
                    return self._make_error_response(f"视觉模型文件不存在: {new_mmproj}", "invalid_request_error", status=404)
            if self.switch_signal:
                self.switch_signal.emit(new_model, new_mmproj, save_config)
                return jsonify({"status": "switching", "message": f"正在切换至 {new_model}"}), 200
            else:
                return self._make_error_response("切换回调未注册，内部错误。", "server_error", status=500)

        @self.app.route('/')
        @self.app.route('/docs')
        def api_docs():
            return self._render_docs()

    def _render_docs(self):
        proxy_port = self.config.proxy_port
        llama_port = self.config.llama_server_port
        model = self.config.model_file
        temp = self.config.temperature
        max_tokens = self.config.max_tokens
        return f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OpenAI 兼容 API 服务器 - 文档与监控</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; background: #1e1e2f; color: #e4e4e7; padding: 1rem; line-height: 1.6; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                h1 {{ font-size: 2.2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #a5f3fc, #c084fc); -webkit-background-clip: text; background-clip: text; color: transparent; }}
                .subtitle {{ color: #a1a1aa; margin-bottom: 1.5rem; border-left: 3px solid #3b82f6; padding-left: 1rem; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
                .card {{ background: #2a2a3a; border-radius: 12px; padding: 1rem 1.5rem; border: 1px solid #3f3f4e; }}
                .card h2 {{ font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.75rem; border-bottom: 1px solid #3f3f4e; padding-bottom: 0.5rem; }}
                .card .item {{ display: flex; justify-content: space-between; padding: 0.2rem 0; font-size: 0.9rem; }}
                .card .item .label {{ color: #94a3b8; }}
                .card .item .value {{ font-family: monospace; color: #fbbf24; }}
                .resource-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
                .resource-item {{ background: #1e1e2f; padding: 0.4rem 0.8rem; border-radius: 6px; display: flex; justify-content: space-between; }}
                .resource-item .label {{ color: #94a3b8; font-size: 0.85rem; }}
                .resource-item .value {{ font-weight: bold; }}
                .status-ok {{ color: #4ade80; }}
                .status-warn {{ color: #fbbf24; }}
                .status-err {{ color: #ef4444; }}
                .cmd-box {{ background: #0f172a; padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0; font-family: monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap; word-break: break-all; border-left: 3px solid #fbbf24; }}
                .cmd-box .comment {{ color: #6b7280; }}
                .footer {{ text-align: center; margin-top: 2rem; color: #71717a; font-size: 0.85rem; }}
                .refresh-info {{ font-size: 0.8rem; color: #6b7280; text-align: right; margin-top: 0.5rem; }}
                @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
                a {{ color: #3b82f6; text-decoration: none; }}
                .tag {{ display: inline-block; background: #3b82f6; color: white; font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 12px; margin-left: 0.5rem; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✨ OpenAI 兼容 API 服务器</h1>
                <div class="subtitle">基于 llama-server + Flask 代理，完全兼容 OpenAI API 规范</div>

                <div class="card" style="margin-bottom:1.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span><strong>服务状态</strong></span>
                        <span id="serverStatus" class="status-ok">✅ 运行中</span>
                    </div>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-top:0.3rem;">
                        代理端口: <span id="proxyPortDisplay">{proxy_port}</span> &nbsp;|&nbsp; 后端端口: <span id="llamaPortDisplay">{llama_port}</span>
                    </div>
                </div>

                <div class="grid">
                    <div class="card">
                        <h2>⚙️ 当前配置</h2>
                        <div id="configDisplay">
                            <div class="item"><span class="label">模型文件</span><span class="value" id="cfg_model">加载中...</span></div>
                            <div class="item"><span class="label">视觉模型</span><span class="value" id="cfg_mmproj">加载中...</span></div>
                            <div class="item"><span class="label">上下文长度</span><span class="value" id="cfg_context">加载中...</span></div>
                            <div class="item"><span class="label">CPU线程数</span><span class="value" id="cfg_threads">加载中...</span></div>
                            <div class="item"><span class="label">温度</span><span class="value" id="cfg_temp">加载中...</span></div>
                            <div class="item"><span class="label">Top P</span><span class="value" id="cfg_top_p">加载中...</span></div>
                            <div class="item"><span class="label">Top K</span><span class="value" id="cfg_top_k">加载中...</span></div>
                            <div class="item"><span class="label">重复惩罚</span><span class="value" id="cfg_rep">加载中...</span></div>
                            <div class="item"><span class="label">存在惩罚</span><span class="value" id="cfg_pres">加载中...</span></div>
                            <div class="item"><span class="label">最大Token</span><span class="value" id="cfg_max_tokens">加载中...</span></div>
                            <div class="item"><span class="label">后端类型</span><span class="value" id="cfg_backend">加载中...</span></div>
                            <div class="item"><span class="label">GPU设备</span><span class="value" id="cfg_gpu_dev">加载中...</span></div>
                            <div class="item"><span class="label">GPU层数</span><span class="value" id="cfg_ngl">加载中...</span></div>
                            <div class="item"><span class="label">批处理大小</span><span class="value" id="cfg_batch">加载中...</span></div>
                            <div class="item"><span class="label">微批处理</span><span class="value" id="cfg_ubatch">加载中...</span></div>
                            <div class="item"><span class="label">Flash Attn</span><span class="value" id="cfg_flash">加载中...</span></div>
                            <div class="item"><span class="label">内存阈值(MB)</span><span class="value" id="cfg_mem_thresh">加载中...</span></div>
                            <div class="item"><span class="label">显存阈值(MB)</span><span class="value" id="cfg_gpu_thresh">加载中...</span></div>
                            <div class="item"><span class="label">最大并发</span><span class="value" id="cfg_concur">加载中...</span></div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>📊 实时资源监控</h2>
                        <div class="resource-grid">
                            <div class="resource-item"><span class="label">系统内存</span><span class="value" id="res_mem">N/A</span></div>
                            <div class="resource-item"><span class="label">GPU 显存</span><span class="value" id="res_gpu">N/A</span></div>
                            <div class="resource-item"><span class="label">llama-server 内存</span><span class="value" id="res_ram">N/A</span></div>
                            <div class="resource-item"><span class="label">llama-server 显存</span><span class="value" id="res_vram">N/A</span></div>
                            <div class="resource-item" style="grid-column: span 2;"><span class="label">当前并发请求</span><span class="value" id="res_concur">0 / 0</span></div>
                        </div>
                        <div class="refresh-info">🔄 自动刷新 (2秒)</div>
                    </div>
                </div>

                <div class="card" style="margin-top:1rem;">
                    <h2>📖 使用方法 (Windows cmd 单行示例)</h2>
                    <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;">
                        以下命令均为 <strong>单行</strong>，可直接复制到 cmd 执行。JSON 内的双引号已用反斜杠转义。
                        请将 <code>http://127.0.0.1:{proxy_port}</code> 替换为您的实际代理地址。
                    </p>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">1. 对话补全 (Chat Completions)</h3>
                    <div class="cmd-box">
                        <span class="comment"># 非流式请求</span>
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/chat/completions -H "Content-Type: application/json" -d "{{\\"model\\": \\"{model}\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": \\"你好\\"}}], \\"temperature\\": {temp}, \\"max_tokens\\": {max_tokens}}}"
                    </div>
                    <div class="cmd-box">
                        <span class="comment"># 流式请求 (SSE)</span>
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/chat/completions -H "Content-Type: application/json" -d "{{\\"model\\": \\"{model}\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": \\"讲个笑话\\"}}], \\"stream\\": true, \\"temperature\\": {temp}}}"
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">2. 文本补全 (Completions)</h3>
                    <div class="cmd-box">
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/completions -H "Content-Type: application/json" -d "{{\\"model\\": \\"{model}\\", \\"prompt\\": \\"Once upon a time,\\", \\"max_tokens\\": 50, \\"temperature\\": 0.7}}"
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">3. 嵌入向量 (Embeddings)</h3>
                    <div class="cmd-box">
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/embeddings -H "Content-Type: application/json" -d "{{\\"model\\": \\"{model}\\", \\"input\\": \\"Hello world\\"}}"
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">4. 文件上传对话 (带图片/文本)</h3>
                    <div class="cmd-box">
                        <span class="comment"># 上传图片文件 (需要先有文件)</span>
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/chat/completions_with_file -F "file=@C:\\path\\to\\image.jpg" -F "text=请描述这张图片" -F "temperature=0.5"
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">5. 切换模型</h3>
                    <div class="cmd-box">
                        curl -X POST http://127.0.0.1:{proxy_port}/v1/models/switch -H "Content-Type: application/json" -d "{{\\"model\\": \\"new_model.gguf\\", \\"mmproj\\": \\"mmproj.gguf\\", \\"save_config\\": true}}"
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">6. 获取模型列表</h3>
                    <div class="cmd-box">
                        curl http://127.0.0.1:{proxy_port}/v1/models
                    </div>

                    <h3 style="color:#cbd5e1; margin-top:1rem;">7. 健康检查</h3>
                    <div class="cmd-box">
                        curl http://127.0.0.1:{proxy_port}/health
                    </div>

                    <p style="color:#6b7280; font-size:0.85rem; margin-top:1rem;">
                        💡 提示：所有命令均为单行，无需换行符。若在 PowerShell 中执行，请将双引号内的 <code>\\"</code> 改为 <code>`"</code>（反引号转义）。
                    </p>
                </div>

                <div class="footer">
                    OpenAI API 服务器 v1.0.0 &nbsp;|&nbsp; <a href="/docs">文档</a> &nbsp;|&nbsp; 数据仅本地处理，不会外传
                </div>
            </div>

            <script>
                function fetchStatus() {{
                    fetch('/api/status')
                        .then(res => res.json())
                        .then(data => {{
                            const cfg = data.config;
                            document.getElementById('cfg_model').textContent = cfg.model_file || 'N/A';
                            document.getElementById('cfg_mmproj').textContent = cfg.mmproj_file || '无';
                            document.getElementById('cfg_context').textContent = cfg.context_size;
                            document.getElementById('cfg_threads').textContent = cfg.cpu_threads;
                            document.getElementById('cfg_temp').textContent = cfg.temperature.toFixed(2);
                            document.getElementById('cfg_top_p').textContent = cfg.top_p.toFixed(2);
                            document.getElementById('cfg_top_k').textContent = cfg.top_k;
                            document.getElementById('cfg_rep').textContent = cfg.repeat_penalty.toFixed(2);
                            document.getElementById('cfg_pres').textContent = cfg.presence_penalty.toFixed(2);
                            document.getElementById('cfg_max_tokens').textContent = cfg.max_tokens;
                            document.getElementById('cfg_backend').textContent = cfg.backend_type;
                            document.getElementById('cfg_gpu_dev').textContent = cfg.gpu_device || '默认';
                            document.getElementById('cfg_ngl').textContent = cfg.ngl;
                            document.getElementById('cfg_batch').textContent = cfg.batch_size;
                            document.getElementById('cfg_ubatch').textContent = cfg.ubatch_size;
                            document.getElementById('cfg_flash').textContent = cfg.flash_attn;
                            document.getElementById('cfg_mem_thresh').textContent = cfg.memory_threshold_mb;
                            document.getElementById('cfg_gpu_thresh').textContent = cfg.gpu_memory_threshold_mb;
                            document.getElementById('cfg_concur').textContent = cfg.max_concurrent_requests;

                            const res = data.resources;
                            document.getElementById('res_mem').textContent = res.memory_available_mb + ' MB / ' + res.memory_total_mb + ' MB (' + res.memory_percent.toFixed(1) + '%)';
                            if (res.gpu_info && res.gpu_info.length > 0) {{
                                const gpu = res.gpu_info[0];
                                document.getElementById('res_gpu').textContent = '空闲 ' + gpu.free_mb + ' MB / 总计 ' + gpu.total_mb + ' MB';
                            }} else {{
                                document.getElementById('res_gpu').textContent = res.gpu_message || '无可用';
                            }}
                            document.getElementById('res_ram').textContent = (res.llama_process_ram_mb > 0) ? res.llama_process_ram_mb + ' MB' : '未获取';
                            const vram = res.llama_process_gpu_mb;
                            if (vram > 0) {{
                                document.getElementById('res_vram').textContent = vram + ' MB (进程独占)';
                            }} else if (res.gpu_total_used_mb > 0) {{
                                document.getElementById('res_vram').textContent = '总 ' + res.gpu_total_used_mb + ' MB (进程独占不可获取)';
                            }} else {{
                                document.getElementById('res_vram').textContent = '无可用';
                            }}
                            document.getElementById('res_concur').textContent = res.active_requests + ' / ' + res.max_concurrent;

                            const statusElem = document.getElementById('serverStatus');
                            if (data.resources.memory_available_mb > 0) {{
                                statusElem.textContent = '✅ 运行中';
                                statusElem.className = 'status-ok';
                            }} else {{
                                statusElem.textContent = '⚠️ 降级';
                                statusElem.className = 'status-warn';
                            }}
                        }})
                        .catch(err => {{
                            console.error('获取状态失败:', err);
                            document.getElementById('serverStatus').textContent = '❌ 连接失败';
                            document.getElementById('serverStatus').className = 'status-err';
                        }});
                }}

                fetchStatus();
                setInterval(fetchStatus, 2000);
            </script>
        </body>
        </html>
        """

    def _is_backend_ready(self):
        try:
            requests.get(f"http://127.0.0.1:{self.config.llama_server_port}/", timeout=1)
            return True
        except:
            return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            from werkzeug.serving import make_server
            self._server = make_server('0.0.0.0', self.config.proxy_port, self.app, threaded=True)
            self._server.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"代理端口 {self.config.proxy_port} 被占用")
            else:
                logger.error(f"代理服务器启动失败: {e}")
            self.running = False
        except Exception as e:
            logger.error(f"代理服务器启动失败: {e}")
            self.running = False

    def shutdown(self):
        logger.info("代理服务器开始优雅关闭...")
        self.running = False
        if self._server:
            self._server.shutdown()
        with self.active_lock:
            while self.active_requests > 0:
                with self.all_done:
                    self.all_done.wait(timeout=1)
        logger.info("所有活动请求已完成，代理服务器关闭")

    def stop(self):
        self.running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception as e:
                logger.warning(f"代理服务器关闭时出现警告: {e}")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        self.thread = None
        self._server = None

# ==================== 设置面板（左右分栏，含资源监控和日志） ====================
class SettingsWindow(QWidget):
    def __init__(self, config: AppConfig, log_handler: LogHandler, parent=None):
        super().__init__(parent)
        self.config = config
        self.log_handler = log_handler
        self.setWindowTitle("OpenAI API 服务器设置")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowTitleHint | Qt.Window)
        self.setMinimumSize(900, 650)
        self.resize(1050, 750)
        self.setStyleSheet("""
            QWidget { background-color: #2d2d3a; color: white; }
            QGroupBox { border: 1px solid #5a5a70; border-radius: 8px; margin-top: 12px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #ddd; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background-color: #3a3a4a; border: 1px solid #5a5a70; border-radius: 4px; padding: 4px; color: white;
            }
            QPushButton {
                background-color: #5a5a70; border: none; border-radius: 6px; padding: 6px 12px; color: white;
            }
            QPushButton:hover { background-color: #6a6a80; }
            QSlider::groove:horizontal { background: #3a3a4a; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #8a8aa0; width: 16px; margin: -5px 0; border-radius: 8px; }
            QPlainTextEdit { background-color: #1e1e2f; color: #c0c0c0; border: 1px solid #5a5a70; font-family: monospace; }
        """)

        self.log_handler.log_signal.connect(self.append_log)

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ========== 左栏：设置选项 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(5, 5, 5, 5)

        # ---- 模型设置 ----
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.refresh_models()
        self.model_combo.setCurrentText(self.config.model_file)
        self.mmproj_combo = QComboBox()
        self.mmproj_combo.setEditable(True)
        self.refresh_mmproj()
        self.mmproj_combo.setCurrentText(self.config.mmproj_file)
        refresh_btn = QPushButton("刷新模型列表")
        refresh_btn.clicked.connect(lambda: (self.refresh_models(), self.refresh_mmproj()))
        model_layout.addWidget(QLabel("主模型文件:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(QLabel("视觉模型文件(mmproj):"))
        model_layout.addWidget(self.mmproj_combo)
        model_layout.addWidget(refresh_btn)

        self.context_spin = QSpinBox()
        self.context_spin.setRange(1024, 100000)
        self.context_spin.setValue(self.config.context_size)
        model_layout.addWidget(QLabel("上下文长度:"))
        model_layout.addWidget(self.context_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(self.config.cpu_threads)
        model_layout.addWidget(QLabel("CPU线程数:"))
        model_layout.addWidget(self.threads_spin)
        model_group.setLayout(model_layout)
        container_layout.addWidget(model_group)

        # ---- AI 采样参数 ----
        ai_group = QGroupBox("AI采样参数")
        ai_layout = QVBoxLayout()
        # 温度
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("温度:"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.config.temperature * 100))
        self.temp_label = QLabel(f"{self.config.temperature:.2f}")
        self.temp_label.setFixedWidth(40)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/100:.2f}"))
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        ai_layout.addLayout(temp_layout)

        # Top P
        topp_layout = QHBoxLayout()
        topp_layout.addWidget(QLabel("Top P:"))
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(int(self.config.top_p * 100))
        self.top_p_label = QLabel(f"{self.config.top_p:.2f}")
        self.top_p_label.setFixedWidth(40)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_label.setText(f"{v/100:.2f}"))
        topp_layout.addWidget(self.top_p_slider)
        topp_layout.addWidget(self.top_p_label)
        ai_layout.addLayout(topp_layout)

        # Top K
        topk_layout = QHBoxLayout()
        topk_layout.addWidget(QLabel("Top K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(self.config.top_k)
        topk_layout.addWidget(self.top_k_spin)
        topk_layout.addStretch()
        ai_layout.addLayout(topk_layout)

        # 重复惩罚
        rep_layout = QHBoxLayout()
        rep_layout.addWidget(QLabel("重复惩罚:"))
        self.rep_slider = QSlider(Qt.Horizontal)
        self.rep_slider.setRange(100, 200)
        self.rep_slider.setValue(int(self.config.repeat_penalty * 100))
        self.rep_label = QLabel(f"{self.config.repeat_penalty:.2f}")
        self.rep_label.setFixedWidth(40)
        self.rep_slider.valueChanged.connect(lambda v: self.rep_label.setText(f"{v/100:.2f}"))
        rep_layout.addWidget(self.rep_slider)
        rep_layout.addWidget(self.rep_label)
        ai_layout.addLayout(rep_layout)

        # 存在惩罚
        pres_layout = QHBoxLayout()
        pres_layout.addWidget(QLabel("存在惩罚:"))
        self.pres_slider = QSlider(Qt.Horizontal)
        self.pres_slider.setRange(0, 200)
        self.pres_slider.setValue(int(self.config.presence_penalty * 100))
        self.pres_label = QLabel(f"{self.config.presence_penalty:.2f}")
        self.pres_label.setFixedWidth(40)
        self.pres_slider.valueChanged.connect(lambda v: self.pres_label.setText(f"{v/100:.2f}"))
        pres_layout.addWidget(self.pres_slider)
        pres_layout.addWidget(self.pres_label)
        ai_layout.addLayout(pres_layout)

        # 最大token
        maxtok_layout = QHBoxLayout()
        maxtok_layout.addWidget(QLabel("最大生成token:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 32768)
        self.max_tokens_spin.setValue(self.config.max_tokens)
        maxtok_layout.addWidget(self.max_tokens_spin)
        maxtok_layout.addStretch()
        ai_layout.addLayout(maxtok_layout)

        # 系统提示词
        ai_layout.addWidget(QLabel("系统提示词:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.config.system_prompt)
        self.prompt_edit.setMaximumHeight(60)
        ai_layout.addWidget(self.prompt_edit)
        ai_group.setLayout(ai_layout)
        container_layout.addWidget(ai_group)

        # ---- 硬件与网络 ----
        hw_group = QGroupBox("硬件与网络")
        hw_layout = QVBoxLayout()
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("后端类型:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "cuda", "vulkan", "cpu"])
        self.backend_combo.setCurrentText(self.config.backend_type)
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addStretch()
        hw_layout.addLayout(backend_layout)

        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("GPU设备ID:"))
        self.gpu_device_edit = QLineEdit()
        self.gpu_device_edit.setText(self.config.gpu_device)
        self.gpu_device_edit.setPlaceholderText("0 或 0,1")
        gpu_layout.addWidget(self.gpu_device_edit)
        hw_layout.addLayout(gpu_layout)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("后端端口:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self.config.llama_server_port)
        port_layout.addWidget(self.port_spin)
        port_layout.addStretch()
        hw_layout.addLayout(port_layout)

        proxy_layout = QHBoxLayout()
        proxy_layout.addWidget(QLabel("代理端口:"))
        self.proxy_port_spin = QSpinBox()
        self.proxy_port_spin.setRange(1024, 65535)
        self.proxy_port_spin.setValue(self.config.proxy_port)
        proxy_layout.addWidget(self.proxy_port_spin)
        proxy_layout.addStretch()
        hw_layout.addLayout(proxy_layout)

        ngl_layout = QHBoxLayout()
        ngl_layout.addWidget(QLabel("GPU层数:"))
        self.ngl_spin = QSpinBox()
        self.ngl_spin.setRange(0, 999)
        self.ngl_spin.setValue(self.config.ngl)
        ngl_layout.addWidget(self.ngl_spin)
        ngl_layout.addStretch()
        hw_layout.addLayout(ngl_layout)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("批处理大小:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(self.config.batch_size)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        hw_layout.addLayout(batch_layout)

        flash_layout = QHBoxLayout()
        flash_layout.addWidget(QLabel("Flash Attention:"))
        self.flash_combo = QComboBox()
        self.flash_combo.addItems(["auto", "on", "off"])
        self.flash_combo.setCurrentText(self.config.flash_attn)
        flash_layout.addWidget(self.flash_combo)
        flash_layout.addStretch()
        hw_layout.addLayout(flash_layout)

        omp_layout = QHBoxLayout()
        omp_layout.addWidget(QLabel("OMP线程:"))
        self.omp_threads_spin = QSpinBox()
        self.omp_threads_spin.setRange(1, 64)
        self.omp_threads_spin.setValue(self.config.omp_threads)
        omp_layout.addWidget(self.omp_threads_spin)
        omp_layout.addStretch()
        hw_layout.addLayout(omp_layout)

        ubatch_layout = QHBoxLayout()
        ubatch_layout.addWidget(QLabel("微批处理:"))
        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setRange(1, 4096)
        self.ubatch_spin.setValue(self.config.ubatch_size)
        ubatch_layout.addWidget(self.ubatch_spin)
        ubatch_layout.addStretch()
        hw_layout.addLayout(ubatch_layout)

        hw_group.setLayout(hw_layout)
        container_layout.addWidget(hw_group)

        # ---- 超时设置 ----
        adv_group = QGroupBox("超时设置")
        adv_layout = QVBoxLayout()
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("启动超时(秒):"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 10000)
        self.timeout_spin.setValue(self.config.server_start_timeout)
        timeout_layout.addWidget(self.timeout_spin)
        timeout_layout.addStretch()
        adv_layout.addLayout(timeout_layout)

        req_layout = QHBoxLayout()
        req_layout.addWidget(QLabel("请求超时(秒):"))
        self.req_timeout_spin = QSpinBox()
        self.req_timeout_spin.setRange(10, 10000)
        self.req_timeout_spin.setValue(self.config.request_timeout)
        req_layout.addWidget(self.req_timeout_spin)
        req_layout.addStretch()
        adv_layout.addLayout(req_layout)
        adv_group.setLayout(adv_layout)
        container_layout.addWidget(adv_group)

        # ---- 资源保护设置 ----
        res_group = QGroupBox("资源保护")
        res_layout = QVBoxLayout()
        mem_thresh_layout = QHBoxLayout()
        mem_thresh_layout.addWidget(QLabel("系统内存阈值 (MB):"))
        self.mem_thresh_spin = QSpinBox()
        self.mem_thresh_spin.setRange(100, 10000)
        self.mem_thresh_spin.setValue(self.config.memory_threshold_mb)
        mem_thresh_layout.addWidget(self.mem_thresh_spin)
        mem_thresh_layout.addStretch()
        res_layout.addLayout(mem_thresh_layout)

        gpu_thresh_layout = QHBoxLayout()
        gpu_thresh_layout.addWidget(QLabel("NVIDIA显存阈值(MB):"))
        self.gpu_mem_spin = QSpinBox()
        self.gpu_mem_spin.setRange(100, 10000)
        self.gpu_mem_spin.setValue(self.config.gpu_memory_threshold_mb)
        gpu_thresh_layout.addWidget(self.gpu_mem_spin)
        gpu_thresh_layout.addStretch()
        res_layout.addLayout(gpu_thresh_layout)

        intel_layout = QHBoxLayout()
        intel_layout.addWidget(QLabel("Intel核显预留(MB):"))
        self.intel_reserve_spin = QSpinBox()
        self.intel_reserve_spin.setRange(0, 5000)
        self.intel_reserve_spin.setValue(self.config.intel_shared_reserve_mb)
        intel_layout.addWidget(self.intel_reserve_spin)
        intel_layout.addStretch()
        res_layout.addLayout(intel_layout)

        conc_layout = QHBoxLayout()
        conc_layout.addWidget(QLabel("最大并发:"))
        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(1, 200)
        self.max_concurrent_spin.setValue(self.config.max_concurrent_requests)
        conc_layout.addWidget(self.max_concurrent_spin)
        conc_layout.addStretch()
        res_layout.addLayout(conc_layout)
        res_group.setLayout(res_layout)
        container_layout.addWidget(res_group)

        # ---- 智能参数推荐 ----
        rec_group = QGroupBox("参数推荐")
        rec_layout = QVBoxLayout()
        self.smart_btn = QPushButton("智能参数推荐")
        self.smart_btn.clicked.connect(self.smart_recommend)
        rec_layout.addWidget(self.smart_btn)
        rec_group.setLayout(rec_layout)
        container_layout.addWidget(rec_group)

        container_layout.addStretch()

        scroll.setWidget(container)
        left_layout.addWidget(scroll)
        main_layout.addWidget(left_widget, 55)

        # ========== 右栏：资源监控 + 日志 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)

        monitor_group = QGroupBox("实时资源监控")
        monitor_layout = QFormLayout()
        monitor_layout.setSpacing(4)
        self.res_mem_label = QLabel("N/A")
        self.res_gpu_label = QLabel("N/A")
        self.res_ram_label = QLabel("N/A")
        self.res_vram_label = QLabel("N/A")
        self.res_concur_label = QLabel("N/A")
        monitor_layout.addRow("系统内存:", self.res_mem_label)
        monitor_layout.addRow("GPU 显存:", self.res_gpu_label)
        monitor_layout.addRow("llama-server 内存:", self.res_ram_label)
        monitor_layout.addRow("llama-server 显存:", self.res_vram_label)
        monitor_layout.addRow("当前并发:", self.res_concur_label)
        monitor_group.setLayout(monitor_layout)
        right_layout.addWidget(monitor_group)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.log_text)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("保存并重启服务")
        cancel_btn = QPushButton("取消")
        clear_btn = QPushButton("清空日志")
        save_btn.clicked.connect(self.save_and_restart)
        cancel_btn.clicked.connect(self.close)
        clear_btn.clicked.connect(self.log_text.clear)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)
        log_layout.addLayout(btn_row)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_widget, 45)

        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self.refresh_resource_info)
        self.res_timer.start(2000)

    # ---------- 智能参数推荐（增强错误反馈） ----------
    def smart_recommend(self):
        try:
            if PSUTIL_AVAILABLE:
                mem = psutil.virtual_memory()
                total_mem_mb = mem.total // (1024 * 1024)
                available_mem_mb = mem.available // (1024 * 1024)
                cpu_physical = psutil.cpu_count(logical=False)
                cpu_logical = psutil.cpu_count(logical=True)
                if cpu_physical is None:
                    cpu_physical = cpu_logical or 2
                if cpu_logical is None:
                    cpu_logical = cpu_physical
            else:
                total_mem_mb = 4096
                available_mem_mb = 2048
                cpu_physical = 4
                cpu_logical = 8

            gpu_total_mb = 0
            has_nvidia = False
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        has_nvidia = True
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_total_mb = mem_info.total // (1024 * 1024)
                    pynvml.nvmlShutdown()
                except Exception as e:
                    logger.warning(f"获取显存信息失败: {e}")

            if gpu_total_mb == 0:
                gpu_total_mb = total_mem_mb // 2

            threads = max(1, min(64, cpu_physical))
            if threads == 0:
                threads = max(1, cpu_logical // 2)

            backend = "cuda" if has_nvidia else "auto"
            gpu_device = "0"
            omp_threads = threads
            flash_attn = "on" if gpu_total_mb >= 4096 else "auto"

            context = max(1024, min(32768, int(available_mem_mb / 4)))
            max_tokens = min(2048, context // 2)
            if gpu_total_mb >= 8192:
                batch = 2048
            elif gpu_total_mb >= 4096:
                batch = 1024
            elif gpu_total_mb >= 2048:
                batch = 512
            else:
                batch = 256
            ubatch = batch
            if gpu_total_mb >= 4096:
                ngl = 999
            elif gpu_total_mb >= 2048:
                ngl = 50
            elif gpu_total_mb >= 1024:
                ngl = 30
            else:
                ngl = 0

            if total_mem_mb >= 16384:
                start_timeout = 600
            elif total_mem_mb >= 8192:
                start_timeout = 300
            else:
                start_timeout = 180
            req_timeout = 120
            mem_threshold = max(100, int(total_mem_mb * 0.05))
            gpu_threshold = max(100, int(gpu_total_mb * 0.05))
            intel_reserve = 1024
            max_concurrent = 50

            self.threads_spin.setValue(threads)
            self.backend_combo.setCurrentText(backend)
            self.gpu_device_edit.setText(gpu_device)
            self.omp_threads_spin.setValue(omp_threads)
            self.flash_combo.setCurrentText(flash_attn)
            self.context_spin.setValue(context)
            self.max_tokens_spin.setValue(max_tokens)
            self.batch_spin.setValue(batch)
            self.ubatch_spin.setValue(ubatch)
            self.ngl_spin.setValue(ngl)
            self.timeout_spin.setValue(start_timeout)
            self.req_timeout_spin.setValue(req_timeout)
            self.mem_thresh_spin.setValue(mem_threshold)
            self.gpu_mem_spin.setValue(gpu_threshold)
            self.intel_reserve_spin.setValue(intel_reserve)
            self.max_concurrent_spin.setValue(max_concurrent)

            sampling_loaded = False
            sampling_info = ""
            port = self.port_spin.value()
            try:
                url = f"http://127.0.0.1:{port}/props"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    params = data.get("default_generation_settings", {}).get("params", {})
                    if params:
                        temp = params.get("temperature", self.temp_slider.value() / 100)
                        self.temp_slider.setValue(int(temp * 100))
                        self.temp_label.setText(f"{temp:.2f}")

                        top_p = params.get("top_p", self.top_p_slider.value() / 100)
                        self.top_p_slider.setValue(int(top_p * 100))
                        self.top_p_label.setText(f"{top_p:.2f}")

                        top_k = params.get("top_k", self.top_k_spin.value())
                        self.top_k_spin.setValue(top_k)

                        repeat = params.get("repeat_penalty", self.rep_slider.value() / 100)
                        self.rep_slider.setValue(int(repeat * 100))
                        self.rep_label.setText(f"{repeat:.2f}")

                        presence = params.get("presence_penalty", self.pres_slider.value() / 100)
                        self.pres_slider.setValue(int(presence * 100))
                        self.pres_label.setText(f"{presence:.2f}")

                        max_tok = params.get("max_tokens", -1)
                        if max_tok > 0:
                            self.max_tokens_spin.setValue(max_tok)
                        sampling_loaded = True
                        sampling_info = (
                            f"温度: {temp:.2f}\nTop P: {top_p:.2f}\nTop K: {top_k}\n"
                            f"重复惩罚: {repeat:.2f}\n存在惩罚: {presence:.2f}\n最大Token: {self.max_tokens_spin.value()}"
                        )
            except requests.exceptions.ConnectionError:
                logger.warning("无法连接到 llama-server 获取采样参数")
            except Exception as e:
                logger.warning(f"加载模型采样参数失败: {e}")

            report = (
                f"智能参数推荐完成！\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 硬件检测结果：\n"
                f"  系统总内存: {total_mem_mb} MB\n"
                f"  可用内存:   {available_mem_mb} MB\n"
                f"  CPU物理核心: {cpu_physical} 核\n"
                f"  CPU逻辑核心: {cpu_logical} 核\n"
                f"  GPU显存:    {gpu_total_mb} MB {'(NVIDIA)' if has_nvidia else '(估算值)'}\n"
                f"\n⚙️  硬件推荐参数：\n"
                f"  CPU线程数:          {threads}\n"
                f"  OMP线程数:          {omp_threads}\n"
                f"  后端类型:           {backend}\n"
                f"  GPU设备ID:          {gpu_device}\n"
                f"  Flash Attention:    {flash_attn}\n"
                f"  上下文长度:         {context}\n"
                f"  批处理大小:         {batch}\n"
                f"  微批处理:           {ubatch}\n"
                f"  GPU层数:            {ngl}\n"
                f"  启动超时(秒):       {start_timeout}\n"
                f"  请求超时(秒):       {req_timeout}\n"
                f"  内存阈值(MB):       {mem_threshold}\n"
                f"  显存阈值(MB):       {gpu_threshold}\n"
                f"  Intel核显预留(MB):  {intel_reserve}\n"
                f"  最大并发:           {max_concurrent}\n"
            )
            if sampling_loaded:
                report += f"\n🎯 从当前模型加载的采样参数：\n{sampling_info}"
            else:
                report += "\n⚠️ 未能从当前模型加载采样参数（请确保服务已启动且支持 /props 端点）。\n   若需要，您可以手动调整温度、Top P 等参数。"
            report += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n您仍可手动微调，然后点击『保存并重启服务』生效。"
            QMessageBox.information(self, "推荐完成", report)

        except Exception as e:
            QMessageBox.warning(
                self, 
                "推荐失败", 
                f"智能推荐时发生错误：{str(e)}\n\n"
                "可能原因：\n"
                "- 硬件信息获取失败\n"
                "- 网络连接异常\n"
                "请检查日志或重试。"
            )

    def append_log(self, msg):
        self.log_text.appendPlainText(msg)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def refresh_resource_info(self):
        parent = self.parent()
        if parent and hasattr(parent, 'proxy') and parent.proxy:
            proxy = parent.proxy
            info = proxy._get_resource_info()
            mem_text = f"空闲 {info['memory_available_mb']} MB / 总计 {info['memory_total_mb']} MB ({info['memory_percent']:.1f}% 已用)"
            self.res_mem_label.setText(mem_text)

            if info["gpu_info"]:
                gpu = info["gpu_info"][0]
                gpu_text = f"NVIDIA GPU {gpu['device_id']}: 空闲 {gpu['free_mb']} MB / 总计 {gpu['total_mb']} MB"
            else:
                gpu_text = info.get("gpu_message", "无可用 GPU 信息")
            self.res_gpu_label.setText(gpu_text)

            ram = info.get("llama_process_ram_mb", 0)
            self.res_ram_label.setText(f"{ram} MB" if ram > 0 else "未获取")

            vram = info.get("llama_process_gpu_mb", 0)
            total_used = info.get("gpu_total_used_mb", 0)
            if vram > 0:
                self.res_vram_label.setText(f"{vram} MB (进程独占)")
            elif total_used > 0:
                self.res_vram_label.setText(f"总 {total_used} MB (进程独占不可获取)")
            else:
                backend = self.config.backend_type.lower()
                if backend == "cuda":
                    self.res_vram_label.setText("未获取到显存信息 (可能 WDDM 模式限制)")
                elif backend == "vulkan":
                    self.res_vram_label.setText("Vulkan 后端（显存信息不适用）")
                elif backend == "cpu":
                    self.res_vram_label.setText("CPU 模式无显存")
                else:
                    self.res_vram_label.setText("无可用显存信息")

            self.res_concur_label.setText(f"{info['active_requests']} / {info['max_concurrent']}")
        else:
            self.res_mem_label.setText("服务未运行")
            self.res_gpu_label.setText("N/A")
            self.res_ram_label.setText("N/A")
            self.res_vram_label.setText("N/A")
            self.res_concur_label.setText("N/A")

    def refresh_models(self):
        self.model_combo.clear()
        model_dir = os.path.join(BASE_PATH, "Model")
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.gguf') and 'mmproj' not in f.lower():
                    self.model_combo.addItem(f)

    def refresh_mmproj(self):
        self.mmproj_combo.clear()
        model_dir = os.path.join(BASE_PATH, "Model")
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.gguf') and 'mmproj' in f.lower():
                    self.mmproj_combo.addItem(f)

    def save_and_restart(self):
        new_config = AppConfig(
            model_file=self.model_combo.currentText(),
            mmproj_file=self.mmproj_combo.currentText(),
            context_size=self.context_spin.value(),
            cpu_threads=self.threads_spin.value(),
            temperature=self.temp_slider.value() / 100,
            top_p=self.top_p_slider.value() / 100,
            top_k=self.top_k_spin.value(),
            repeat_penalty=self.rep_slider.value() / 100,
            presence_penalty=self.pres_slider.value() / 100,
            max_tokens=self.max_tokens_spin.value(),
            system_prompt=self.prompt_edit.toPlainText(),
            backend_type=self.backend_combo.currentText(),
            gpu_device=self.gpu_device_edit.text().strip(),
            omp_threads=self.omp_threads_spin.value(),
            ngl=self.ngl_spin.value(),
            batch_size=self.batch_spin.value(),
            ubatch_size=self.ubatch_spin.value(),
            flash_attn=self.flash_combo.currentText(),
            llama_server_port=self.port_spin.value(),
            proxy_port=self.proxy_port_spin.value(),
            server_start_timeout=self.timeout_spin.value(),
            request_timeout=self.req_timeout_spin.value(),
            server_ready_timeout=self.config.server_ready_timeout,
            memory_threshold_mb=self.mem_thresh_spin.value(),
            gpu_memory_threshold_mb=self.gpu_mem_spin.value(),
            intel_shared_reserve_mb=self.intel_reserve_spin.value(),
            max_concurrent_requests=self.max_concurrent_spin.value(),
            settings_window_x=self.config.settings_window_x,
            settings_window_y=self.config.settings_window_y,
            settings_window_width=self.config.settings_window_width,
            settings_window_height=self.config.settings_window_height,
        )
        ok, err = new_config.save()
        if not ok:
            QMessageBox.warning(self, "保存失败", f"配置保存失败：{err}\n\n请检查权限或磁盘空间。")
            return
        if self.parent():
            self.parent().apply_config_and_restart(new_config)
        self.close()

    def showEvent(self, event):
        super().showEvent(event)
        x = self.config.settings_window_x
        y = self.config.settings_window_y
        w = self.config.settings_window_width
        h = self.config.settings_window_height

        if x >= 0 and y >= 0:
            self.move(x, y)
        else:
            screen = QApplication.primaryScreen().geometry()
            self.move(screen.center() - self.rect().center())

        if w > 0 and h > 0:
            self.resize(w, h)

    def closeEvent(self, event):
        self.res_timer.stop()
        pos = self.pos()
        size = self.size()
        self.config.settings_window_x = pos.x()
        self.config.settings_window_y = pos.y()
        self.config.settings_window_width = size.width()
        self.config.settings_window_height = size.height()
        self.config.save()
        super().closeEvent(event)

# ==================== 主程序：系统托盘应用 ====================
class TrayApp(QWidget):
    switch_model_signal = pyqtSignal(str, str, bool)

    def __init__(self):
        super().__init__()
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base, "Llama-OpenAI-Server.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.hide()
        self.config = AppConfig.load()
        self.server_manager = LlamaServerManager()
        self.proxy = None
        self.service_running = False
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_tray_icon)
        self.settings_window = None
        self.log_handler = LogHandler()
        logging.getLogger().addHandler(self.log_handler)
        self.init_tray()
        self.switch_model_signal.connect(self._do_switch_model)
        self.start_service_if_needed()
        self.status_timer.start(2000)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

    def _signal_handler(self, signum, frame):
        logger.info(f"收到信号 {signum}，准备退出...")
        QApplication.quit()

    def _cleanup(self):
        try:
            logging.getLogger().removeHandler(self.log_handler)
        except Exception:
            pass
        if self.proxy and self.proxy.running:
            self.proxy.shutdown()
        try:
            self.server_manager.stop()
        except Exception:
            pass
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def init_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            QMessageBox.critical(None, "错误", "系统托盘不可用，程序无法运行。")
            sys.exit(1)

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("OpenAI API 服务器 (llama-server)")

        menu = QMenu()
        self.start_action = QAction("启动服务", self)
        self.stop_action = QAction("停止服务", self)
        self.restart_action = QAction("重启服务", self)
        self.settings_action = QAction("设置", self)
        self.quit_action = QAction("退出", self)

        self.start_action.triggered.connect(self.start_service)
        self.stop_action.triggered.connect(self.stop_service)
        self.restart_action.triggered.connect(self.restart_service)
        self.settings_action.triggered.connect(self.show_settings)
        self.quit_action.triggered.connect(self.quit_app)

        menu.addAction(self.start_action)
        menu.addAction(self.stop_action)
        menu.addAction(self.restart_action)
        menu.addSeparator()
        menu.addAction(self.settings_action)
        menu.addSeparator()
        menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(menu)
        self.update_tray_icon()
        self.tray_icon.show()
        self.tray_icon.activated.connect(self.on_tray_activated)

    def restart_service(self):
        self.stop_service()
        time.sleep(1)
        self.start_service()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_settings()

    def update_tray_icon(self):
        is_running = self.server_manager.is_running()
        self.service_running = is_running
        size = 32
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(0, 200, 0) if is_running else QColor(200, 0, 0)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, size-8, size-8)
        painter.end()
        self.tray_icon.setIcon(QIcon(pixmap))

        proxy_status = f"代理端口: {self.config.proxy_port}" if FLASK_AVAILABLE else "代理不可用"
        model_name = self.config.model_file
        if len(model_name) > 30:
            model_name = model_name[:27] + "..."
        tooltip = f"""OpenAI API 服务器
状态: {'🟢 运行中' if is_running else '🔴 已停止'}
模型: {model_name}
后端端口: {self.config.llama_server_port}
{proxy_status}
双击打开设置"""
        self.tray_icon.setToolTip(tooltip)
        self.start_action.setEnabled(not is_running)
        self.stop_action.setEnabled(is_running)
        self.restart_action.setEnabled(is_running)

    def start_service(self):
        if self.service_running:
            logger.warning("服务已在运行中")
            return
        success, err_msg, suggestion = self.server_manager.start(self.config)
        if not success:
            QMessageBox.critical(
                None,
                "启动失败",
                f"错误：{err_msg}\n\n建议：{suggestion}"
            )
            return
        self.service_running = True
        if FLASK_AVAILABLE:
            try:
                self.proxy = ProxyServer(self.config, switch_signal=self.switch_model_signal)
                if self.server_manager.pid:
                    self.proxy.set_llama_pid(self.server_manager.pid)
                self.proxy.start()
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "代理启动失败",
                    f"Flask 代理启动异常：{str(e)}\n\n请检查端口 {self.config.proxy_port} 是否可用。"
                )
                self.service_running = False
                self.server_manager.stop()
                return
        else:
            QMessageBox.warning(None, "代理不可用", "Flask 未安装，无法启动代理服务。\n请运行：pip install Flask flask_cors")
            self.service_running = False
            self.server_manager.stop()
            return
        self.update_tray_icon()
        self.tray_icon.showMessage("服务启动", f"llama-server 已启动\n代理端口 {self.config.proxy_port}", QSystemTrayIcon.Information, 2000)

    def stop_service(self):
        if not self.service_running:
            return
        if self.proxy:
            self.proxy.shutdown()
        self.server_manager.stop()
        self.service_running = False
        self.update_tray_icon()
        self.tray_icon.showMessage("服务停止", "llama-server 已停止", QSystemTrayIcon.Information, 2000)

    def apply_config_and_restart(self, new_config: AppConfig):
        was_running = self.service_running
        new_config.settings_window_x = self.config.settings_window_x
        new_config.settings_window_y = self.config.settings_window_y
        new_config.settings_window_width = self.config.settings_window_width
        new_config.settings_window_height = self.config.settings_window_height
        self.config = new_config
        if was_running:
            self.stop_service()
            time.sleep(1)
            self.start_service()
        self.update_tray_icon()

    def show_settings(self):
        if hasattr(self, 'settings_window') and self.settings_window is not None and self.settings_window.isVisible():
            self.settings_window.raise_()
            self.settings_window.activateWindow()
            return
        self.settings_window = SettingsWindow(self.config, self.log_handler, self)
        self.settings_window.show()

    def start_service_if_needed(self):
        self.start_service()

    def _do_switch_model(self, model_file, mmproj_file, save_config):
        was_running = self.service_running
        if was_running:
            self.stop_service()
            time.sleep(1)
        self.config.model_file = model_file
        self.config.mmproj_file = mmproj_file
        if save_config:
            ok, err = self.config.save()
            if not ok:
                QMessageBox.warning(None, "保存配置失败", f"无法保存配置：{err}")
        if was_running:
            self.start_service()
            if self.service_running:
                self.tray_icon.showMessage("模型切换", f"已切换到 {model_file}", QSystemTrayIcon.Information, 2000)
            else:
                QMessageBox.warning(None, "切换失败", "模型切换后启动服务失败，请查看日志。")
        else:
            logger.info(f"模型已更新为 {model_file}，服务未启动")
        self.update_tray_icon()

    def quit_app(self):
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if self.settings_window:
            self.settings_window.close()
            self.settings_window = None
        self.stop_service()
        QApplication.quit()

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    tray_app = TrayApp()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()