# -*- coding: utf-8 -*-
import shutil
import sys
import os
import time
import json
import threading
import queue
import logging
import base64
import requests
import subprocess
import numpy as np
import cv2
import traceback
import uuid
from PIL import Image
import io
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
import socket

# Import PyQt5 components safely
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QGridLayout, QLabel, QPushButton,
                                 QTextEdit, QTabWidget, QComboBox, QLineEdit,
                                 QGroupBox, QFormLayout, QSpinBox, QFileDialog,
                                 QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
                                 QDialog, QHeaderView, QSplitter, QCheckBox, QScrollArea, QSizePolicy, QRadioButton,
                                 QGraphicsScene, QGraphicsView, QInputDialog, QGraphicsItem, QToolTip, QListWidgetItem,
                                 QListWidget, QStackedWidget, QDoubleSpinBox, QAbstractItemView)
    from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QTransform, QPainter, QPen  # Added QColor
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QMutex, pyqtSlot, QMetaObject, Q_ARG, QRectF, \
        QPointF, QMutexLocker
except ImportError:
    print("PyQt5 is not installed. Please install it: pip install PyQt5")
    sys.exit(1)

# --- Configuration ---

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smartphone_automation.log", encoding='utf-8'),  # Specify encoding
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smartphone_automation")

# 为 AI 决策创建单独的 logger
ai_decision_logger = logging.getLogger("AIDecisionLogger")
ai_decision_logger.setLevel(logging.INFO)  # 设置 AI 决策日志级别
# 移除默认的传播，避免重复输出到主日志
ai_decision_logger.propagate = False

# 全局配置 (Defaults)
DEFAULT_CONFIG = {
    "ESP_IP": "192.168.101.11",
    "ESP_PORT": 8080,
    "SCREENSHOT_RESOLUTION": (1080, 1980),  # Default, can be overridden per device
    "CROPPED_RESOLUTION": (1080, 1440),  # Default, can be overridden per device
    "OCR_API_URL": "http://127.0.0.1:1224/api/ocr",
    "AI_API_URL": "https://api.lingyiwanwu.com/v1/chat/completions",
    "AI_API_KEY": "",
    "AI_MODEL": "yi-large-rag",  # Specify default AI model
    "ADB_PATH": "adb",
    "CAMERA_DEVICE_ID": "",  # Primary camera device ID
    "COMMAND_TIMEOUT": 15,  # Increased default timeout
    "WAIT_TIMEOUT": 300,
    "RETRY_COUNT": 3,
    "SUBTASK_RETRY_COUNT": 1,  # Specific retries for subtasks
    "TASK_POLLING_INTERVAL": 1,
    "CAMERA_STABILIZATION_DELAY_SECONDS": 1.0,
    "MAX_TASK_TIME": 3600,
    "AI_INTERVENTION_TIMEOUT": 300,  # 这个超时现在可能意义不大了，因为有人工干预
    "HUMAN_INTERVENTION_ALERT": True,  # 这个是错误提示音，保留
    "MIN_ESP_COMMAND_INTERVAL": 0.05,  # Min interval between ESP commands
    "ADB_COMMAND_TIMEOUT": 20,  # Timeout for ADB operations
    "OCR_CONFIDENCE_THRESHOLD": 0.6,  # Minimum confidence for OCR text to be considered reliable
    "TEMPLATE_MATCHING_THRESHOLD": 0.75,  # Default threshold for template matching
    "DEVICE_CONFIGS": {
        # --- 示例设备配置 (包含新的原点偏移量) ---
        "Device_001_Example": {
            "SCREENSHOT_RESOLUTION": [1080, 1920],
            "CROPPED_RESOLUTION": [1080, 1440],
            "HOME_SCREEN_TEMPLATE_NAME": "home_template_example",
            "HOME_SCREEN_TEMPLATE_THRESHOLD": 0.8,
            "HOME_SCREEN_ANCHOR_TEXTS": ["电话", "短信", "微信", "相机", "设置", "图库"],
            "HOME_SCREEN_MIN_ANCHORS": 3,
            "COORDINATE_MAP": {"scale_x": 1.0, "offset_y": 10}, # 这个用于像素内的微调 (如果需要)
            # --- 新增：设备在机器人坐标系中的物理原点偏移 ---
            "machine_origin_x": 0,  # 第一个设备通常在原点
            "machine_origin_y": 0
            # --- 新增结束 ---
        },
        "Device_002_Example": { # 假设这是第二个设备
            "SCREENSHOT_RESOLUTION": [1080, 1920],
            "CROPPED_RESOLUTION": [1080, 1440],
            "HOME_SCREEN_ANCHOR_TEXTS": ["Phone", "Messages"],
            "HOME_SCREEN_MIN_ANCHORS": 2,
            "machine_origin_x": 100, # 第二个设备原点在 X100, Y0
            "machine_origin_y": 0
        },
        "Device_004_Example": { # 假设这是第四个设备
             "SCREENSHOT_RESOLUTION": [1080, 1920],
             "CROPPED_RESOLUTION": [1080, 1440],
             "machine_origin_x": 0,  # 第四个设备原点在 X0, Y100
             "machine_origin_y": 100
        }
    },
    # "TASK_DEFINITIONS_FILE": "tasks.json", # 移除此项，统一使用 USER_TASKS
    "USER_TASKS": [],  # 新增：用于存储用户定义任务的列表
    "MAX_LOOP_ITERATIONS": 1000, # 防止无限循环的最大迭代次数

    # --- 新增配置 ---
    "ENABLE_HUMAN_INTERVENTION": False,  # 是否默认启用人工干预模式
    "APP_LAUNCH_TIMEOUT": 60,  # 启动应用的总超时时间 (秒)
    "APP_LAUNCH_SWIPE_ATTEMPTS": 4,  # 查找应用时允许的最大滑动次数 (右滑*2 + 左滑*2)
    "APP_LAUNCH_CLICK_WAIT_SECONDS": 5,  # 点击应用图标后等待的时间
    "MAX_BACK_ATTEMPTS_TO_HOME": 8,  # 返回主屏幕时最多按几次返回键
    # --- 新增的默认主屏幕验证配置 ---

    "HOME_SCREEN_MIN_ANCHORS": 2,  # 全局默认最少锚点数
    "HOME_SCREEN_TEMPLATE_NAME": None,  # 全局默认不使用模板验证主屏幕
    "HOME_SCREEN_TEMPLATE_THRESHOLD": 0.85,  # 全局默认模板阈值（如果使用模板）
    # --- 新增配置结束 ---
}
# Example Device Config:
# "DEVICE_CONFIGS": {
#   "Device_001": {
#     "SCREENSHOT_RESOLUTION": [1080, 1980],
#     "CROPPED_RESOLUTION": [1080, 1440],
#     "COORDINATE_MAP": {
#         "scale_x": 1.0, "offset_x": 0,
#         "scale_y": 1.0, "offset_y": 0
#     }
#   }
# }

# Load config from file, overriding defaults
CONFIG = DEFAULT_CONFIG.copy()
try:
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding='utf-8') as f:
            loaded_config = json.load(f)
            # Deep merge dictionaries like DEVICE_CONFIGS if necessary
            for key, value in loaded_config.items():
                # 修正：DEVICE_CONFIGS 和 USER_TASKS 直接覆盖，其他字典类型更新
                if isinstance(value, dict) and isinstance(CONFIG.get(key), dict) and key not in ["DEVICE_CONFIGS",
                                                                                                 "USER_TASKS"]:
                    CONFIG[key].update(value)
                else:
                    CONFIG[key] = value  # 直接使用加载的值（覆盖默认）
        logger.info("Configuration loaded from config.json")
except Exception as e:
    logger.error(f"加载配置文件 config.json 错误: {str(e)}", exc_info=True)

# --- Enums ---
class DeviceStatus(Enum):
    DISCONNECTED = "断开连接"
    IDLE = "空闲"
    BUSY = "执行中"
    WAITING = "等待中"
    ERROR = "错误"
    INITIALIZING = "初始化中"  # Added status


class TaskType(Enum):
    CLICK_AD = "点击广告"
    WATCH_VIDEO = "刷视频"
    WATCH_LIVE = "挂直播间"
    PLAY_GAME = "挂游戏"
    CUSTOM = "自定义任务"  # Can be AI or Subtask driven
    # Add more specific types if needed


class ActionType(Enum):
    CLICK = "M1"
    LONG_PRESS = "M2"
    SWIPE = "M3"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"  # Added status


class SubtaskType(Enum):
    WAIT = "wait"
    FIND_AND_CLICK_TEXT = "find_and_click_text"
    TEMPLATE_CLICK = "template_click"
    SWIPE = "swipe"
    BACK = "back"
    AI_STEP = "ai_step"
    ESP_COMMAND = "esp_command"
    CHECK_TEXT_EXISTS = "check_text_exists"        # 检查文本是否存在 (可带条件跳转)
    CHECK_TEMPLATE_EXISTS = "check_template_exists" # 检查模板是否存在 (可带条件跳转)
    LOOP_START = "loop_start"                      # 循环开始标记
    LOOP_END = "loop_end"                          # 循环结束标记
    COMMENT = "comment"                            # 新增：注释/空操作类型


# --- Core Classes ---

class Device:
    """Represents a target device being controlled."""

    def __init__(self, name: str, apps: Optional[List[str]] = None, position: Optional[str] = None):
        self.name: str = name
        self.apps: List[str] = apps or []
        self.position: str = position or ""
        self.status: DeviceStatus = DeviceStatus.IDLE
        self.current_task: Optional['Task'] = None
        self.start_time: Optional[datetime] = None
        self.last_screenshot: Optional[np.ndarray] = None
        self.last_ocr_result: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = [] # 列表中的每个字典现在可以包含 'result_success' 和 'result_error'
        self.error_count: int = 0
        self.waiting_until: Optional[datetime] = None
        self.last_update_time: datetime = datetime.now()
        self.task_progress: str = ""

        # --- 修改开始: 规范化设备配置的键 ---
        # 从全局 CONFIG 获取原始的设备配置字典 (可能是小写键)
        raw_device_config = CONFIG.get("DEVICE_CONFIGS", {}).get(self.name, {})
        # 创建一个新的字典，将已知的配置键转换为大写
        normalized_config = {}
        # 定义需要规范化大小写的已知配置键 (可以根据需要扩展)
        known_keys_to_normalize = [
            "screenshot_resolution", "cropped_resolution",
            "home_screen_template_name", "home_screen_template_threshold",
            "home_screen_anchor_texts", "home_screen_min_anchors",
            "coordinate_map", "machine_origin_x", "machine_origin_y",
            "apps", "position" # 也包含一些可能在设备配置中但通常是 Device 属性的键
        ]
        # 将原始配置中的所有键值对复制到新字典
        # 如果键（忽略大小写）匹配已知键，则使用大写键存储
        raw_keys_lower = {k.lower(): k for k in raw_device_config.keys()} # 创建小写键到原始键的映射

        for known_key_upper in known_keys_to_normalize:
            known_key_lower = known_key_upper.lower()
            if known_key_lower in raw_keys_lower:
                original_case_key = raw_keys_lower[known_key_lower]
                normalized_config[known_key_upper] = raw_device_config[original_case_key]
            # 保留原始配置中未知的键（保持原始大小写）
            # elif known_key_upper not in [k.upper() for k in raw_keys_lower]: # 避免重复添加
            #     # 这个逻辑有点复杂，暂时简化为只处理已知键
            #     pass

        # 将原始配置中不在 known_keys_to_normalize 列表中的键也添加进来（保持原样）
        known_keys_lower_set = {k.lower() for k in known_keys_to_normalize}
        for key, value in raw_device_config.items():
             if key.lower() not in known_keys_lower_set:
                 normalized_config[key] = value # 添加未知键

        self._config: Dict[str, Any] = normalized_config # 使用规范化后的配置
        # --- 修改结束 ---

        logger.info(f"Device '{name}' initialized with normalized config: {self._config}") # 日志输出规范化后的配置

    def get_config(self, key: str, default: Any = None) -> Any:
        """Gets a device-specific config value, falling back to global default."""
        # 现在 self._config 内部已经是大写键了，可以直接查找
        # 但为了更健壮，仍然可以保留 fallback 到全局 CONFIG
        # 注意：确保传入的 key 是大写的
        key_upper = key.upper()  # 确保查找时使用大写
        return self._config.get(key_upper, CONFIG.get(key_upper, default))

    def start_task(self, task: 'Task') -> None:
        self.current_task = task
        self.start_time = datetime.now()
        self.status = DeviceStatus.INITIALIZING  # Start in initializing state
        self.action_history = []
        self.error_count = 0
        self.task_progress = "任务初始化中..."
        self.last_update_time = datetime.now()
        logger.info(f"Device '{self.name}' starting task '{task.name}'")

    def set_waiting(self, wait_seconds: Union[int, float]) -> None:
        if wait_seconds <= 0:
            return
        self.status = DeviceStatus.WAITING
        self.waiting_until = datetime.now() + timedelta(seconds=wait_seconds)
        self.task_progress = f"等待中... ({wait_seconds:.1f}秒)"
        self.last_update_time = datetime.now()
        logger.info(f"Device '{self.name}' set to wait for {wait_seconds} seconds.")

    def complete_task(self, success: bool = True) -> None:
        task_name = self.current_task.name if self.current_task else "N/A"
        self.current_task = None
        self.status = DeviceStatus.IDLE if success else DeviceStatus.ERROR
        self.waiting_until = None
        self.task_progress = "任务完成" if success else "任务失败"
        self.last_update_time = datetime.now()
        logger.info(f"Device '{self.name}' completed task '{task_name}' (Success: {success})")

    def update_progress(self, progress: str) -> None:
        self.task_progress = progress
        self.last_update_time = datetime.now()

    def add_action(self,
                   action_details: Dict[str, Any],
                   action_result: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds an action to the history, including its result if provided.
        Args:
            action_details: Dictionary containing 'action', 'rationale'.
            action_result: Dictionary from an executed action, typically containing 'success' and 'error'.
        """
        action_entry = {
            "action": action_details.get("action", "Unknown"),
            "rationale": action_details.get("rationale", ""),
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Add milliseconds
        }
        if action_result:
            action_entry["result_success"] = action_result.get("success", False)
            if not action_result.get("success"):
                action_entry["result_error"] = action_result.get("error", "未知错误")
            # 如果是检查类子任务，也记录其检查结果
            if "condition_met" in action_result:
                 action_entry["condition_met"] = action_result.get("condition_met")


        self.action_history.append(action_entry)
        # Limit history size
        if len(self.action_history) > 50: # 保持合理数量的历史记录
            self.action_history = self.action_history[-50:]
        self.last_update_time = datetime.now()
        logger.debug(f"Device '{self.name}' action: {action_entry['action']} - Result: {action_result.get('success') if action_result else 'N/A'}")


    def get_runtime(self) -> str:
        if not self.start_time:
            return "未开始"
        return str(datetime.now() - self.start_time).split('.')[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "apps": self.apps,
            "position": self.position,
            "status": self.status.value,  # Use enum value
            "current_task": self.current_task.name if self.current_task else None,
            "running_time": self.get_runtime(),
            "task_progress": self.task_progress
        }

class Task:
    """Represents an automation task."""

    def __init__(self,
                 name: str,
                 task_type: TaskType,
                 app_name: str = "",
                 subtasks: Optional[List[Dict[str, Any]]] = None,
                 priority: int = 0,
                 use_ai_driver: bool = True,
                 task_id: Optional[str] = None,
                 max_retries: int = 0): # 任务级别的总重试次数
        self.name: str = name
        self.type: TaskType = task_type
        self.app_name: str = app_name
        self.subtasks: List[Dict[str, Any]] = subtasks or []
        self.priority: int = priority
        # 如果有子任务，则强制 use_ai_driver 为 False
        self.use_ai_driver: bool = use_ai_driver if not subtasks else False
        self.status: TaskStatus = TaskStatus.PENDING
        self.assigned_device: Optional[Device] = None
        self.assigned_device_name: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        # current_step 用于 AI 驱动模式
        self.current_step: int = 0
        # current_subtask_index 用于子任务驱动模式，指向下一个要执行的子任务的索引
        self.current_subtask_index: int = 0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.task_id: str = task_id if task_id is not None else uuid.uuid4().hex
        self.max_retries: int = max_retries # 任务整体失败时的最大重试次数
        self.retry_count: int = 0 # 当前任务整体重试次数

        # --- 新增：子任务执行状态 ---
        self.current_subtask_retry_count: int = 0 # 当前子任务已重试次数 (运行时状态)
        # --- 新增：循环相关运行时状态 (不需要保存) ---
        self.loop_counters: Dict[int, int] = {} # {loop_start_index: current_iteration}
        self.loop_stack: List[int] = []          # 存储当前活动的 loop_start 的索引
        # --- 新增结束 ---

        self.task_stage: str = "PENDING"

    def start(self, device: Device) -> None:
        """标记任务开始运行（由调度器调用）。"""
        self.status = TaskStatus.RUNNING
        self.task_stage = "PREPARING"
        self.assigned_device = device
        self.assigned_device_name = device.name
        if not self.start_time:
             self.start_time = datetime.now()
        self.current_step = 0
        self.current_subtask_index = 0
        # --- 重置运行时状态 ---
        self.current_subtask_retry_count = 0
        self.loop_counters = {}
        self.loop_stack = []
        # --- 重置结束 ---
        # self.retry_count = 0 # 任务级别的重试计数只在需要时增加
        logger.info(f"任务 '{self.name}' (ID: {self.task_id}) 分配给设备 '{device.name}'，进入准备阶段。")

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """标记任务完成或失败（由调度器在处理完步骤结果后调用）。"""
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.task_stage = "COMPLETED" if success else "FAILED" # 更新阶段
        self.end_time = datetime.now()
        self.error = error
        log_level = logging.INFO if success else logging.ERROR
        logger.log(log_level, f"任务 '{self.name}' (ID: {self.task_id}) 标记为 {self.status.value}。成功: {success}, 错误: {error}")

    def cancel(self) -> None:
        """取消任务。"""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.CANCELED
            self.task_stage = "CANCELED"
            logger.info(f"任务 '{self.name}' (ID: {self.task_id}) 在等待时被取消。")
        elif self.status == TaskStatus.RUNNING:
            # 标记为取消，调度器需要在循环中检查这个状态
            self.status = TaskStatus.CANCELED
            self.task_stage = "CANCELED"
            self.error = "手动取消"
            logger.info(f"任务 '{self.name}' (ID: {self.task_id}) 在运行时被标记为取消。")
        else:
            logger.warning(f"无法取消任务 '{self.name}'，当前状态: {self.status.value}")

    def get_runtime(self) -> str:
        if not self.start_time:
            return "N/A"
        end = self.end_time or datetime.now()
        # 如果任务仍在运行，显示当前运行时长
        if self.status == TaskStatus.RUNNING or self.status == TaskStatus.PAUSED: # 假设有 PAUSED 状态
            duration = datetime.now() - self.start_time
        elif self.end_time:
             duration = self.end_time - self.start_time
        else: # 其他状态（如 PENDING, CANCELED）且未开始
             return "0s" if self.status != TaskStatus.PENDING else "N/A"

        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes}m{seconds}s"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"


    def get_progress_display(self) -> str:
        """返回表示当前进度的字符串，包含重试信息。"""
        base_progress = self.status.value # 默认为状态
        subtask_retry_info = "" # 子任务重试信息
        if self.status == TaskStatus.RUNNING:
            # 【新增】如果当前子任务正在重试，添加提示
            if not self.use_ai_driver and self.current_subtask_retry_count > 0:
                 subtask_retry_info = f" (子任务重试 {self.current_subtask_retry_count}/{CONFIG.get('SUBTASK_RETRY_COUNT', 1)})"
            # 【结束新增】

            if self.task_stage == "PREPARING":
                base_progress = "准备环境"
            elif self.task_stage == "WAITING":
                 if self.assigned_device and self.assigned_device.status == DeviceStatus.WAITING:
                      base_progress = self.assigned_device.task_progress
                 else:
                      base_progress = "等待中..."
            elif self.task_stage == "RUNNING":
                if self.use_ai_driver:
                    base_progress = f"AI 步骤 {self.current_step + 1}"
                else:
                    total_subtasks = len(self.subtasks)
                    if total_subtasks > 0 and 0 <= self.current_subtask_index < total_subtasks:
                        current_index = self.current_subtask_index
                        subtask_desc = self.subtasks[current_index].get('description', self.subtasks[current_index].get('type', 'N/A'))
                        base_progress = f"子任务 {current_index + 1}/{total_subtasks}: {subtask_desc[:30]}"
                    elif total_subtasks > 0: # 索引超出范围，可能已完成或出错
                        base_progress = f"子任务 {total_subtasks+1}/{total_subtasks}" # 显示超出的索引
                    else:
                        base_progress = "运行中 (无子任务)"

                base_progress += subtask_retry_info # 添加子任务重试信息

        # 添加任务级别的重试信息
        if self.retry_count > 0 and self.status != TaskStatus.COMPLETED:
            if self.status == TaskStatus.PENDING:
                 return f"等待重试 ({self.retry_count}/{self.max_retries})"
            else: # RUNNING 状态下的重试
                 return f"{base_progress} (任务重试 {self.retry_count}/{self.max_retries})"
        elif self.status == TaskStatus.FAILED and self.max_retries > 0 and self.retry_count >= self.max_retries:
             return f"失败 (任务重试 {self.retry_count}/{self.max_retries} 次)"

        return base_progress


    def to_dict(self) -> Dict[str, Any]:
        """将 Task 对象转换为字典，用于保存。"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "type": self.type.value,
            "app_name": self.app_name,
            "priority": self.priority,
            "use_ai_driver": self.use_ai_driver,
            "subtasks": self.subtasks,
            "status": self.status.value,
            # "task_stage": self.task_stage, # 保存阶段信息
            "assigned_device_name": self.assigned_device_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_step": self.current_step,
            "current_subtask_index": self.current_subtask_index,
            "error": self.error,
            # "runtime": self.get_runtime(), # runtime 可以动态计算，不一定需要保存
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典（例如从 JSON 加载）创建 Task 对象。"""
        task = cls(
            name=data.get("name", "Unnamed Task"),
            task_type=TaskType(data.get("type", TaskType.CUSTOM.value)),
            app_name=data.get("app_name", ""),
            subtasks=data.get("subtasks", []),
            priority=data.get("priority", 0),
            use_ai_driver=data.get("use_ai_driver", True),
            task_id=data.get("task_id"),
            max_retries=data.get("max_retries", 0) # 加载重试次数
        )
        task.status = TaskStatus(data.get("status", TaskStatus.PENDING.value))
        # --- 加载任务阶段，如果不存在则根据状态推断 ---
        task.task_stage = data.get("task_stage")
        if not task.task_stage:
            if task.status == TaskStatus.RUNNING: task.task_stage = "RUNNING" # 粗略恢复
            elif task.status == TaskStatus.COMPLETED: task.task_stage = "COMPLETED"
            elif task.status == TaskStatus.FAILED: task.task_stage = "FAILED"
            elif task.status == TaskStatus.CANCELED: task.task_stage = "CANCELED"
            else: task.task_stage = "PENDING"
        # --- 恢复结束 ---
        task.assigned_device_name = data.get("assigned_device_name")
        task.start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None
        task.end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        task.current_step = data.get("current_step", 0)
        task.current_subtask_index = data.get("current_subtask_index", 0)
        task.error = data.get("error")
        task.retry_count = data.get("retry_count", 0) # 恢复重试计数
        return task

class ScreenshotManager:
    """Handles taking screenshots via ADB."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adb_path: str = config["ADB_PATH"]
        # CAMERA_DEVICE_ID is the *primary* device used for screenshots
        self.primary_device_id: Optional[str] = config.get("CAMERA_DEVICE_ID")
        self.screenshot_counter: int = 0
        self.mutex = QMutex()
        self.connected_devices: Dict[str, bool] = {}  # Track connection status per device ID
        self.last_connect_attempt: Dict[str, float] = {}  # Track last attempt time per device
        self.last_captured_image: Optional[np.ndarray] = None  # Cache last taken screenshot

        # Ensure screenshots directory exists
        os.makedirs("screenshots", exist_ok=True)

    def _get_adb_device_list(self) -> List[str]:
        """Gets a list of currently connected ADB devices."""
        try:
            cmd = [self.adb_path, "devices"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False, encoding='utf-8')
            if result.returncode != 0:
                logger.error(f"ADB 'devices' command failed: {result.stderr}")
                return []

            lines = result.stdout.strip().splitlines()[1:]  # Skip header
            devices = []
            for line in lines:
                if '\t' in line:
                    device_id, status = line.split('\t', 1)
                    if status.strip() == "device":
                        devices.append(device_id.strip())
            return devices
        except subprocess.TimeoutExpired:
            logger.error("ADB 'devices' command timed out.")
            return []
        except FileNotFoundError:
            logger.error(f"ADB executable not found at: {self.adb_path}")
            return []
        except Exception as e:
            logger.error(f"Error getting ADB device list: {e}", exc_info=True)
            return []

    def connect_device(self, device_id: Optional[str] = None) -> bool:
        """Connects to a specific ADB device or the primary one."""
        target_device_id = device_id or self.primary_device_id
        if not target_device_id:
            # Auto-detect first available device if no primary is set
            available_devices = self._get_adb_device_list()
            if not available_devices:
                logger.error("ADB Auto-detect failed: No devices found.")
                return False
            target_device_id = available_devices[0]
            self.primary_device_id = target_device_id  # Set the detected one as primary for future use
            logger.info(f"Auto-detected and connected to primary ADB device: {target_device_id}")

        current_time = time.time()
        last_attempt = self.last_connect_attempt.get(target_device_id, 0)

        if current_time - last_attempt < 5:  # Prevent rapid retries
            return self.connected_devices.get(target_device_id, False)

        self.last_connect_attempt[target_device_id] = current_time

        self.mutex.lock()
        try:
            cmd = [self.adb_path, "-s", target_device_id, "get-state"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False, encoding='utf-8')

            if result.returncode == 0 and "device" in result.stdout.strip():
                logger.info(f"Successfully connected to ADB device: {target_device_id}")
                self.connected_devices[target_device_id] = True
                return True
            else:
                logger.error(
                    f"Failed to connect to ADB device '{target_device_id}'. State: {result.stdout.strip()}, Error: {result.stderr.strip()}")
                self.connected_devices[target_device_id] = False
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"ADB 'get-state' command timed out for device: {target_device_id}")
            self.connected_devices[target_device_id] = False
            return False
        except FileNotFoundError:
            logger.error(f"ADB executable not found at: {self.adb_path}")
            self.connected_devices[target_device_id] = False
            return False
        except Exception as e:
            logger.error(f"ADB connection error for device '{target_device_id}': {e}", exc_info=True)
            self.connected_devices[target_device_id] = False
            return False
        finally:
            self.mutex.unlock()

    def take_screenshot(self, device_obj: Optional[Device] = None) -> Optional[np.ndarray]:
        """Takes a screenshot using the specified device object or the primary camera device."""
        target_device_id = self.primary_device_id  # Default to primary camera device
        # Note: This assumes the screenshot source *is* the primary camera device ID.
        # If different devices need screenshots (not via ADB), this logic needs adjustment.

        if not target_device_id:
            logger.error("Screenshot failed: No primary ADB device ID configured or detected.")
            return None

        # Use device-specific resolution if available
        screenshot_res = CONFIG["SCREENSHOT_RESOLUTION"]
        crop_res = CONFIG["CROPPED_RESOLUTION"]
        if device_obj:
            screenshot_res = device_obj.get_config("SCREENSHOT_RESOLUTION", screenshot_res)
            crop_res = device_obj.get_config("CROPPED_RESOLUTION", crop_res)

        if not self.connected_devices.get(target_device_id) and not self.connect_device(target_device_id):
            logger.error(f"Screenshot failed: ADB device '{target_device_id}' not connected.")
            self.last_captured_image = None
            return None

        self.mutex.lock()
        try:
            cmd = [self.adb_path, "-s", target_device_id, "shell", "screencap", "-p"]
            adb_timeout = self.config.get("ADB_COMMAND_TIMEOUT", 20)

            try:
                # Use PIPE for stdout/stderr to avoid potential deadlocks with large output
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=adb_timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                logger.error(f"ADB screenshot command timed out ({adb_timeout}s) for device '{target_device_id}'")
                process.kill()  # Ensure the process is terminated
                stdout, stderr = process.communicate()  # Get any remaining output
                self.connected_devices[target_device_id] = False
                self.last_captured_image = None
                return None
            except FileNotFoundError:
                logger.error(f"ADB executable not found at: {self.adb_path}")
                self.connected_devices[target_device_id] = False
                self.last_captured_image = None
                return None

            if returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore').strip()
                logger.error(
                    f"Screenshot failed for device '{target_device_id}' (Return Code: {returncode}): {error_msg}")
                if "device offline" in error_msg or "device not found" in error_msg:
                    self.connected_devices[target_device_id] = False
                self.last_captured_image = None
                return None

            # Process the image data (handle potential CRLF issues)
            image_bytes = stdout.replace(b'\r\n', b'\n')
            if not image_bytes:
                logger.error(f"Screenshot failed for device '{target_device_id}': Received empty image data.")
                self.last_captured_image = None
                return None

            try:
                image = Image.open(io.BytesIO(image_bytes))
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as img_err:
                logger.error(f"Error processing screenshot image data: {img_err}", exc_info=True)
                self.last_captured_image = None
                return None

            # Crop image (using potentially device-specific resolution)
            h, w = opencv_image.shape[:2]
            target_w, target_h = crop_res  # Target cropped size

            if h > target_h:
                y_start = (h - target_h) // 2
                y_end = y_start + target_h
            else:
                y_start, y_end = 0, h

            if w > target_w:
                x_start = (w - target_w) // 2
                x_end = x_start + target_w
            else:
                x_start, x_end = 0, w

            cropped_image = opencv_image[y_start:y_end, x_start:x_end]

            # Save a copy for debugging
            self.screenshot_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            screenshot_filename = f"screenshot_{target_device_id}_{timestamp}_{self.screenshot_counter}.jpg"
            screenshot_path = os.path.join("screenshots", screenshot_filename)

            try:
                cv2.imwrite(screenshot_path, cropped_image)
                logger.debug(f"Screenshot saved: {screenshot_path}")
            except Exception as save_err:
                logger.warning(f"Failed to save screenshot file '{screenshot_path}': {save_err}")

            self.last_captured_image = cropped_image
            return cropped_image

        except Exception as e:
            logger.error(f"Screenshot processing error: {e}", exc_info=True)
            self.connected_devices[target_device_id] = False
            self.last_captured_image = None
            return None
        finally:
            self.mutex.unlock()

    def enhance_image(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Enhances image quality for potentially better OCR."""
        if image is None:
            return None
        try:
            # Convert to LAB color space for CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Reduced clipLimit slightly
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Optional: Mild sharpening (use carefully, can increase noise)
            # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # sharpened = cv2.filter2D(enhanced_image, -1, kernel)
            # return sharpened

            return enhanced_image
        except Exception as e:
            logger.error(f"Image enhancement error: {e}", exc_info=True)
            return image  # Return original on error


class AIAnalyzer:
    """处理 OCR、AI 决策和模板匹配。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_api_url: str = config["OCR_API_URL"]
        self.ai_api_url: str = config["AI_API_URL"]
        self.ai_api_key: str = config["AI_API_KEY"]
        self.ai_model: str = config["AI_MODEL"]
        self.max_context_tokens: int = 3500
        self.ocr_mutex = QMutex()
        self.ai_mutex = QMutex()
        self.decision_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.cache_expire_time: int = 60

        self.templates: Dict[str, np.ndarray] = {}
        self.template_dir: str = "templates"
        self.load_templates()  # 初始化时加载模板

    def load_templates(self) -> None:
        """从模板目录加载图像模板。"""
        try:
            os.makedirs(self.template_dir, exist_ok=True)  # 确保目录存在
            count = 0
            for file in os.listdir(self.template_dir):
                if file.lower().endswith((".jpg", ".png")):
                    template_name = os.path.splitext(file)[0]  # 文件名作为模板名
                    template_path = os.path.join(self.template_dir, file)
                    try:
                        # 以彩色模式加载，确保不是 None
                        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                        if template_img is not None:
                            self.templates[template_name] = template_img
                            count += 1
                            logger.debug(f"Loaded template: {template_name}")  # 调试信息
                        else:
                            logger.warning(f"无法加载模板图像: {template_path}")
                    except Exception as load_err:
                        logger.warning(f"加载模板 '{template_path}' 时出错: {load_err}")

            logger.info(f"从 '{self.template_dir}' 加载了 {len(self.templates)} 个模板")  # 修正日志输出
        except Exception as e:
            logger.error(f"加载模板目录 '{self.template_dir}' 时出错: {e}", exc_info=True)

    def perform_ocr(self, image: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """使用配置的 API 执行 OCR。"""
        if image is None:
            logger.error("OCR 失败：输入图像为 None。")
            return None

        if not self.ocr_api_url:  # 检查 URL 是否已配置
            logger.error("OCR 失败：OCR API URL 未配置。")
            return None

        self.ocr_mutex.lock()  # 获取互斥锁，保证线程安全
        try:
            # 将 BGR (OpenCV) 转换为 RGB (PIL)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # 将 PIL 图像保存到内存中的 BytesIO 对象
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=90)  # 使用 JPEG 格式以减小大小
            # 进行 Base64 编码
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 构造请求体 (payload)
            payload = {
                "base64": base64_image,
                "options": {
                    "data.format": "dict",
                    "ocr.angle": True,
                }
            }

            headers = {"Content-Type": "application/json"}
            ocr_timeout = self.config.get("OCR_TIMEOUT", 15)

            logger.debug(f"向 OCR API ({self.ocr_api_url}) 发送请求...")
            try:
                response = requests.post(self.ocr_api_url, json=payload, headers=headers, timeout=ocr_timeout)
                response.raise_for_status()
                result = response.json()

                if result.get("code") in [100, 101] and "data" in result:
                    if result["code"] == 100:
                        logger.info(f"OCR 成功，找到 {len(result['data'])} 个文本框。")
                        result["data"] = self._filter_ocr_results(result["data"])
                    else:
                        logger.info("OCR 成功完成，但未在图像中找到文本。")
                        result["data"] = []
                    return result
                else:
                    error_message = result.get('data', '无详细错误信息') if isinstance(result.get('data'),
                                                                                       str) else '非预期的错误数据结构'
                    logger.error(
                        f"OCR API 请求失败或返回非预期数据。状态码: {result.get('code', 'N/A')}, 消息: {error_message}")
                    return {"code": result.get('code', -1), "data": error_message}

            except requests.exceptions.Timeout:
                logger.error(f"OCR 请求超时（{ocr_timeout}秒）。")
                return {"code": -1, "data": "请求超时"}
            except requests.exceptions.RequestException as req_err:
                logger.error(f"OCR 请求错误: {req_err}", exc_info=True)
                return {"code": -1, "data": f"网络或请求错误: {req_err}"}

        except Exception as e:
            logger.error(f"OCR 处理错误: {e}", exc_info=True)
            return {"code": -1, "data": f"内部处理错误: {e}"}
        finally:
            self.ocr_mutex.unlock()

    def _filter_ocr_results(self, ocr_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据置信度过滤 OCR 结果。"""
        min_confidence = self.config.get("OCR_CONFIDENCE_THRESHOLD", 0.6)
        filtered_data = [
            # 修改：使用 "score" 键进行过滤，并提供默认值以防万一
            item for item in ocr_data
            if item.get("score", 0.0) >= min_confidence  # 使用 "score"，若缺失则默认为 0.0
        ]
        removed_count = len(ocr_data) - len(filtered_data)
        if removed_count > 0:
            logger.debug(f"过滤掉 {removed_count} 个低于置信度阈值 {min_confidence} 的 OCR 结果 (基于 score)")
        return filtered_data

    def _estimate_token_count(self, text: str) -> int:
        """Roughly estimates token count (simple space-based split)."""
        return len(text.split()) + text.count('\n')

    def get_ai_decision(self, prompt_text: str, history_actions: List[Dict[str, Any]], current_task_info: str) -> \
    Optional[Dict[str, Any]]:
        """Gets a decision from the AI API."""
        if not self.ai_api_key:
            logger.error("AI decision failed: API key is not configured.")
            return None

        self.ai_mutex.lock()
        try:
            # --- Context/Token Management ---
            prompt_tokens = self._estimate_token_count(prompt_text)
            history_text = json.dumps(history_actions, ensure_ascii=False)
            history_tokens = self._estimate_token_count(history_text)

            while prompt_tokens + history_tokens > self.max_context_tokens and history_actions:
                history_actions = history_actions[1:]  # Remove oldest action
                history_text = json.dumps(history_actions, ensure_ascii=False)
                history_tokens = self._estimate_token_count(history_text)
                logger.warning("Truncating AI history due to context length limit.")

            final_prompt = prompt_text

            # --- Cache Check ---
            cache_key = f"{self.ai_model}:{hash(final_prompt)}"
            if cache_key in self.decision_cache:
                cache_time, cached_result = self.decision_cache[cache_key]
                if time.time() - cache_time < self.cache_expire_time:
                    logger.debug(f"Using cached AI decision for task: {current_task_info}")
                    # 返回缓存结果时，也记录一下完整的缓存内容（如果需要）
                    # ai_decision_logger.info(f"使用缓存决策: {json.dumps(cached_result, indent=2, ensure_ascii=False)}")
                    return cached_result

            # --- API Request ---
            system_prompt = """You are a precise and efficient smartphone automation assistant. Analyze the provided screen information (text, elements) and recent action history. Based on the current task goal, determine the single best next action to perform using ONLY the available functions. Provide the function call and a brief justification. If unsure or stuck, suggest 'analyze_screen()' or 'back()'. Response format: FUNCTION_CALL\nJustification: [Your reason]"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ]

            payload = {
                "model": self.ai_model,
                "messages": messages,
                "temperature": 0.2,
                "stop": ["\nJustification:"]  # 保持这个 stop token
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.ai_api_key}"
            }
            ai_timeout = self.config.get("AI_TIMEOUT", 30)

            logger.info(
                f"Sending request to AI API ({self.ai_model}) for task: {current_task_info}. Estimated prompt tokens: ~{prompt_tokens + history_tokens}")

            try:
                response = requests.post(self.ai_api_url, json=payload, headers=headers, timeout=ai_timeout)
                response.raise_for_status()  # 检查 HTTP 错误
                result = response.json()  # 解析 JSON 响应

                # --- 新增：记录完整的原始 AI 响应 ---
                # 使用 json.dumps 美化输出，ensure_ascii=False 保留中文
                try:
                    raw_response_str = json.dumps(result, indent=2, ensure_ascii=False)
                except Exception as json_err:
                    raw_response_str = f"无法序列化为JSON: {result} (错误: {json_err})"  # 异常处理
                ai_decision_logger.info(f"--- 完整 AI 响应 ---\n{raw_response_str}\n--------------------")
                # --- 新增结束 ---

                # --- 原有的响应解析逻辑 ---
                if result.get("choices") and result["choices"][0].get("message"):
                    ai_content = result["choices"][0]["message"].get("content", "").strip()
                    justification = ""
                    # 尝试解析决策和理由（即使 stop token 可能没起作用）
                    if "\nJustification:" in ai_content:
                        parts = ai_content.split("\nJustification:", 1)
                        ai_decision = parts[0].strip()
                        if len(parts) > 1:
                            justification = parts[1].strip()
                    else:
                        ai_decision = ai_content  # 如果没有 Justification，整个内容作为决策

                    decision_result = {
                        "decision": ai_decision,  # 解析出的决策部分
                        "justification": justification,  # 解析出的理由部分
                        "raw_response": result  # 仍然保留原始响应在返回值中（虽然主要通过日志查看）
                    }
                    self.decision_cache[cache_key] = (time.time(), decision_result)
                    # logger.info(f"AI decision received: {ai_decision} (Justification: {justification})") # 这行日志现在有点重复，可以考虑注释掉或保留
                    return decision_result
                else:
                    logger.error(f"AI API response format unexpected: {result}")
                    # 即使格式不符合预期，也要返回原始响应（如果存在）以供调试
                    return {"decision": None, "justification": None, "raw_response": result,
                            "error": "Unexpected response format"}

            except requests.exceptions.Timeout:
                logger.error(f"AI API request timed out after {ai_timeout} seconds.")
                return None  # 或者返回带错误信息的字典
            except requests.exceptions.RequestException as req_err:
                logger.error(f"AI API request error: {req_err}", exc_info=True)
                error_details = None
                try:
                    # 尝试获取服务器返回的错误详情
                    error_details = response.json()
                    logger.error(f"AI API Error Details: {error_details}")
                except:
                    pass  # 如果无法解析错误详情，忽略
                # 返回包含错误信息的字典，可能包含原始响应（如果部分接收）
                return {"decision": None, "justification": None, "raw_response": error_details, "error": str(req_err)}

        except Exception as e:
            logger.error(f"AI decision processing error: {e}", exc_info=True)
            return None  # 或者返回带错误信息的字典
        finally:
            self.ai_mutex.unlock()

    def template_matching(self,
                          image: Optional[np.ndarray],
                          template_name: Optional[str] = None,
                          template_image: Optional[np.ndarray] = None,
                          threshold: Optional[float] = None
                          ) -> Dict[str, Any]:
        """Performs template matching on the image."""
        if image is None:
            return {"match": False, "error": "Input image is None"}
        if template_name is None and template_image is None:
            return {"match": False, "error": "No template name or image provided"}

        match_threshold = threshold if threshold is not None else self.config.get("TEMPLATE_MATCHING_THRESHOLD", 0.75)

        try:
            template: Optional[np.ndarray] = None
            if template_image is not None:
                template = template_image
                if template_name is None: template_name = "Unnamed Template"
            elif template_name:
                template = self.templates.get(template_name)
                if template is None:
                    return {"match": False, "error": f"Template '{template_name}' not found in loaded templates."}

            if template is None:
                return {"match": False, "error": "Template image is invalid."}

            img_h, img_w = image.shape[:2]
            tpl_h, tpl_w = template.shape[:2]
            if tpl_h > img_h or tpl_w > img_w:
                logger.warning(
                    f"Template '{template_name}' ({tpl_w}x{tpl_h}) is larger than image ({img_w}x{img_h}). Skipping match.")
                return {"match": False, "error": "Template larger than image"}

            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= match_threshold:
                center_x = max_loc[0] + tpl_w // 2
                center_y = max_loc[1] + tpl_h // 2
                logger.debug(
                    f"Template '{template_name}' matched with confidence {max_val:.3f} at {max_loc}, center ({center_x}, {center_y})")
                return {
                    "match": True,
                    "confidence": float(max_val),
                    "name": template_name,
                    "center": (center_x, center_y),
                    "top_left": max_loc,
                    "bottom_right": (max_loc[0] + tpl_w, max_loc[1] + tpl_h),
                    "width": tpl_w,
                    "height": tpl_h
                }
            else:
                return {"match": False, "confidence": float(max_val)}
        except Exception as e:
            logger.error(f"Template matching error for '{template_name}': {e}", exc_info=True)
            return {"match": False, "error": str(e)}

    def match_all_templates(self, image: Optional[np.ndarray], threshold: Optional[float] = None) -> List[
        Dict[str, Any]]:
        """Matches the image against all loaded templates."""
        if image is None or not self.templates:
            return []

        results = []
        for name in self.templates.keys():
            match_result = self.template_matching(image, template_name=name, threshold=threshold)
            if match_result["match"]:
                results.append(match_result)

        return results

    # --- Template Management ---
    def add_template(self, name: str, image: np.ndarray) -> bool:
        """在内存中添加或更新模板，并将其保存到磁盘。"""
        if not name or image is None:
            logger.error("无法添加模板：名称或图像无效。")
            return False
        try:
            # 确保存储的是 BGR 格式 (OpenCV 默认)
            if len(image.shape) == 2:  # 如果是灰度图，转为 BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # 如果是 BGRA，转为 BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            self.templates[name] = image  # 存储 BGR 图像
            save_path = os.path.join(self.template_dir, f"{name}.png")  # 统一保存为 png
            # 使用 imwrite 保存 BGR 图像
            success = cv2.imwrite(save_path, image)
            if success:
                logger.info(f"模板 '{name}' 已添加/更新并保存到 {save_path}")
                return True
            else:
                logger.error(f"使用 cv2.imwrite 保存模板 '{name}' 到 {save_path} 失败。")
                # 尝试从内存中移除失败的模板
                if name in self.templates:
                    del self.templates[name]
                return False
        except Exception as e:
            logger.error(f"添加/保存模板 '{name}' 时出错: {e}", exc_info=True)
            # 尝试从内存中移除失败的模板
            if name in self.templates:
                del self.templates[name]
            return False

    def remove_template(self, name: str) -> bool:
        """Removes a template from memory and deletes its file."""
        if name not in self.templates:
            logger.warning(f"Template '{name}' not found, cannot remove.")
            return False
        try:
            del self.templates[name]
            for ext in [".png", ".jpg"]:
                template_path = os.path.join(self.template_dir, f"{name}{ext}")
                if os.path.exists(template_path):
                    os.remove(template_path)
                    logger.info(f"Removed template file: {template_path}")
            logger.info(f"Template '{name}' removed from memory.")
            return True
        except Exception as e:
            logger.error(f"Error removing template '{name}': {e}", exc_info=True)
            return False


class ESPController:
    """Handles communication with the ESP-12F based physical controller."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.esp_ip: str = config["ESP_IP"]
        self.esp_port: int = config["ESP_PORT"]
        self.connected: bool = False
        self.socket: Optional[socket.socket] = None
        self.command_queue: queue.Queue = queue.Queue()
        self.response_queue: queue.Queue = queue.Queue()  # 用于存放最终结果
        self.stop_event = threading.Event()
        self.comm_thread: Optional[threading.Thread] = None
        self.mutex = QMutex()  # 用于保护 socket 和 connected 状态
        self.command_lock = threading.Lock()  # 用于控制命令发送间隔
        self.last_command_time: float = 0
        self.min_interval: float = config.get("MIN_ESP_COMMAND_INTERVAL", 0.05)
        # --- 新增：连续超时计数器和阈值 ---
        self.max_consecutive_timeouts: int = config.get("ESP_MAX_CONSECUTIVE_TIMEOUTS", 5) # 增加默认值
        self.consecutive_timeouts: int = 0
        # --- 新增结束 ---
        logger.info(f"ESPController 初始化，最小命令间隔: {self.min_interval}s, 最大连续超时: {self.max_consecutive_timeouts}")


    def connect(self) -> bool:
        """Connects to the ESP device with retry logic."""
        with QMutexLocker(self.mutex):  # 使用 QMutexLocker 简化锁管理
            if self.connected:
                return True

            try:
                max_retries = 3
                base_retry_delay = 1

                for attempt in range(max_retries):
                    logger.info(
                        f"尝试连接 ESP ({self.esp_ip}:{self.esp_port}) - 第 {attempt + 1}/{max_retries} 次尝试")
                    try:
                        # --- 创建新的 Socket ---
                        if self.socket: # 如果旧 socket 存在，先关闭
                            try: self.socket.close()
                            except: pass
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.settimeout(5) # 设置连接超时
                        self.socket.connect((self.esp_ip, self.esp_port))
                        # 连接成功后，设置一个默认的读取超时，防止 recv 卡死
                        self.socket.settimeout(self.config.get("COMMAND_TIMEOUT", 15) + 5)  # 比命令超时稍长
                        self.connected = True
                        self.stop_event.clear()
                        self.consecutive_timeouts = 0 # 连接成功，重置超时计数

                        # 启动通信线程 (如果不存在或已停止)
                        if self.comm_thread is None or not self.comm_thread.is_alive():
                            self.comm_thread = threading.Thread(target=self._communication_thread, daemon=True,
                                                                name="ESPCommThread")
                            self.comm_thread.start()
                            logger.info("ESP 通信线程已启动。")
                        else:
                            logger.info("ESP 通信线程已在运行。")

                        logger.info(f"成功连接到 ESP ({self.esp_ip}:{self.esp_port})")
                        return True
                    except socket.timeout:
                        logger.warning(f"ESP 连接超时 (尝试 {attempt + 1})")
                    except OSError as e:
                        logger.warning(f"ESP 连接 OS 错误 (尝试 {attempt + 1}): {e}")
                    except Exception as e:
                        logger.warning(f"ESP 连接失败 (尝试 {attempt + 1}): {e}", exc_info=False)

                    # --- 连接失败后的清理 ---
                    if self.socket:
                        try: self.socket.close()
                        except: pass
                        self.socket = None
                    self.connected = False

                    if attempt < max_retries - 1:
                        retry_delay = base_retry_delay * (2 ** attempt)
                        logger.info(f"将在 {retry_delay} 秒后重试 ESP 连接...")
                        time.sleep(retry_delay)
                    else:
                        logger.error("ESP 连接失败，已达最大重试次数。")
                        return False
                return False # 理论上不会到达这里
            except Exception as e:
                logger.error(f"ESP 连接过程中发生意外错误: {e}", exc_info=True)
                if self.socket:
                    try: self.socket.close()
                    except: pass
                self.socket = None
                self.connected = False
                return False

    def disconnect(self) -> None:
        """Disconnects from the ESP device."""
        with QMutexLocker(self.mutex):
            if not self.connected:
                logger.debug("ESP 已断开连接。")
                return

            logger.info("正在断开与 ESP 的连接...")
            self.stop_event.set()  # 通知通信线程停止

            # 清理命令队列，防止线程阻塞在 get()
            while not self.command_queue.empty():
                try: self.command_queue.get_nowait(); self.command_queue.task_done()
                except queue.Empty: break
            # 清理响应队列 (虽然理论上线程停止后不会再放东西)
            while not self.response_queue.empty():
                try: self.response_queue.get_nowait(); self.response_queue.task_done()
                except queue.Empty: break

            comm_thread = self.comm_thread  # 获取线程引用

        # 在锁外部等待线程结束
        if comm_thread and comm_thread.is_alive():
            logger.debug("等待 ESP 通信线程结束...")
            comm_thread.join(timeout=2.0)  # 等待线程结束
            if comm_thread.is_alive():
                logger.warning("ESP 通信线程未能优雅停止。")

        # 再次获取锁进行最终清理
        with QMutexLocker(self.mutex):
            if self.socket:
                try:
                    # 关闭socket之前尝试shutdown，更优雅地关闭连接
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    # 忽略特定错误 (例如 "not connected")
                    if e.errno != 10057 and e.errno != socket.ENOTCONN: # Win / Linux
                         logger.warning(f"关闭 ESP socket 时发生 shutdown 错误: {e}")
                except Exception as e:
                    logger.warning(f"关闭 ESP socket 时发生意外错误: {e}")
                finally:
                    try: self.socket.close()
                    except: pass # 忽略关闭错误
                    self.socket = None

            self.connected = False
            self.comm_thread = None  # 清理线程引用
            logger.info("ESP 连接已断开。")

    def _communication_thread(self) -> None:
        """
        【已修改】在单独的线程中处理发送命令和接收响应。
        专门等待以 'OK' 或 'ERROR' 开头的行作为最终响应。
        更健壮地处理连接错误，并优化响应读取以减少延迟。
        【新增】增加了更详细的接收和处理日志，便于诊断超时问题。
        """
        logger.info("ESP 通信线程启动。")
        read_buffer = b""  # 字节缓冲区用于处理不完整的行

        while not self.stop_event.is_set():
            command_to_process = None
            response_received = False  # 标记当前命令是否已收到最终响应
            try:
                # 1. 尝试获取命令 (短时阻塞)
                try:
                    command_to_process = self.command_queue.get(block=True, timeout=0.1)
                    logger.debug(f"ESP 通信线程: 从队列获取到命令 '{command_to_process}'。")
                except queue.Empty:
                    time.sleep(0.05)
                    continue
                except Exception as get_q_err:
                    logger.error(f"ESP 通信线程: 从队列获取命令时出错: {get_q_err}", exc_info=True)
                    time.sleep(0.5)
                    continue

                # 2. 如果获取到命令，处理命令
                if command_to_process:
                    # --- 发送命令 (加锁保护 socket 和发送间隔) ---
                    send_success = False
                    try:
                        with QMutexLocker(self.mutex):
                            if not self.connected or not self.socket:
                                logger.warning(f"ESP 通信线程: 未连接，无法发送 '{command_to_process}'。正在放入错误响应。")
                                try: self.response_queue.put("Error: Not Connected")
                                except queue.Full: pass
                                response_received = True
                                continue

                            with self.command_lock:
                                current_time = time.time()
                                time_since_last = current_time - self.last_command_time
                                if time_since_last < self.min_interval:
                                    sleep_duration = self.min_interval - time_since_last
                                    logger.debug(f"ESP 通信线程: 等待 {sleep_duration:.3f} 秒以满足最小间隔...")
                                    time.sleep(sleep_duration)
                                full_command = (command_to_process + "\n").encode('utf-8')
                                logger.debug(f"ESP 通信线程: 尝试发送: {full_command!r}")
                                self.socket.sendall(full_command)
                                self.last_command_time = time.time()
                                logger.info(f"ESP 通信线程: 成功发送: {command_to_process}")
                                send_success = True

                    except socket.timeout:
                        logger.error(f"ESP 通信线程: 发送命令 '{command_to_process}' 时超时。正在断开连接。")
                        self._handle_connection_error()
                        try: self.response_queue.put("Error: Send Timeout")
                        except queue.Full: pass
                        response_received = True
                    except OSError as e:
                        logger.error(f"ESP 通信线程: 发送 '{command_to_process}' 时发生 OSError: {e}。正在断开连接。")
                        self._handle_connection_error()
                        try: self.response_queue.put(f"Error: Send Failed ({e})")
                        except queue.Full: pass
                        response_received = True
                    except Exception as send_err:
                        logger.error(f"ESP 通信线程: 发送 '{command_to_process}' 时发生意外错误: {send_err}", exc_info=True)
                        self._handle_connection_error()
                        try: self.response_queue.put(f"Error: Send Exception ({send_err})")
                        except queue.Full: pass
                        response_received = True

                    if not send_success:
                        continue # 进入 finally 处理 task_done

                    # --- 等待最终响应 (在锁外部进行主要循环) ---
                    final_response = None
                    response_start_time = time.time()
                    response_timeout = self.config.get("COMMAND_TIMEOUT", 15)
                    logger.debug(f"ESP 通信线程: 正在等待 '{command_to_process}' 的最终响应 (超时={response_timeout}秒)...")

                    while final_response is None and not self.stop_event.is_set():
                        if time.time() - response_start_time > response_timeout:
                            logger.warning(f"ESP 通信线程: 命令 '{command_to_process}' 最终响应超时 ({response_timeout}秒)。连续超时次数: {self.consecutive_timeouts + 1}")
                            try: self.response_queue.put("Error: Timeout")
                            except queue.Full: pass
                            response_received = True
                            self.consecutive_timeouts += 1
                            if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                                logger.error(f"ESP 通信线程: 达到最大连续 ESP 超时次数 ({self.max_consecutive_timeouts})。断开连接。")
                                self._handle_connection_error()
                            break

                        chunk = b""
                        read_error = False
                        is_connected_before_read = False

                        with QMutexLocker(self.mutex):
                            is_connected_before_read = self.connected and self.socket is not None

                        if not is_connected_before_read:
                            logger.warning("ESP 通信线程: 等待响应时检测到连接已断开。")
                            response_received = True
                            break

                        try:
                            with QMutexLocker(self.mutex):
                                if not self.connected or not self.socket:
                                    raise socket.error("Socket closed between check and read")
                                self.socket.settimeout(0.1)
                                # 【日志】记录 recv 之前的 buffer
                                logger.debug(f"ESP 通信线程: Recv前 buffer: {read_buffer!r}")
                                chunk = self.socket.recv(1024)
                                # 【日志】记录 recv 返回的 chunk
                                logger.debug(f"ESP 通信线程: Recv 返回 chunk (len={len(chunk)}): {chunk!r}")

                            if not chunk:
                                logger.error(f"ESP 通信线程: 等待命令 '{command_to_process}' 响应时连接被对端关闭。正在断开连接。")
                                self._handle_connection_error()
                                response_received = True
                                break

                            if chunk:
                                read_buffer += chunk
                                # 【日志】记录 recv 之后的 buffer
                                logger.debug(f"ESP 通信线程: Recv后 buffer: {read_buffer!r}")
                                self.consecutive_timeouts = 0

                        except socket.timeout:
                            # logger.debug("ESP 通信线程: Recv 超时 (0.1秒)，正常。") # 这个日志可以保持注释，除非需要极详细的调试
                            pass
                        except socket.error as e:
                            logger.error(f"ESP 通信线程: 读取 '{command_to_process}' 响应时发生 Socket 错误: {e}。正在断开连接。")
                            self._handle_connection_error()
                            response_received = True
                            read_error = True
                            break
                        except Exception as read_err:
                            logger.error(f"ESP 通信线程: 读取 '{command_to_process}' 响应时发生意外错误: {read_err}", exc_info=True)
                            self._handle_connection_error()
                            response_received = True
                            read_error = True
                            break

                        # --- 处理缓冲区中的行 ---
                        # 【日志】处理 buffer 前
                        logger.debug(f"ESP 通信线程: 准备处理 buffer: {read_buffer!r}")
                        processed_line_this_iteration = False # 标记本次循环是否处理了行
                        while b'\n' in read_buffer:
                            processed_line_this_iteration = True # 标记处理过行
                            line_bytes, read_buffer = read_buffer.split(b'\n', 1)
                            try:
                                # 【修改】使用 errors='replace' 替换无法解码的字节，更利于调试
                                line = line_bytes.decode('utf-8', errors='replace').strip()
                                if line:
                                    # 【日志】记录解码后的有效行
                                    logger.info(f"ESP 通信线程: 收到并解码行: '{line}'") # 改为 INFO 级别，更容易看到收到的内容
                                    if line.startswith("OK") or line.startswith("ERROR"):
                                        final_response = line
                                        logger.info(f"ESP 通信线程: 找到命令 '{command_to_process}' 的最终响应: '{final_response}'")
                                        try:
                                            self.response_queue.put(final_response)
                                            logger.debug(f"ESP 通信线程: 已将最终响应 '{final_response}' 放入响应队列。")
                                        except queue.Full:
                                            logger.error(f"ESP 通信线程: 放入最终响应 '{final_response}' 时响应队列已满。")
                                        response_received = True
                                        self.consecutive_timeouts = 0
                                        break # 找到最终响应，退出内层 while b'\n' 循环
                                    # else: # 非最终响应行 (保持 debug 级别)
                                    #     logger.debug(f"ESP 通信线程: 收到非最终响应行: '{line}'")
                            except UnicodeDecodeError as ude:
                                logger.error(f"ESP 通信线程: 响应解码错误: {ude}。原始字节: {line_bytes!r}")
                            except Exception as proc_err:
                                logger.error(f"处理 ESP 行时出错: {proc_err}", exc_info=True)
                        # 【日志】处理 buffer 后
                        if processed_line_this_iteration:
                             logger.debug(f"ESP 通信线程: 处理后剩余 buffer: {read_buffer!r}")
                        # --- 缓冲区处理结束 ---

                        if final_response is not None: break
                        if read_error: break
                        time.sleep(0.02)

                    # --- 循环结束后的处理 ---
                    if not response_received:
                        if self.stop_event.is_set():
                            logger.warning(f"ESP 通信线程: 在等待 '{command_to_process}' 响应时被停止。")
                            try: self.response_queue.put("Error: Interrupted by Stop")
                            except queue.Full: pass
                        else:
                            # 如果是因为超时退出循环，上面的超时逻辑已经 put 了 "Error: Timeout"
                            # 这里是预防其他未知原因退出循环
                            if final_response is None: # 确保不是因为找到响应而退出
                                logger.error(f"ESP 通信线程: 命令 '{command_to_process}' 意外退出响应等待循环且未找到最终响应。假设出错。")
                                try: self.response_queue.put("Error: Unexpected Wait Exit")
                                except queue.Full: pass
                        response_received = True

            except Exception as loop_err:
                logger.critical(f"ESP 通信线程: 主循环发生未处理异常: {loop_err}", exc_info=True)
                self._handle_connection_error()
                if command_to_process and not response_received:
                    try: self.response_queue.put("Error: Communication Loop Error", block=False)
                    except queue.Full: pass
                time.sleep(1) # 异常后稍作等待

            finally:
                # 确保 command_queue 中的任务被标记为完成
                if command_to_process:
                    logger.debug(f"ESP 通信线程: 正在标记命令 '{command_to_process}' 在命令队列中为完成。")
                    try:
                        self.command_queue.task_done()
                    except ValueError: pass # 可能已被标记
                    except Exception as td_err: logger.error(f"ESP 通信线程: 调用 task_done 处理 '{command_to_process}' 时出错: {td_err}", exc_info=True)

        logger.info("ESP 通信线程停止。")
        self._handle_connection_error(log_disconnection=False)

    def _handle_connection_error(self, log_disconnection=True):
        """内部辅助函数，用于处理连接错误和断开状态。"""
        with QMutexLocker(self.mutex):
            if self.connected:
                if log_disconnection:
                    logger.error("ESP 连接错误或超时，正在断开。")
                self.connected = False # 标记为断开
                if self.socket:
                    try: self.socket.close() # 关闭 socket
                    except Exception: pass
                    self.socket = None
                # 不要在这里设置 stop_event，让通信线程自行根据 stop_event 或错误退出
                # self.stop_event.set()
                # 清空响应队列，防止旧的响应干扰后续命令
                while not self.response_queue.empty():
                    try: self.response_queue.get_nowait(); self.response_queue.task_done()
                    except queue.Empty: break
                    except ValueError: pass # task_done 可能已调用


    def send_command(self, command: str, wait_for_response: bool = True, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        发送命令到 ESP 并等待最终的 'OK' 或 'ERROR' 响应。
        更健壮地处理超时和连接问题。
        """
        # (连接检查部分保持不变) ...
        with QMutexLocker(self.mutex):
            if not self.connected:
                logger.warning("ESP 未连接。尝试重新连接...")
                if not self.connect(): # connect 内部会重置 consecutive_timeouts
                    logger.error("ESP 重连失败。无法发送命令。")
                    return {"success": False, "error": "ESP not connected"}
                # 重连成功后，继续执行

        response_timeout = timeout if timeout is not None else self.config.get("COMMAND_TIMEOUT", 15)
        queue_timeout = response_timeout + 5 # 给队列操作和线程调度留多一点时间 (增加缓冲)
        logger.debug(f"ESP send_command: 命令='{command}', 等待响应={wait_for_response}, 队列超时={queue_timeout:.1f}秒") # <<< 新增日志

        try:
            # (线程检查部分保持不变) ...
            if self.comm_thread is None or not self.comm_thread.is_alive():
                 logger.error("ESP 通信线程未运行。无法发送命令。")
                 # 尝试重新连接以重启线程
                 if not self.connect():
                      return {"success": False, "error": "ESP Comm thread dead, reconnect failed"}

            # 将命令放入队列
            logger.debug(f"ESP send_command: 正在将 '{command}' 放入命令队列 (command_queue)...") # <<< 新增日志
            self.command_queue.put(command)
            logger.debug(f"ESP send_command: 命令 '{command}' 已放入队列。") # <<< 新增日志

            if wait_for_response:
                logger.debug(f"ESP send_command: 正在等待 '{command}' 的响应从响应队列 (response_queue) 返回 (超时={queue_timeout}秒)...") # <<< 新增日志
                try:
                    # 从响应队列获取最终结果
                    response: str = self.response_queue.get(timeout=queue_timeout)
                    # !!! 重要：添加 task_done 调用，即使下面解析失败 !!!
                    try:
                        self.response_queue.task_done()
                    except ValueError:
                        logger.warning(f"ESP send_command: 响应队列的 task_done 已经被调用了？") # 可能并发问题？

                    logger.debug(f"ESP send_command: 收到命令 '{command}' 的响应: '{response}'") # <<< 新增日志

                    # (响应解析部分保持不变) ...
                    if response.startswith("OK"):
                        return {"success": True, "response": response}
                    elif response == "Error: Timeout":
                        logger.warning(f"ESP 命令 '{command}' 超时 (由通信线程报告)。")
                        # 超时本身不立即断开连接，由连续超时机制处理
                        return {"success": False, "error": "Command response timeout", "response": response}
                    elif response.startswith("Error:") or response.startswith("ERROR"):
                        logger.warning(f"ESP 命令 '{command}' 失败，错误信息: {response}")
                        # 根据错误类型决定是否处理断开
                        if any(err_str in response for err_str in ["Not Connected", "Send Failed", "Read Failed", "Connection Closed", "Communication Loop Error"]):
                             logger.error(f"检测到严重 ESP 错误 ({response})，处理断开连接。")
                             self._handle_connection_error() # 这些错误表明连接有问题
                        return {"success": False, "error": response, "response": response}
                    else:
                        logger.error(f"收到命令 '{command}' 的意外最终响应: {response}")
                        return {"success": False, "error": f"Unexpected response: {response}", "response": response}

                except queue.Empty:
                    # <<< 修改了超时日志 >>>
                    logger.error(f"ESP send_command: 等待响应队列中 '{command}' 的响应超时 ({queue_timeout}秒)。通信线程可能卡死或已终止。")
                    # 响应队列超时通常意味着通信线程卡住或已死，需要检查连接
                    self._handle_connection_error() # 队列超时，假设连接失败
                    return {"success": False, "error": "Command response timeout (queue empty)"}
                # <<< 新增：捕获 get/task_done 期间的潜在异常 >>>
                except Exception as q_err:
                    logger.error(f"ESP send_command: 从队列获取 '{command}' 的响应时出错: {q_err}", exc_info=True)
                    self._handle_connection_error()
                    return {"success": False, "error": f"Response queue error: {q_err}"}
            else:
                # 不需要等待响应
                logger.debug(f"ESP send_command: 命令 '{command}' 已发送，无需等待响应。") # <<< 新增日志
                return {"success": True, "message": "Command sent, no response requested"}

        except Exception as e:
            logger.error(f"发送/等待 ESP 命令 '{command}' 时出错: {e}", exc_info=True)
            self._handle_connection_error() # 发生未知错误，假设连接失败
            return {"success": False, "error": f"Send/Wait error: {str(e)}"}

    # --- Specific ESP Actions (保持不变) ---
    def _send_gcode(self, gcode: str) -> Dict[str, Any]:
        """Helper to send G-code like commands."""
        return self.send_command(gcode)

    def click(self, x: int, y: int) -> Dict[str, Any]:
        """Sends a click command (M1) using **absolute physical coordinates** (mm)."""
        # 确保发送的是物理坐标
        return self._send_gcode(f"G X{int(x)} Y{int(y)} M1")

    def pixel_click(self, x: int, y: int) -> Dict[str, Any]:
        """Sends a click command using **pixel coordinates** (if ESP supports)."""
        # 假设 ESP 能区分大小写 x, y 代表像素坐标
        return self._send_gcode(f"G x{int(x)} y{int(y)} M1")

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> Dict[str, Any]:
        """Sends a long press command (M2) using **absolute physical coordinates** (mm)."""
        # 确保发送的是物理坐标
        # 注意: ESP 是否处理 duration_ms 取决于其固件
        return self._send_gcode(f"G X{int(x)} Y{int(y)} M2")

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 500) -> Dict[str, Any]:
        """Sends a swipe command (M3) using **absolute physical coordinates** (mm)."""
        # 确保发送的是物理坐标
        sx, sy, ex, ey = map(int, [start_x, start_y, end_x, end_y])
        # 注意: ESP M3 是否处理结束坐标和 duration_ms 取决于固件
        # G X{sx} Y{sy} M3 X{ex} Y{ey} 是一个合理的猜测格式
        return self._send_gcode(f"G X{sx} Y{sy} M3 X{ex} Y{ey}") # 假设 M3 能处理结束坐标

    def home(self) -> Dict[str, Any]:
        """Sends command to home the axes (absolute 0,0)."""
        return self.send_command("HOME")

    def get_status(self) -> Dict[str, Any]:
        """Requests status from the ESP."""
        return self.send_command("STATUS")

    def set_origin(self) -> Dict[str, Any]:
        """Sets the current position as the origin (0,0)."""
        # 注意：这通常是 G代码的 G92 X0 Y0，但这里是自定义命令 "O"
        return self.send_command("O")

    def move_x(self, distance: float) -> Dict[str, Any]:
        """Moves X axis by a relative distance (mm)."""
        # 注意：这通常是 G代码 G91 G0 X{dist} G90，但这里是自定义命令 "X {dist}"
        # 需要确认 ESP 是否能处理这种相对移动命令
        return self.send_command(f"X {distance:.2f}")

    def move_y(self, distance: float) -> Dict[str, Any]:
        """Moves Y axis by a relative distance (mm)."""
        # 同上，需要确认 ESP 是否支持 "Y {dist}"
        return self.send_command(f"Y {distance:.2f}")


class TaskExecutor(QObject):
    """
    以串行方式执行设备上的任务步骤。由 TaskScheduler 的主循环调用。
    Executes task steps on a device in a serial manner. Called by the TaskScheduler's main loop.
    """
    # 信号定义 (保持不变)
    screenshot_updated = pyqtSignal(object, str)
    task_progress_updated = pyqtSignal(str, str)
    request_human_intervention = pyqtSignal(dict) # 这个信号现在由 _execute_parsed_action 触发

    # 动作执行后的物理稳定延时（秒）
    POST_ACTION_PHYSICAL_DELAY_SECONDS = 1.0 # 点击/滑动/返回原点后，等待机械臂稳定
    # 内部动作之间的短延时（例如，点击命令发送后，等待执行器完成再返回原点）
    INTERNAL_ACTION_DELAY_SECONDS = 0.5 # M1/M2/M3 执行后可能需要短暂等待

    def __init__(self,
                 config: Dict[str, Any],
                 screenshot_manager: 'ScreenshotManager',
                 ai_analyzer: 'AIAnalyzer',
                 esp_controller: 'ESPController',
                 task_scheduler: Optional['TaskScheduler'], # 类型提示改为 Optional
                 main_ui_ref: 'MainUI'):
        super().__init__()
        self.config = config
        self.screenshot_manager = screenshot_manager
        self.ai_analyzer = ai_analyzer
        self.esp_controller = esp_controller
        self.task_scheduler = task_scheduler # 保存引用
        self.main_ui = main_ui_ref

        self.last_ai_click_target: Dict[str, Optional[Tuple[int, int]]] = {} # 保留用于 UI 显示
        self.last_ai_context: Dict[str, Dict[str, Any]] = {} # 保留用于调试/干预

        self.intervention_mutex = QMutex() # 人工干预互斥锁
        self.intervention_result: Optional[bool] = None
        # self.request_human_intervention.connect(self._handle_intervention_request) # 不再直接连接，通过主UI弹出

        # --- 全局 ESP 物理动作锁 (保护原子操作序列) ---
        self.esp_action_lock = threading.Lock()
        logger.info("TaskExecutor 初始化: ESP 物理动作锁已创建。")

        # --- 环境准备锁 (现在用于保护单个准备步骤) ---
        self.preparation_lock = threading.Lock()
        logger.info("TaskExecutor 初始化: 环境准备锁已创建。")

        # 可供 AI 或子任务调用的函数及其映射 (保持不变)
        self.available_functions: Dict[str, Callable[..., Any]] = {
            "click": self._execute_click,
            "long_press": self._execute_long_press,
            "swipe": self._execute_swipe,
            "wait": self._execute_wait,           # 特殊处理：返回等待标记
            "back": self._execute_back,
            "home": self._execute_home,           # 绝对硬件原点
            "return_to_device_origin": self._return_to_device_origin, # 设备配置原点
            "analyze_screen": self._execute_analyze_screen, # 无物理动作
            "search_text": self._execute_search_text,     # 无物理动作
            "scroll": self._execute_scroll,
            "esp_command": self._execute_esp_command, # 可能包含移动
            "complete_task": self._execute_complete_task # 新增：标记任务完成
        }
        # 函数描述 (更新)
        self.function_descriptions: Dict[str, str] = {
            "click(x, y)": "移动到像素坐标 (x, y) 并点击，然后返回设备原点。 [原子操作]",
            "long_press(x, y)": "移动到像素坐标 (x, y) 并长按，然后返回设备原点。 [原子操作]",
            "swipe(start_x, start_y, end_x, end_y)": "从起点像素滑动到终点像素，然后返回设备原点。 [原子操作]",
            "scroll(direction)": "滚动屏幕 ('up', 'down', 'left', 'right')，然后返回设备原点。 [原子操作]",
            "wait(seconds)": "暂停执行指定秒数。调度器将处理等待。 [无物理动作]",
            "back()": "模拟按下返回键（尝试模板/文本/默认坐标），然后返回设备原点。 [原子操作]",
            "home()": "将物理臂移动到其绝对硬件原点 (0,0)。 [原子操作]",
            "return_to_device_origin()": "将物理臂移动到当前操作设备的配置原点。 [原子操作]",
            "analyze_screen()": "强制 AI 重新详细分析当前屏幕。 [无物理动作]",
            "search_text(text)": "在屏幕上搜索文本，返回位置。 [无物理动作]",
            "esp_command(command_string)": "直接向 ESP 发送原始命令。移动命令会自动返回原点和延时。 [原子操作]",
            "complete_task()": "标记当前任务已成功完成。 [无物理动作]",
        }
        logger.info("TaskExecutor (串行模式) 初始化完成。")

    def _emit_progress(self, device_name: str, message: str):
        """辅助函数，用于发送任务进度信号。"""
        try:
            # 使用 invokeMethod 确保在主线程发送信号
            QMetaObject.invokeMethod(self, "emit_progress_signal", Qt.QueuedConnection,
                                     Q_ARG(str, device_name), Q_ARG(str, message))
        except Exception as e:
            logger.warning(f"发送进度更新信号失败: {e}")

    @pyqtSlot(str, str)
    def emit_progress_signal(self, device_name: str, message: str):
        """实际发送信号的槽函数（在主线程执行）。"""
        self.task_progress_updated.emit(device_name, message)

    def _build_ai_context(self, device: Device, task: Task, screen_text: str, text_positions: List[Dict],
                          matched_templates: List[Dict]) -> Dict[str, Any]:
        """
        构建发送给 AI 的上下文信息和 Prompt。
        【已修改】增加上一动作结果到历史记录，调整 Prompt。
        """
        try:
            prompt_lines = []

            # 1. 任务目标
            prompt_lines.append("## Your Task Goal ##") # 标题更清晰
            prompt_lines.append(f"Current Task Name: {task.name}")
            if task.app_name:
                prompt_lines.append(f"Target Application: {task.app_name}")
            prompt_lines.append(f"Overall Task Goal: {task.type.value}")
            if not task.use_ai_driver and 0 <= task.current_subtask_index < len(task.subtasks):
                 subtask_info = task.subtasks[task.current_subtask_index]
                 sub_desc = subtask_info.get('description', subtask_info.get('type'))
                 prompt_lines.append(f"Current Sub-Task (if applicable): {sub_desc} - Goal: {subtask_info.get('goal', 'Execute this sub-task')}")
            prompt_lines.append("-" * 20)

            # 2. 当前屏幕信息 (OCR 文本)
            prompt_lines.append("## Current Screen Text ##")
            max_text_len = 1500
            truncated_text = screen_text[:max_text_len]
            if len(screen_text) > max_text_len:
                truncated_text += "\n... (文本过长已截断)"
            prompt_lines.append(truncated_text if truncated_text else "(No text detected on screen)")
            prompt_lines.append("-" * 20)

            # 3. 可交互元素 (基于 OCR 和模板)
            prompt_lines.append("## Identified Interactive Elements on Screen ##")
            element_lines = []
            for i, item in enumerate(text_positions[:20]): # 限制数量, item 来自 text_positions
                text = item.get("text", "").replace("\n", " ").strip()
                center = item.get("center", "N/A")
                if text:
                    # item['confidence'] 此时应该是浮点数了
                    # 保留 .get() 和默认值作为最后防线，但理论上应该不会触发默认值
                    confidence_to_display = item.get('confidence', 0.0) # 假设已是 float
                    element_lines.append(f"  - Text[{i}]: '{text}' (Center: {center}, Confidence: {confidence_to_display:.2f})")


            for i, match in enumerate(matched_templates[:10]): # 限制数量
                name = match.get("name", "Unknown Template")
                center = match.get("center", "N/A")
                confidence = match.get("confidence", 0) # template_matching 返回的 confidence 是 float
                element_lines.append(f"  - Template[{i}]: '{name}' (Center: {center}, Confidence: {confidence:.2f})")

            if element_lines:
                prompt_lines.extend(element_lines)
                if len(text_positions) > 20 or len(matched_templates) > 10:
                    prompt_lines.append("  ... (more elements identified but omitted for brevity)")
            else:
                prompt_lines.append("(No specific text or template elements identified for interaction)")
            prompt_lines.append("-" * 20)

            # 4. 可用动作 (更新 wait 描述)
            prompt_lines.append("## Available Actions You Can Use ##")
            for func_name, desc in self.function_descriptions.items():
                if func_name == "wait(seconds)":
                    prompt_lines.append(f"  - wait(seconds): Pause execution for a specific duration (e.g., wait(3.5) for 3.5s). You decide the duration if needed.")
                else:
                    prompt_lines.append(f"  - {desc}")
            prompt_lines.append("-" * 20)

            # 5. 近期动作历史 (包含结果)
            prompt_lines.append("## Recent Action History (Oldest to Newest, with Results) ##")
            history_limit = 7 # 调整历史记录数量
            recent_history = device.action_history[-history_limit:]
            if recent_history:
                for action_entry in recent_history:
                    ts = action_entry.get('timestamp', '')
                    act = action_entry.get('action', 'Unknown Action')
                    rat = action_entry.get('rationale', 'No rationale provided')

                    result_str = ""
                    if 'result_success' in action_entry: # 检查key是否存在
                        if action_entry['result_success']:
                            result_str = "-> Result: Succeeded."
                            # 如果是检查类，也显示检查结果
                            if 'condition_met' in action_entry:
                                result_str += f" (Condition met: {action_entry['condition_met']})"
                        else:
                            err_detail = action_entry.get('result_error', 'Unknown error')
                            result_str = f"-> Result: Failed. Reason: {err_detail}"
                    else:
                        result_str = "-> Result: Not recorded or N/A." # 如果没有结果信息

                    prompt_lines.append(f"  - [{ts}] Action: `{act}`. Rationale: {rat}. {result_str}")
            else:
                prompt_lines.append("(No recent actions in history for this task execution.)")
            prompt_lines.append("-" * 20)

            # 6. 指令 (强调基于历史和当前屏幕，以及任务完成的条件)
            prompt_lines.append("## Your Instruction ##")
            prompt_lines.append(
                "Based on the Overall Task Goal, Current Sub-Task (if any), current screen information, AND the results of recent actions, "
                "determine the single best next action. Use ONLY the available functions.")
            prompt_lines.append(
                "If the previous action was successful and achieved a significant step or the overall goal, consider using `complete_task()`."
                "If you need to wait for something to load or for an animation, use `wait(appropriate_seconds)` with a duration you deem necessary.")
            prompt_lines.append(
                "Provide the function call ONLY, followed by a newline and 'Justification:'.")
            prompt_lines.append("Example: click(123, 456)\nJustification: Clicking the 'Login' button to proceed.")
            prompt_lines.append("If unsure, or if the screen is unexpected, use `analyze_screen()` to re-evaluate, or `back()` to attempt to return to a known state.")

            final_prompt = "\n".join(prompt_lines)
            return {"prompt": final_prompt}

        except Exception as e:
            logger.error(f"[{device.name}] 构建 AI 上下文时出错: {e}", exc_info=True)
            # 在这里也做一个保护，万一还是出错了，至少 AI prompt 能生成一部分
            safe_prompt = "\n".join(prompt_lines) if prompt_lines else ""
            safe_prompt += f"\nError building full context: {e}\nPlease try to guess the next step or use `analyze_screen()`."
            return {"prompt": safe_prompt}

    def _internal_return_to_device_origin(self, device: Device) -> Dict[str, Any]:
        """
        【内部辅助函数】发送命令使机械臂移动到指定设备的原点 (mm 坐标)。
        这个函数 *不* 包含锁和外部延时，由调用者确保在锁内执行。
        返回 ESP 的原始响应字典。
        """
        origin_x = device.get_config("machine_origin_x", 0.0)
        origin_y = device.get_config("machine_origin_y", 0.0)
        try:
            origin_x_abs = float(origin_x)
            origin_y_abs = float(origin_y)
        except (ValueError, TypeError):
            logger.error(f"[{device.name}] 设备原点坐标配置无效 ({origin_x}, {origin_y})，使用绝对原点 (0,0)！")
            # 如果配置错误，内部改为发送绝对 Home 命令
            return self.esp_controller.send_command("HOME") # 发送绝对 Home

        # 使用浮点数发送，ESP 端负责处理
        command = f"G X{origin_x_abs:.2f} Y{origin_y_abs:.2f}" # G 默认是快速移动 (G0)
        logger.debug(f"[{device.name}] 内部: 发送返回设备原点命令: {command}")
        # 直接发送命令，不处理锁和延时
        return self.esp_controller.send_command(command, wait_for_response=True)

    def _execute_home(self, device: Device) -> Dict[str, Any]:
        """
        【原子操作】执行 ESP **绝对**回原点 (HOME) 命令，包含锁和延时。
        """
        logger.info(f"[{device.name}] 动作: 请求执行 ESP **绝对** Home...") # <<< 新增日志
        result = {"success": False, "error": "锁获取失败"}
        logger.debug(f"[{device.name}] 动作 _execute_home: 尝试获取 esp_action_lock...") # <<< 新增日志
        if not self.esp_action_lock.acquire(timeout=10): # 尝试获取锁，设置超时
            logger.error(f"[{device.name}] 动作 _execute_home: 获取 ESP 动作锁超时！") # <<< 新增日志
            return result
        try:
            logger.info(f"[{device.name}] [ESP 锁已获取] 动作 _execute_home: 调用 esp_controller.home()...") # <<< 新增日志
            result = self.esp_controller.home() # 调用 ESP 控制器发送 HOME
            logger.debug(f"[{device.name}] 动作 _execute_home: esp_controller.home() 返回结果: {result}") # <<< 新增日志
            if result.get("success"):
                logger.debug(f"[{device.name}] 动作 _execute_home: Home 命令成功。等待延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS}s...") # <<< 新增日志
                time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS) # 在锁内延时
                logger.debug(f"[{device.name}] 动作 _execute_home: 延时结束。") # <<< 新增日志
            else:
                logger.error(f"[{device.name}] 动作 _execute_home: ESP Home 命令失败: {result.get('error')}") # <<< 新增日志
            # logger.info(f"[{device.name}] 完成 ESP **绝对** Home。") # 这行日志意义不大，可以在 finally 后加
        finally:
            logger.info(f"[{device.name}] [ESP 锁已释放] 动作 _execute_home.") # <<< 新增日志
            self.esp_action_lock.release()
        logger.info(f"[{device.name}] 动作: ESP 绝对 Home 执行完毕。") # <<< 新增日志
        return result

    def _return_to_device_origin(self, device: Device) -> Dict[str, Any]:
        """
        【原子操作】将物理臂移动到当前活动设备的**配置原点**，包含锁和延时。
        """
        logger.info(f"[{device.name}] 请求返回到设备原点...")
        # 验证坐标有效性在 _internal_return_to_device_origin 内部处理

        result = {"success": False, "error": "锁获取失败"}
        if not self.esp_action_lock.acquire(timeout=10):
            logger.error(f"[{device.name}] 获取 ESP 动作锁超时 (返回设备原点)。")
            return result
        try:
            logger.info(f"[{device.name}] [Lock Acquired] 返回设备原点...")
            # 调用内部辅助函数发送命令
            result = self._internal_return_to_device_origin(device)
            if result.get("success"):
                logger.debug(f"[{device.name}] 返回设备原点命令成功。等待延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS} 秒...")
                time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS) # 在锁内延时
                logger.debug(f"[{device.name}] 返回设备原点后延时结束。")
            else:
                logger.error(f"[{device.name}] 返回到设备原点失败: {result.get('error')}")
            logger.info(f"[{device.name}] 完成返回设备原点。")
        finally:
            logger.debug(f"[{device.name}] [Lock Releasing] 返回设备原点")
            self.esp_action_lock.release()
        return result

    def _pixel_to_commanded_machine_coords(self, x_pixel: int, y_pixel: int, device: Device) -> Tuple[float, float]:
        """
        【新 V5 函数】将屏幕像素坐标转换为 ESP 控制器需要接收的绝对物理机台指令坐标 (mm)。
        该计算已包含笔尖相对于机台的偏移量。

        转换推导:
        1. 像素到物理位置 (相机视角中心点对应的机台坐标):
           校准点1: Px(0, 0)   -> M(115, 274)
           校准点2: Px(1080, 1440) -> M(-15, 84)
           target_mx = (-130/1080) * px + 115
           target_my = (-190/1440) * py + 274
        2. 笔尖偏移 (机台在0,0时笔尖在5,114):
           pen_offset_x = 5
           pen_offset_y = 114
        3. 计算指令坐标 (让笔尖落在 target_m):
           commanded_mx = target_mx - pen_offset_x
           commanded_my = target_my - pen_offset_y
           commanded_mx = (-130/1080) * px + 110
           commanded_my = (-190/1440) * py + 160

        参数:
            x_pixel (int): 屏幕像素 X 坐标 (裁剪后)
            y_pixel (int): 屏幕像素 Y 坐标 (裁剪后)
            device (Device): 设备对象 (当前未使用，但保留以备将来扩展)

        返回:
            Tuple[float, float]: 机器人需要移动到的绝对物理坐标 (Commanded_X_mm, Commanded_Y_mm)
        """
        logger.debug(f"[{device.name}] 像素到指令坐标转换 (V5): 输入 Px=({x_pixel}, {y_pixel})")

        # 应用推导出的转换公式
        scale_x = -130.0 / 1080.0
        offset_x = 110.0  # 已减去笔尖 X 偏移
        scale_y = -190.0 / 1440.0
        offset_y = 160.0  # 已减去笔尖 Y 偏移

        commanded_mx = scale_x * float(x_pixel) + offset_x
        commanded_my = scale_y * float(y_pixel) + offset_y

        logger.debug(f"[{device.name}] ... 计算得出的指令坐标 (未钳位): Cmd=({commanded_mx:.2f}, {commanded_my:.2f}) mm")

        try:
            # Use get_config to safely fetch device-specific or default values
            device_origin_x = float(device.get_config("MACHINE_ORIGIN_X", 0.0))  # Use UPPERCASE keys
            device_origin_y = float(device.get_config("MACHINE_ORIGIN_Y", 0.0))  # Use UPPERCASE keys
            logger.debug(
                f"[{device.name}] ... Target device origin offset: ({device_origin_x:.2f}, {device_origin_y:.2f})")
        except (ValueError, TypeError) as e:
            logger.error(f"[{device.name}] Invalid device origin config. Using (0,0). Error: {e}")
            device_origin_x = 0.0
            device_origin_y = 0.0

        final_x = commanded_mx + device_origin_x
        final_y = commanded_my + device_origin_y

        # 返回计算出的最终指令浮点坐标 (钳位将在发送前进行)
        return final_x, final_y

    # --- 人工干预处理 (保持不变) ---
    @pyqtSlot(dict)
    def _handle_intervention_request(self, intervention_data: dict):
        """槽函数：在主线程中处理人工干预请求，弹出对话框。"""
        # 确保在主线程执行
        if threading.current_thread() != threading.main_thread():
            logger.warning("尝试在非主线程处理人工干预请求，将使用 invokeMethod。")
            QMetaObject.invokeMethod(self, "_handle_intervention_request", Qt.QueuedConnection, Q_ARG(dict, intervention_data))
            return

        dialog = HumanInterventionDialog(
            action_str=intervention_data.get("action_str", "N/A"),
            justification=intervention_data.get("justification", "N/A"),
            ai_prompt=intervention_data.get("ai_prompt", "N/A"),
            ai_response=intervention_data.get("ai_response_raw", "N/A"),
            parent=self.main_ui # 确保父窗口设置正确
        )
        result = dialog.exec_()

        with QMutexLocker(self.intervention_mutex):
            self.intervention_result = (result == QDialog.Accepted)


    # --- 核心执行逻辑 ---
    def execute_next_step(self, device: Device, task: Task) -> Dict[str, Any]:
        """
        执行任务的下一个逻辑步骤。由 TaskScheduler 调用。
        返回一个字典，包含 success 和其他可选信息。
        【已修改 V4】准备成功后返回 "skipped" 标记，让 handle_step_result 忽略。
        """
        if task.status == TaskStatus.CANCELED:
            logger.info(f"[{device.name}] 任务 '{task.name}' 在执行步骤前已被取消。")
            return {"success": False, "error": "任务已取消", "canceled": True}

        allowed_statuses = [DeviceStatus.BUSY, DeviceStatus.INITIALIZING]
        if device.status not in allowed_statuses:
            logger.warning(f"[{device.name}] execute_next_step 调用时设备状态为 {device.status.value} (不在允许状态 {allowed_statuses} 中)，跳过执行。")
            # 返回特殊标记，表示非错误，只是未执行
            return {"success": True, "skipped": True, "reason": f"Device status {device.status.value} not ready"}

        logger.debug(f"[{device.name}] execute_next_step: 开始处理任务 '{task.name}', 设备状态: {device.status.value}, 任务阶段: {task.task_stage}")

        try:
            # --- 1. 处理准备阶段 ---
            if task.task_stage == "PREPARING":
                if device.status != DeviceStatus.INITIALIZING:
                     logger.warning(f"[{device.name}] 尝试在非 INITIALIZING 状态 ({device.status.value}) 下执行 PREPARING 阶段，跳过。")
                     return {"success": True, "skipped": True, "reason": "Incorrect state for PREPARING stage."}

                logger.info(f"[{device.name}] execute_next_step: 任务处于 PREPARING 阶段，尝试获取 'preparation_lock'...")
                # 使用 with 确保锁释放
                with self.preparation_lock:
                    logger.info(f"[{device.name}] [准备锁已获取] execute_next_step: 执行准备步骤 (任务: '{task.name}')...")
                    prep_result = self._execute_preparation_step(device, task)
                    if prep_result.get("success"):
                        logger.info(f"[{device.name}] 环境准备成功，将设备状态从 INITIALIZING 更改为 BUSY。")
                        device.status = DeviceStatus.BUSY
                        task.task_stage = "RUNNING"
                        task.current_step = 0
                        task.current_subtask_index = 0 # 准备完成后索引必须是0
                        device.update_progress("环境准备完成，开始执行任务...")
                        self._emit_progress(device.name, "环境准备完成，开始执行...")
                        logger.info(f"[{device.name}] 任务 '{task.name}' 环境准备成功，进入运行阶段。")
                        # *** !!! 返回 skipped 标记 !!! ***
                        # 告诉 _handle_step_result 这只是准备完成，不需要处理步骤结果
                        return {"success": True, "skipped": True, "reason": "Preparation complete"}
                    else:
                        logger.error(f"[{device.name}] 任务 '{task.name}' 环境准备失败: {prep_result.get('error')}")
                        device.update_progress(f"环境准备失败: {prep_result.get('error')}")
                        self._emit_progress(device.name, f"环境准备失败: {prep_result.get('error')}")
                        # 返回准备失败结果，让 _handle_step_result 处理失败/重试
                        return {"success": False, "error": f"环境准备失败: {prep_result.get('error')}", "force_fail": True}
                # 锁自动释放

            # --- 2. 处理运行阶段 (AI 或 子任务) ---
            elif task.task_stage == "RUNNING":
                if device.status != DeviceStatus.BUSY:
                     logger.warning(f"[{device.name}] 尝试在非 BUSY 状态 ({device.status.value}) 下执行 RUNNING 阶段，跳过。")
                     return {"success": True, "skipped": True, "reason": "Incorrect state for RUNNING stage."}

                if task.use_ai_driver:
                    logger.info(f"[{device.name}] 执行 AI 步骤 {task.current_step + 1}: 任务 '{task.name}'")
                    ai_step_result = self._execute_single_ai_step(device, task)
                    # 返回 AI 步骤的实际结果，让 _handle_step_result 处理
                    return ai_step_result
                else:
                    total_subtasks = len(task.subtasks)
                    # 检查索引有效性
                    if not (0 <= task.current_subtask_index < total_subtasks):
                         logger.error(f"[{device.name}] 任务 '{task.name}': 无效的子任务索引 ({task.current_subtask_index})，总数 {total_subtasks}。任务将失败。")
                         return {"success": False, "error": f"无效的子任务索引 {task.current_subtask_index}", "force_fail": True}

                    subtask = task.subtasks[task.current_subtask_index]
                    subtask_desc = subtask.get('description', subtask.get('type', 'N/A'))
                    # 使用正确的索引号 (index+1)
                    logger.info(f"[{device.name}] 执行子任务 {task.current_subtask_index + 1}/{total_subtasks} ('{subtask_desc}'): 任务 '{task.name}'")
                    progress_msg = f"子任务 {task.current_subtask_index + 1}/{total_subtasks}: {subtask_desc[:40]}"
                    device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)

                    # 执行子任务
                    subtask_result = self._dispatch_subtask(device, task, subtask)
                    # 返回子任务的实际结果，让 _handle_step_result 处理
                    return subtask_result

            # --- 3. 处理等待阶段 ---
            elif task.task_stage == "WAITING":
                logger.debug(f"[{device.name}] 任务 '{task.name}' 处于等待阶段，跳过执行。")
                return {"success": True, "skipped": True, "reason": "Task is WAITING"}

            # --- 4. 处理已完成/失败/取消阶段 ---
            elif task.task_stage in ["COMPLETED", "FAILED", "CANCELED"]:
                logger.warning(f"[{device.name}] 尝试执行已处于 '{task.task_stage}' 状态的任务 '{task.name}'。")
                # 理论上不应到达这里，因为任务完成后会从 running_tasks 移除
                # 但作为防御性编程，返回一个错误
                return {"success": False, "error": f"任务已处于 {task.task_stage} 状态"}

            else:
                logger.error(f"[{device.name}] 未知任务阶段: {task.task_stage} for task '{task.name}'")
                return {"success": False, "error": f"未知任务阶段: {task.task_stage}", "force_fail": True}

        except Exception as e:
            logger.error(f"[{device.name}] 执行任务步骤时发生意外异常，任务 '{task.name}': {e}", exc_info=True)
            return {"success": False, "error": f"执行步骤异常: {e}", "exception": True, "force_fail": True}

    def _execute_single_ai_step(self, device: Device, task: Task) -> Dict[str, Any]:
        """
        执行 AI 驱动任务的单个步骤：截图 -> 分析 -> 决策 -> 执行动作。
        内部调用的动作函数 (_execute_click 等) 已包含锁和延时。
        【已修改】调用新增的 _build_ai_context 方法。
        """
        annotated_image = None
        step_num = task.current_step + 1

        try:
            progress_msg = f"AI 步骤 {step_num}: 准备环境 (截图等)"
            device.update_progress(progress_msg);
            self._emit_progress(device.name, progress_msg)

            origin_result = self._return_to_device_origin(device)
            if not origin_result.get("success"):
                logger.warning(
                    f"[{device.name}] AI 步骤 {step_num}: 获取截图前返回原点失败: {origin_result.get('error')}")

            stabilization_delay = self.config.get("CAMERA_STABILIZATION_DELAY_SECONDS", 1.0)  # 从配置读取延时
            if stabilization_delay > 0:
                logger.debug(f"[{device.name}] AI 步骤 {step_num}: 等待 {stabilization_delay:.2f} 秒让摄像头稳定...")
                time.sleep(stabilization_delay)

            screenshot = self.screenshot_manager.take_screenshot(device)
            if screenshot is None:
                logger.warning(f"[{device.name}] AI 步骤 {step_num}: 获取截屏失败。")
                return {"success": False, "error": "获取截屏失败"}
            device.last_screenshot = screenshot
            annotated_image = screenshot.copy()

            progress_msg = f"AI 步骤 {step_num}: 图像处理与 OCR"
            device.update_progress(progress_msg);
            self._emit_progress(device.name, progress_msg)
            enhanced_image = self.screenshot_manager.enhance_image(screenshot)
            image_for_ocr = enhanced_image if enhanced_image is not None else screenshot
            if image_for_ocr is None:
                logger.error(f"[{device.name}] AI 步骤 {step_num}: OCR失败，增强和原始截图都无效。")
                return {"success": False, "error": "无法获取有效图像进行OCR"}
            ocr_result = self.ai_analyzer.perform_ocr(image_for_ocr)
            device.last_ocr_result = ocr_result

            text_positions = []  # 这是传递给 _build_ai_context 的
            if ocr_result and ocr_result.get("code") == 100 and "data" in ocr_result:
                for item in ocr_result["data"]:  # 这里的 item 是经过 _filter_ocr_results 处理的
                    text = item.get("text", "");
                    box = item.get("box")
                    if text and box and len(box) == 4:
                        try:
                            center_x = sum(p[0] for p in box) // 4;
                            center_y = sum(p[1] for p in box) // 4
                            # 修改：从 item.get("score") 获取置信度，并存到 "confidence" 键
                            # 提供默认值 0.0，以防 "score" 意外缺失
                            confidence_score = item.get("score", 0.0)
                            text_positions.append({
                                "text": text,
                                "center": (center_x, center_y),
                                "box": box,
                                "confidence": confidence_score  # 将 score 映射到 confidence
                            })
                            # 绘制 OCR 框到 annotated_image
                            pts = np.array(box, np.int32).reshape((-1, 1, 2))
                            if annotated_image is not None:
                                cv2.polylines(annotated_image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                            else:
                                logger.warning("无法绘制 OCR 框，annotated_image 为 None。")
                        except Exception as draw_err:
                            logger.warning(f"填充 text_positions 或绘制 OCR 框时出错: {draw_err}, item: {item}")

            matched_templates = []
            if annotated_image is not None:
                matched_templates = self.ai_analyzer.match_all_templates(annotated_image)
                for match in matched_templates:
                    try:
                        top_left, bottom_right = match['top_left'], match['bottom_right']
                        cv2.rectangle(annotated_image, top_left, bottom_right, (255, 0, 0), 2)
                        cv2.putText(annotated_image, match['name'], (top_left[0], top_left[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    except Exception as draw_err:
                        logger.warning(f"绘制模板框 '{match.get('name')}' 时出错: {draw_err}")
            else:
                logger.warning("无法进行模板匹配或绘制，annotated_image 为 None。")

            progress_msg = f"AI 步骤 {step_num}: 准备 AI 上下文"
            device.update_progress(progress_msg);
            self._emit_progress(device.name, progress_msg)

            screen_text_for_context = "\n".join(
                [tp_item.get('text', '') for tp_item in text_positions])  # 从已处理的 text_positions 获取
            context_data = self._build_ai_context(device, task, screen_text_for_context, text_positions,
                                                  matched_templates)
            self.last_ai_context[device.name] = context_data

            progress_msg = f"AI 步骤 {step_num}: 查询 AI..."
            device.update_progress(progress_msg);
            self._emit_progress(device.name, progress_msg)
            ai_response = self.ai_analyzer.get_ai_decision(context_data["prompt"], device.action_history,
                                                           f"{task.name} - 步骤 {step_num}")

            if not ai_response or ai_response.get("decision") is None:
                error_detail = ai_response.get('error', '无响应') if ai_response else '无响应'
                logger.error(f"[{device.name}] AI 步骤 {step_num}: 获取 AI 决策失败。错误: {error_detail}")
                if annotated_image is not None: self.screenshot_updated.emit(annotated_image.copy(), device.name)
                return {"success": False, "error": f"AI 决策失败: {error_detail}"}

            decision_log_msg = f"设备: {device.name} | 任务: {task.name} | 步骤: {step_num} | 决策: {ai_response['decision']} | 理由: {ai_response.get('justification', 'N/A')}"
            ai_decision_logger.info(decision_log_msg)

            parsed_action = self._parse_ai_decision_only(ai_response["decision"])
            action_name = parsed_action.get("name");
            action_args = parsed_action.get("args", [])
            raw_ai_response_str = "N/A"
            if ai_response.get("raw_response"):
                try:
                    raw_ai_response_str = json.dumps(ai_response["raw_response"], indent=2, ensure_ascii=False)
                except Exception:
                    raw_ai_response_str = str(ai_response["raw_response"])

            if annotated_image is not None and action_name in ["click", "long_press"] and len(action_args) >= 2:
                try:
                    click_x, click_y = int(action_args[0]), int(action_args[1])
                    h_img, w_img = annotated_image.shape[:2]  # 使用 annotated_image 的尺寸
                    if 0 <= click_x < w_img and 0 <= click_y < h_img:  # 检查点击坐标是否在图像范围内
                        cv2.circle(annotated_image, (click_x, click_y), 15, (0, 0, 255), 2)
                        cv2.drawMarker(annotated_image, (click_x, click_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    else:
                        logger.warning(
                            f"AI 建议点击坐标 ({click_x}, {click_y}) 超出截图范围 ({w_img}x{h_img})，无法标注。")
                except (ValueError, TypeError, IndexError) as mark_err:
                    logger.warning(f"解析或标注 AI 点击坐标时出错: {mark_err}")

            if annotated_image is not None: self.screenshot_updated.emit(annotated_image.copy(), device.name)

            progress_msg = f"AI 步骤 {step_num}: 执行 AI 动作 ({action_name or 'N/A'})"
            device.update_progress(progress_msg);
            self._emit_progress(device.name, progress_msg)

            action_result = self._execute_parsed_action(
                device=device, task=task, func_name=action_name, args=action_args,
                justification=ai_response.get("justification", ""),
                text_positions=text_positions,  # 传递已处理的 text_positions
                matched_templates=matched_templates,
                ai_prompt=context_data["prompt"], ai_response_raw=raw_ai_response_str
            )
            return action_result

        except Exception as loop_err:
            logger.error(f"[{device.name}] AI 步骤 {step_num} 执行中发生意外异常: {loop_err}", exc_info=True)
            if annotated_image is not None:
                try:
                    self.screenshot_updated.emit(annotated_image.copy(), device.name)
                except Exception:
                    pass
            return {"success": False, "error": f"AI 步骤异常: {loop_err}", "exception": True}

    def _execute_preparation_step(self, device: Device, task: Task) -> Dict[str, Any]:
        """
        执行任务环境准备步骤。
        内部调用的函数 (_navigate_to_home_screen, _launch_application) 已包含原子操作。
        """
        # 注意：调用此函数时，preparation_lock 已被持有
        logger.info(f"[{device.name}] --- 开始环境准备 (任务: '{task.name}') ---") # <<< 新增日志
        try:
            # --- 1. 初始绝对回原点 (原子操作) ---
            progress_msg = "准备: 初始绝对回原点...";
            logger.info(f"[{device.name}] 准备步骤 1/5: {progress_msg}") # <<< 改为 INFO 并增加步骤号
            device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
            home_result = self._execute_home(device) # 已包含锁和延时
            if not home_result.get("success"):
                logger.error(f"[{device.name}] 准备步骤 1/5 失败: 初始绝对回原点失败: {home_result.get('error')}") # <<< 新增日志
                return {"success": False, "error": f"初始绝对回原点失败: {home_result.get('error')}"}
            logger.info(f"[{device.name}] 准备步骤 1/5 成功: 初始绝对回原点完成。") # <<< 新增日志

            # --- 2. 基础连接检查 ---
            progress_msg = "准备: 检查连接...";
            logger.info(f"[{device.name}] 准备步骤 2/5: {progress_msg}") # <<< 改为 INFO 并增加步骤号
            device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
            # 检查 ESP
            if not self.esp_controller.connected:
                logger.info(f"[{device.name}] 准备步骤 2/5: ESP 未连接，尝试连接...")
                if not self.esp_controller.connect():
                    logger.error(f"[{device.name}] 准备步骤 2/5 失败: ESP 控制器连接失败。")
                    return {"success": False, "error": "ESP 控制器连接失败"}
            # 检查主摄像头 ADB
            if not self.screenshot_manager.connect_device(): # 检查主摄像头
                logger.error(f"[{device.name}] 准备步骤 2/5 失败: 主摄像头 ADB 连接失败。")
                return {"success": False, "error": "主摄像头 ADB 连接失败"}
            logger.info(f"[{device.name}] 准备步骤 2/5 成功: 连接检查通过。") # <<< 新增日志

            # --- 3. 导航到主屏幕 (内部包含原子操作) ---
            progress_msg = "准备: 导航到主屏幕...";
            logger.info(f"[{device.name}] 准备步骤 3/5: {progress_msg}") # <<< 改为 INFO 并增加步骤号
            device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
            home_nav_result = self._navigate_to_home_screen(device) # 不再需要 stop_event
            if not home_nav_result.get("success"):
                logger.error(f"[{device.name}] 准备步骤 3/5 失败: {home_nav_result.get('error', '导航到主屏幕失败')}") # <<< 新增日志
                return {"success": False, "error": home_nav_result.get("error", "导航到主屏幕失败")}
            logger.info(f"[{device.name}] 准备步骤 3/5 成功: 导航到主屏幕完成。") # <<< 新增日志

            # --- 4. 启动目标应用 (如果需要, 内部包含原子操作) ---
            if task.app_name:
                progress_msg = f"准备: 启动应用 {task.app_name}...";
                logger.info(f"[{device.name}] 准备步骤 4/5: {progress_msg}") # <<< 改为 INFO 并增加步骤号
                device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
                launch_result = self._launch_application(device, task.app_name) # 不再需要 stop_event
                if not launch_result.get("success"):
                    logger.error(f"[{device.name}] 准备步骤 4/5 失败: {launch_result.get('error', f'启动应用 {task.app_name} 失败')}") # <<< 新增日志
                    return {"success": False, "error": launch_result.get("error", f"启动应用 {task.app_name} 失败")}
                logger.info(f"[{device.name}] 准备步骤 4/5 成功: 应用 '{task.app_name}' 启动完成。") # <<< 新增日志
            else:
                logger.info(f"[{device.name}] 准备步骤 4/5: 无需启动应用。") # <<< 新增日志

            # --- 5. 重置设备状态变量 ---
            logger.info(f"[{device.name}] 准备步骤 5/5: 重置设备任务状态变量。") # <<< 改为 INFO 并增加步骤号
            device.action_history = []
            device.error_count = 0
            device.waiting_until = None
            self.last_ai_context.pop(device.name, None) # 清除旧的 AI 上下文

            logger.info(f"[{device.name}] --- 环境准备成功 (任务: '{task.name}') ---") # <<< 新增日志
            return {"success": True}

        except Exception as e:
            logger.error(f"[{device.name}] --- 环境准备失败 (任务: '{task.name}', 发生异常) ---", exc_info=True) # <<< 新增日志
            return {"success": False, "error": f"环境准备异常: {e}"}


    def _launch_application(self, device: Device, app_name: str) -> Dict[str, Any]:
        """尝试通过模板匹配和滑动查找并启动应用。不接收 stop_event。"""
        app_template_name = app_name # 假设模板名与应用名一致
        timeout = self.config.get("APP_LAUNCH_TIMEOUT", 60)
        swipe_attempts = self.config.get("APP_LAUNCH_SWIPE_ATTEMPTS", 4)
        click_wait = self.config.get("APP_LAUNCH_CLICK_WAIT_SECONDS", 5)
        logger.info(f"[{device.name}] 尝试启动应用 '{app_name}' (模板: '{app_template_name}', 超时: {timeout}s)...")

        start_time = time.time()
        swipe_count = 0
        swipe_directions = ['right'] * (swipe_attempts // 2) + ['left'] * (swipe_attempts - swipe_attempts // 2)

        while time.time() - start_time < timeout:
            # 检查任务是否已被外部取消（通过调度器）
            # 注意：这里无法直接访问 task 对象，但可以检查设备状态
            # if device.status == DeviceStatus.IDLE or device.status == DeviceStatus.ERROR:
            #     logger.warning(f"[{device.name}] 启动应用时检测到设备状态为 {device.status.value}，可能任务已被取消或失败。")
            #     return {"success": False, "error": f"设备状态异常 ({device.status.value})"}

            # --- 每次查找前，确保机械臂在设备原点 (原子操作) ---
            logger.debug(f"[{device.name}] LaunchApp: 查找前确保在设备原点...")
            origin_result = self._return_to_device_origin(device)
            if not origin_result.get("success"):
                logger.warning(f"[{device.name}] LaunchApp: 查找前返回设备原点失败: {origin_result.get('error')}. 仍尝试查找...")
                # 即使失败也继续尝试查找

            logger.debug(f"[{device.name}] 查找应用 '{app_name}' 模板...")
            coords = self._find_template_on_screen(device, app_template_name, threshold=0.75)

            if coords:
                logger.info(f"[{device.name}] 找到应用 '{app_name}' 模板，点击坐标 {coords}。")
                # --- 调用点击 (原子操作) ---
                click_result = self._execute_click(device, coords[0], coords[1])
                device.add_action({"action": f"Auto: Click App '{app_name}'", "rationale": "Launch Application"})

                if not click_result.get("success"):
                    logger.error(f"[{device.name}] 点击应用 '{app_name}' 失败: {click_result.get('error')}")
                    # 失败后继续循环，允许滑动或重试查找
                else:
                    logger.info(f"[{device.name}] 点击并返回原点完成，额外等待 {click_wait} 秒让应用启动...")
                    self._emit_progress(device.name, f"等待应用 '{app_name}' 启动...")
                    # 使用非阻塞等待，防止卡住调度器 (虽然这里是在执行器内部，但好习惯)
                    # 实际上，由于是串行执行，这里的 time.sleep 会阻塞当前步骤
                    # 但因为是准备阶段，通常可以接受
                    time.sleep(click_wait)
                    return {"success": True}

            elif swipe_count < swipe_attempts:
                direction = swipe_directions[swipe_count]
                logger.info(f"[{device.name}] 未找到应用，尝试向 {direction} 滑动 (第 {swipe_count + 1}/{swipe_attempts} 次)...")
                # --- 调用滚动 (原子操作) ---
                scroll_result = self._execute_scroll(device, direction, duration_ms=600)
                device.add_action({"action": f"Auto: Swipe {direction} for App", "rationale": "Launch Application"})

                if not scroll_result.get("success"):
                    logger.warning(f"[{device.name}] 滑动操作失败: {scroll_result.get('error')}")
                    # 滑动失败也继续循环
                swipe_count += 1
            else:
                logger.error(f"[{device.name}] 启动应用失败：滑动 {swipe_attempts} 次后仍未找到模板 '{app_template_name}'。")
                break

        error_msg = f"无法找到并启动应用 '{app_name}' (超时或未找到模板)"
        logger.error(f"[{device.name}] {error_msg}")
        self._return_to_device_origin(device) # 失败后尝试返回原点
        return {"success": False, "error": error_msg}

    def _navigate_to_home_screen(self, device: Device) -> Dict[str, Any]:
        """
        尝试通过重复按返回键导航到主屏幕，并验证。不接收 stop_event。
        """
        home_anchor_texts = device.get_config("HOME_SCREEN_ANCHOR_TEXTS", [])
        min_anchor_count = device.get_config("HOME_SCREEN_MIN_ANCHORS", 0)
        home_template_name = device.get_config("HOME_SCREEN_TEMPLATE_NAME")
        home_template_threshold = device.get_config("HOME_SCREEN_TEMPLATE_THRESHOLD")
        max_attempts = self.config.get("MAX_BACK_ATTEMPTS_TO_HOME", 8)

        use_ocr_check = bool(home_anchor_texts) and min_anchor_count > 0
        use_template_check = bool(home_template_name) and home_template_threshold is not None

        if not use_ocr_check and not use_template_check:
            logger.warning(f"[{device.name}] 未配置主屏幕验证方法。将执行最多 {max_attempts} 次返回操作并假定成功。")
            for attempt in range(max_attempts):
                logger.debug(f"[{device.name}] 执行返回操作 (第 {attempt + 1}/{max_attempts} 次)...")
                # --- 调用 back (原子操作) ---
                back_result = self._execute_back(device)
                device.add_action({"action": "Auto: Navigate Home (Back)", "rationale": f"Attempt {attempt + 1} (No Verify)"})
                if not back_result.get("success"):
                    logger.warning(f"[{device.name}] 返回操作失败: {back_result.get('error')}")
                    # 返回失败也继续尝试
            logger.info(f"[{device.name}] 已执行最大返回次数，假定已在主屏幕。")
            origin_result = self._return_to_device_origin(device)
            return {"success": origin_result.get("success", True)}

        logger.info(f"[{device.name}] 尝试导航到主屏幕 (最多 {max_attempts} 次返回)。验证方式: "
                    f"{'OCR锚点 ' if use_ocr_check else ''}"
                    f"{'+ 模板 ' if use_ocr_check and use_template_check else ''}"
                    f"{'模板' if not use_ocr_check and use_template_check else ''}")

        for attempt in range(max_attempts + 1):
            logger.debug(f"[{device.name}] 检查/尝试导航到主屏幕 (尝试 {attempt + 1})...")
            # --- 每次检查前，确保在设备原点 (原子操作) ---
            origin_result = self._return_to_device_origin(device)
            if not origin_result.get("success"):
                logger.warning(f"[{device.name}] NavHome: 检查前返回设备原点失败: {origin_result.get('error')}. 等待后继续...")
                time.sleep(1.5) # 失败后等待一下

            # --- 获取屏幕信息 ---
            screenshot, ocr_result, text_positions, matched_templates = self._get_current_screen_context(device)

            # --- 验证是否在主屏幕 ---
            is_home = False
            if screenshot is not None:
                 # ... (验证逻辑保持不变) ...
                template_matched = False
                if use_template_check:
                    template_match_result = self.ai_analyzer.template_matching(screenshot, template_name=home_template_name, threshold=home_template_threshold)
                    if template_match_result.get("match"): template_matched = True; logger.debug(f"[{device.name}] 主屏幕模板匹配成功。")
                    if not use_ocr_check: is_home = template_matched # 只用模板
                ocr_verified = False
                if use_ocr_check:
                    if text_positions:
                        found_anchors = {anchor.strip() for anchor in home_anchor_texts if anchor.strip() and any(anchor.strip() == item.get("text","").strip() for item in text_positions)}
                        if len(found_anchors) >= min_anchor_count: ocr_verified = True; logger.debug(f"[{device.name}] 主屏幕 OCR 锚点验证成功。")
                    if not use_template_check: is_home = ocr_verified # 只用 OCR
                    elif template_matched and ocr_verified: is_home = True # 两者都需要
            else: logger.warning(f"[{device.name}] 导航检查失败：无法获取截图。")

            # --- 判断结果 ---
            if is_home:
                logger.info(f"[{device.name}] 已确认在主屏幕 (尝试 {attempt + 1})。")
                self._return_to_device_origin(device) # 确保最后在原点
                return {"success": True}

            if attempt < max_attempts:
                logger.debug(f"[{device.name}] 未在主屏幕，执行返回操作...")
                # --- 调用 back (原子操作) ---
                back_result = self._execute_back(device)
                device.add_action({"action": "Auto: Navigate Home (Back)", "rationale": f"Attempt {attempt + 1}"})
                if not back_result.get("success"):
                    logger.warning(f"[{device.name}] 返回操作失败: {back_result.get('error')}")
            else:
                logger.error(f"[{device.name}] 导航到主屏幕失败：尝试 {max_attempts} 次返回后仍未能确认主屏幕。")
                self._return_to_device_origin(device)
                return {"success": False, "error": f"无法导航到主屏幕 (尝试 {max_attempts} 次返回后未确认)"}

        # 理论上不会执行到这里
        return {"success": False, "error": "导航到主屏幕逻辑异常结束"}


    def _parse_ai_decision_only(self, decision_text: str) -> Dict[str, Any]:
        """仅解析 AI 决策字符串，返回函数名和参数，不执行。(增加 complete_task 解析)"""
        logger.debug(f"仅解析 AI 决策: '{decision_text}'")
        decision_text = decision_text.strip()
        if decision_text.startswith("```") and decision_text.endswith("```"):
             decision_text = decision_text[3:-3].strip()
             if '\n' in decision_text: decision_text = decision_text.split('\n', 1)[0].strip()

        result = {"name": None, "args": [], "completed": False, "error": None}

        # 检查是否是完成指令
        if decision_text.lower() in ["complete_task()", "complete_task", "task complete", "task_complete", "任务完成"]:
            result["completed"] = True
            result["name"] = "complete_task" # 使用特定名称
            return result

        # 解析函数调用
        match = re.match(r'(\w+)\s*\((.*)\)\s*$', decision_text)
        if not match:
            # 检查是否是简单命令
            simple_commands = ["back", "home", "analyze_screen", "return_to_device_origin"]
            if decision_text.lower() in simple_commands:
                 result["name"] = decision_text.lower()
            else:
                result["error"] = f"无法解析函数调用或简单命令: '{decision_text}'"
                result["name"] = "unknown" # 标记为未知
            return result

        func_name = match.group(1).strip()
        args_str = match.group(2).strip()
        result["name"] = func_name
        args = []
        if args_str:
            try:
                # 使用更健壮的参数分割（处理带引号的字符串）
                arg_parts = re.split(r",\s*(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)(?=(?:[^']*'[^']*')*[^']*$)", args_str)
                for arg in arg_parts:
                    arg = arg.strip()
                    if not arg: continue # 跳过空参数（例如 "func(a, ,c)"）
                    if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                        args.append(arg[1:-1]) # 保留为字符串
                    else:
                        try: args.append(int(arg)) # 尝试整数
                        except ValueError:
                            try: args.append(float(arg)) # 尝试浮点数
                            except ValueError: args.append(arg) # 保留为原始字符串
            except Exception as parse_err:
                result["error"] = f"解析参数 '{args_str}' 出错: {parse_err}"
                logger.error(f"解析AI决策参数 '{args_str}' 出错: {parse_err}")
        result["args"] = args
        return result

    def _execute_parsed_action(self,
                               device: Device,
                               task: Task,
                               func_name: Optional[str],
                               args: List[Any],
                               justification: str,
                               text_positions: List[Dict], # 保留这些参数，以备将来使用或调试
                               matched_templates: List[Dict],
                               ai_prompt: Optional[str] = None,
                               ai_response_raw: Optional[str] = None
                               ) -> Dict[str, Any]:
        """
        执行由 AI 解析出的动作。处理人工干预，并调用包含锁和延时的原子动作函数。
        【已修改】将动作结果记录到 device.action_history。
        """
        if func_name is None:
            # 记录这个失败的尝试到历史
            device.add_action(
                {"action": "AI Action: None", "rationale": justification or "AI proposed no function"},
                action_result={"success": False, "error": "未提供函数名"}
            )
            return {"success": False, "error": "未提供函数名"}

        # --- 人工干预检查 (逻辑不变) ---
        intervention_enabled = self.main_ui.is_human_intervention_enabled()
        actions_requiring_intervention = ["click", "long_press", "swipe", "scroll", "back", "home", "esp_command", "return_to_device_origin"]
        if intervention_enabled and func_name in actions_requiring_intervention:
            logger.info(f"[{device.name}] 需要人工干预: {func_name}({args})")
            self._emit_progress(device.name, f"等待用户确认操作: {func_name}...")
            self.intervention_result = None
            intervention_data = {
                "action_str": f"{func_name}({', '.join(map(str, args))})",
                "justification": justification or "AI 未提供理由",
                "ai_prompt": ai_prompt or "无可用 Prompt 上下文",
                "ai_response_raw": ai_response_raw or "无可用 Response 上下文"
            }
            QMetaObject.invokeMethod(self, "_request_intervention_ui", Qt.QueuedConnection, Q_ARG(dict, intervention_data))

            wait_start_time = time.time(); intervention_timeout = 300
            while self.intervention_result is None and time.time() - wait_start_time < intervention_timeout:
                with self.task_scheduler.lock: # 确保线程安全访问任务状态
                    # 重新获取 task 对象，因为原始的 task 引用可能已过时
                    current_task_on_device = self.task_scheduler.running_tasks.get(device.name)
                    if current_task_on_device and current_task_on_device.status == TaskStatus.CANCELED:
                        logger.warning(f"[{device.name}] 在等待人工干预时任务 '{current_task_on_device.name}' 被取消。")
                        # 记录这个拒绝的尝试到历史
                        device.add_action(
                            {"action": f"AI Action (Rejected): {func_name}({args})", "rationale": "任务在干预等待期间被取消"},
                            action_result={"success": False, "error": "等待人工干预时任务取消", "user_rejected": True}
                        )
                        return {"success": False, "error": "等待人工干预时任务取消", "user_rejected": True}
                time.sleep(0.2)

            if self.intervention_result is None:
                logger.warning(f"[{device.name}] 人工干预超时 ({intervention_timeout}s)。"); self._emit_progress(device.name, "人工干预超时，操作取消。")
                device.add_action(
                    {"action": f"AI Action (Timeout): {func_name}({args})", "rationale": "人工干预超时"},
                    action_result={"success": False, "error": "人工干预超时", "user_rejected": True}
                )
                return {"success": False, "error": "人工干预超时", "user_rejected": True}
            elif not self.intervention_result:
                logger.info(f"[{device.name}] 用户拒绝执行操作: {func_name}({args})"); self._emit_progress(device.name, "用户已拒绝执行此操作。")
                device.add_action(
                    {"action": f"AI Action (Rejected): {func_name}({args})", "rationale": "用户拒绝"},
                    action_result={"success": False, "error": "用户拒绝执行此操作", "user_rejected": True}
                )
                return {"success": False, "error": "用户拒绝执行此操作", "user_rejected": True}
            else:
                logger.info(f"[{device.name}] 用户同意执行操作: {func_name}({args})"); self._emit_progress(device.name, "用户已同意，继续执行...")
        # --- 人工干预检查结束 ---

        # --- 执行实际操作 ---
        action_result_dict = {"success": False, "error": "函数未执行"} # 初始化为失败
        action_details_for_history = {"action": f"AI Action: {func_name}({args})", "rationale": justification}

        if func_name in self.available_functions:
            func_to_call = self.available_functions[func_name]
            logger.info(f"[{device.name}] 请求执行动作: {func_name}({', '.join(map(str, args))})")
            try:
                call_args = [device] + args
                action_result_dict = func_to_call(*call_args) # 这是实际执行动作并获取结果
                if not isinstance(action_result_dict, dict) or "success" not in action_result_dict:
                    logger.warning(f"函数 '{func_name}' 返回意外结果: {action_result_dict}。假定失败。")
                    action_result_dict = {"success": False, "error": f"函数 '{func_name}' 返回意外结果", "response": str(action_result_dict)}
            except TypeError as te:
                import inspect; sig = None; param_names = []
                try:
                    sig = inspect.signature(func_to_call); param_names = list(sig.parameters.keys())
                except ValueError: pass
                expected_args_msg = f"(预期参数: {param_names[1:] if len(param_names)>1 else '无'})"
                err_msg = f"调用 {func_name} 参数错误 {expected_args_msg}: {te}"
                logger.error(f"[{device.name}] {err_msg}，传入参数: {args}", exc_info=False)
                action_result_dict = {"success": False, "error": err_msg}
            except Exception as exec_err:
                err_msg = f"执行 {func_name} 时发生错误: {exec_err}"
                logger.error(f"[{device.name}] {err_msg}", exc_info=True)
                action_result_dict = {"success": False, "error": err_msg, "exception": True}
        else:
            logger.error(f"[{device.name}] AI 决定调用未知函数: '{func_name}'")
            action_result_dict = {"success": False, "error": f"AI 请求未知函数: {func_name}"}
            action_details_for_history["action"] = f"AI Action (Unknown): {func_name}({args})"


        # 将动作及其结果添加到历史记录
        device.add_action(action_details_for_history, action_result=action_result_dict)
        return action_result_dict

    @pyqtSlot(dict)
    def _request_intervention_ui(self, intervention_data: dict):
        """槽函数：在主线程中触发人工干预对话框。"""
        # 实际的对话框显示由 MainUI 的槽函数处理
        self.request_human_intervention.emit(intervention_data)

    # --- Subtask Dispatcher and Handlers (基本保持不变, 确保调用原子操作) ---
    def _dispatch_subtask(self, device: Device, task: Task, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据子任务类型调用相应的处理函数。
        【已修改】处理 COMMENT 类型，动作结果传递给 device.add_action。
        """
        subtask_type_str = subtask.get("type")
        current_index = task.current_subtask_index

        try:
            st_type = SubtaskType(subtask_type_str)
        except ValueError:
            err_msg = f"未知子任务类型: {subtask_type_str}"
            logger.error(f"任务 '{task.name}' 中包含未知子任务类型 '{subtask_type_str}' (索引 {current_index})")
            device.add_action({"action": f"Subtask Error: Unknown Type '{subtask_type_str}'", "rationale": "Invalid subtask definition"},
                              action_result={"success": False, "error": err_msg})
            return {"success": False, "error": err_msg, "force_fail": True}

        # --- 处理循环标记 (逻辑不变) ---
        if st_type == SubtaskType.LOOP_START:
            loop_count = subtask.get("count", 1)
            try: loop_count = int(loop_count)
            except ValueError: loop_count = 1
            max_loops = CONFIG.get("MAX_LOOP_ITERATIONS", 1000)
            if loop_count <= 0 or loop_count > max_loops:
                 logger.warning(f"任务 '{task.name}' 的 LOOP_START (索引 {current_index}) 次数无效 ({loop_count})，限制为 1 到 {max_loops}。使用 1 次。")
                 loop_count = 1; subtask['count'] = 1

            if current_index not in task.loop_counters:
                task.loop_stack.append(current_index)
                task.loop_counters[current_index] = 0
                logger.info(f"任务 '{task.name}': 进入循环 (索引 {current_index}, 总次数 {loop_count}, 当前第 {task.loop_counters[current_index] + 1} 次)")
            task.current_subtask_index += 1
            # LOOP_START 本身不记录到 action_history，因为它不是一个 "动作"
            return {"success": True, "skipped": True, "reason": "Processed LOOP_START"}

        elif st_type == SubtaskType.LOOP_END:
            if not task.loop_stack:
                err_msg = "循环结构错误: 未找到匹配的 LOOP_START"
                logger.error(f"任务 '{task.name}': 遇到 LOOP_END (索引 {current_index}) 但循环堆栈为空！{err_msg}")
                device.add_action({"action": "Subtask Error: LOOP_END without LOOP_START", "rationale": err_msg},
                                  action_result={"success": False, "error": err_msg})
                return {"success": False, "error": err_msg, "force_fail": True}

            start_index = task.loop_stack[-1]
            start_subtask = task.subtasks[start_index] if 0 <= start_index < len(task.subtasks) else None
            if not start_subtask or start_subtask.get('type') != SubtaskType.LOOP_START.value:
                 err_msg = "循环结构错误: LOOP_START 类型不匹配或丢失"
                 logger.error(f"任务 '{task.name}': LOOP_END (索引 {current_index}) 对应的 LOOP_START (索引 {start_index}) 类型错误或丢失！{err_msg}")
                 task.loop_stack.pop()
                 device.add_action({"action": "Subtask Error: LOOP_START Mismatch", "rationale": err_msg},
                                   action_result={"success": False, "error": err_msg})
                 return {"success": False, "error": err_msg, "force_fail": True}

            loop_count = start_subtask.get("count", 1); # ... (validation as above) ...
            task.loop_counters[start_index] += 1
            current_iteration = task.loop_counters[start_index]
            logger.info(f"任务 '{task.name}': 到达循环结束标记 (索引 {current_index})，对应开始 {start_index}。当前完成第 {current_iteration}/{loop_count} 次迭代。")
            if current_iteration < loop_count:
                task.current_subtask_index = start_index + 1
                logger.info(f"任务 '{task.name}': 循环继续，跳转回子任务索引 {task.current_subtask_index}")
            else:
                logger.info(f"任务 '{task.name}': 循环 (开始于 {start_index}) 已完成 {loop_count} 次，退出循环。")
                task.loop_stack.pop(); del task.loop_counters[start_index]
                task.current_subtask_index += 1
            # LOOP_END 也不记录到 action_history
            return {"success": True, "skipped": True, "reason": "Processed LOOP_END"}

        # --- 新增：处理 COMMENT 类型 ---
        elif st_type == SubtaskType.COMMENT:
            comment_text = subtask.get('text', subtask.get('description', '(空注释)'))
            logger.info(f"任务 '{task.name}' Subtask Comment (索引 {current_index}): {comment_text}")
            # 注释子任务总是成功，并且可以被记录到历史中，但不算作一个有 "结果" 的动作
            device.add_action({"action": f"Subtask: Comment - {comment_text}", "rationale": "User-defined comment"})
            # 它不执行物理动作，直接成功并前进
            # _handle_step_result 会处理索引递增
            return {"success": True} # 直接返回成功

        # --- 处理其他普通子任务 ---
        handler_map = {
            SubtaskType.WAIT: self._handle_subtask_wait,
            SubtaskType.FIND_AND_CLICK_TEXT: self._handle_subtask_find_click_text,
            SubtaskType.TEMPLATE_CLICK: self._handle_subtask_template_click,
            SubtaskType.SWIPE: self._handle_subtask_swipe,
            SubtaskType.BACK: self._handle_subtask_back,
            SubtaskType.AI_STEP: self._handle_subtask_ai_step, # AI_STEP 内部会调用 _execute_parsed_action
            SubtaskType.ESP_COMMAND: self._handle_subtask_esp_command,
            SubtaskType.CHECK_TEXT_EXISTS: self._handle_subtask_check_text, # 返回 condition_met
            SubtaskType.CHECK_TEMPLATE_EXISTS: self._handle_subtask_check_template, # 返回 condition_met
        }
        handler = handler_map.get(st_type)
        action_result_dict = {"success": False, "error": "未找到处理器"} # 初始化

        if handler:
            call_args = [device, task, subtask] if st_type == SubtaskType.AI_STEP else [device, subtask]
            try:
                action_result_dict = handler(*call_args) # 执行子任务处理器
                if not isinstance(action_result_dict, dict) or "success" not in action_result_dict:
                    logger.warning(f"子任务类型 '{st_type.value}' (索引 {current_index}) 的处理器返回了意外结果: {action_result_dict}。假定失败。")
                    action_result_dict = {"success": False, "error": f"处理器返回意外结果: {action_result_dict}"}

            except Exception as handler_err:
                logger.error(f"执行子任务类型 '{st_type.value}' (索引 {current_index}) 的处理器时出错: {handler_err}", exc_info=True)
                action_result_dict = {"success": False, "error": f"处理器异常: {handler_err}", "exception": True}
        else:
            logger.error(f"未实现子任务类型 '{st_type.value}' (索引 {current_index}) 的处理器。")
            action_result_dict = {"success": False, "error": f"未实现处理器: {st_type.value}", "force_fail": True}

        # --- 记录子任务动作及其结果 (AI_STEP 除外，它内部已记录) ---
        if st_type != SubtaskType.AI_STEP: # AI_STEP 在 _execute_parsed_action 中记录
            action_name_for_history = subtask.get('description', st_type.value)
            # 为检查类子任务添加更清晰的动作历史条目
            if st_type in [SubtaskType.CHECK_TEXT_EXISTS, SubtaskType.CHECK_TEMPLATE_EXISTS]:
                target_val = subtask.get('target_text') or subtask.get('template_name')
                condition_met_val = action_result_dict.get('condition_met', 'N/A')
                action_name_for_history = f"Check '{target_val}': Found = {condition_met_val}"

            device.add_action(
                {"action": f"Subtask: {action_name_for_history}", "rationale": subtask.get('description', '')},
                action_result=action_result_dict
            )
        return action_result_dict # 返回给 _handle_step_result

    def _handle_subtask_wait(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 wait 子任务，返回 waiting 标记给调度器。"""
        duration = subtask.get("duration", 1.0)
        try:
            wait_seconds = float(duration)
            if wait_seconds > 0:
                # 不直接 time.sleep()，而是返回标记
                return {"success": True, "waiting": True, "wait_duration": wait_seconds}
            else:
                return {"success": False, "error": "等待时长必须为正数"}
        except ValueError:
            return {"success": False, "error": f"无效的等待时长: {duration}"}

    def _get_current_screen_context(self, device: Device) -> Tuple[Optional[np.ndarray], Optional[Dict], List[Dict], List[Dict]]:
        """辅助函数：获取截图、OCR结果和模板匹配结果。"""
        # 先尝试返回设备原点（原子操作）
        # 注意：频繁调用此函数可能会因返回原点而变慢
        # origin_result = self._return_to_device_origin(device)
        # if not origin_result.get("success"):
        #     logger.warning(f"[{device.name}] 获取屏幕上下文前返回原点失败: {origin_result.get('error')}")

        screenshot = self.screenshot_manager.take_screenshot(device)
        if screenshot is None: return None, None, [], []
        enhanced = self.screenshot_manager.enhance_image(screenshot); device.last_screenshot = enhanced # 保存增强后的图
        ocr_result = self.ai_analyzer.perform_ocr(enhanced); device.last_ocr_result = ocr_result
        templates = self.ai_analyzer.match_all_templates(enhanced)
        text_positions = []
        if ocr_result and ocr_result.get("code") == 100 and "data" in ocr_result:
            for item in ocr_result["data"]: # 这里的 item 是经过 _filter_ocr_results 处理的
                if isinstance(item, dict) and "text" in item and "box" in item and len(item["box"]) == 4:
                    try:
                        center_x = sum(p[0] for p in item["box"]) // 4; center_y = sum(p[1] for p in item["box"]) // 4
                        # 修改：从 item.get("score") 获取置信度，并存到 "confidence" 键
                        # 提供默认值 0.0，以防 "score" 意外缺失
                        confidence_score = item.get("score", 0.0)
                        text_positions.append({
                            "text": item["text"],
                            "center": (center_x, center_y),
                            "box": item["box"],
                            "confidence": confidence_score # 将 score 映射到 confidence
                        })
                    except Exception as e_parse: # 更具体的异常捕获
                        logger.warning(f"解析OCR item 时出错: {e_parse}, item: {item}")
                        pass # 忽略计算错误
        return enhanced, ocr_result, text_positions, templates

    def _find_text_on_screen(self, device: Device, target_text: str, use_partial_match: bool = True) -> Optional[Tuple[int, int]]:
        """查找文本中心坐标，如果未找到则返回 None。(保持不变)"""
        _, _, text_positions, _ = self._get_current_screen_context(device) # 获取最新屏幕信息
        if not text_positions: return None
        target_text_norm = target_text.strip()
        if not target_text_norm: return None

        best_match = None # 用于部分匹配时寻找最佳匹配
        for item in text_positions:
            text_on_screen_norm = item.get("text", "").strip()
            if not text_on_screen_norm: continue

            match = False
            if use_partial_match:
                if target_text_norm in text_on_screen_norm: match = True
            else:
                if target_text_norm == text_on_screen_norm: match = True

            if match:
                # 精确匹配优先返回
                if not use_partial_match:
                     logger.info(f"[{device.name}] 精确找到文本 '{target_text}' at {item['center']}")
                     return item['center']
                # 部分匹配，记录第一个找到的，或者可以加入更复杂的评分逻辑
                if best_match is None: best_match = item
        # 循环结束后处理部分匹配结果
        if best_match:
            logger.info(f"[{device.name}] 部分匹配找到文本 '{target_text}' within '{best_match['text']}' at {best_match['center']}")
            return best_match['center']

        logger.debug(f"[{device.name}] 文本 '{target_text}' 未在屏幕上找到。")
        return None

    def _find_template_on_screen(self, device: Device, template_name: str, threshold: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """查找模板中心坐标，如果未找到则返回 None。(保持不变)"""
        screenshot, _, _, _ = self._get_current_screen_context(device) # 获取最新屏幕信息
        if screenshot is None: return None
        match_result = self.ai_analyzer.template_matching(screenshot, template_name=template_name, threshold=threshold)
        if match_result.get("match"): logger.info(f"[{device.name}] 找到模板 '{template_name}' at {match_result['center']}"); return match_result['center']
        else:
            conf = match_result.get('confidence')
            err = match_result.get('error')
            log_extra = f"(最高置信度: {conf:.3f})" if conf is not None else f"(错误: {err})" if err else ""
            logger.debug(f"[{device.name}] 模板 '{template_name}' 未在屏幕上找到 {log_extra}。")
            return None


    def _handle_subtask_find_click_text(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 find_and_click_text 子任务。"""
        target_text = subtask.get("target_text")
        if not target_text: return {"success": False, "error": "缺少 'target_text' 参数"}
        timeout = subtask.get("timeout", 10.0)
        partial_match = subtask.get("partial_match", True)
        max_attempts = subtask.get("attempts", 3) # 增加尝试次数参数

        for attempt in range(max_attempts):
            # --- 查找前确保在设备原点 (原子操作) ---
            origin_result = self._return_to_device_origin(device)
            if not origin_result.get("success"): logger.warning(f"[{device.name}] Subtask FindClickText: 查找前返回原点失败。")

            logger.debug(f"[{device.name}] 子任务 FindClickText: 尝试 {attempt + 1}/{max_attempts} 查找 '{target_text}'...")
            coords = self._find_text_on_screen(device, target_text, partial_match)
            if coords:
                logger.info(f"[{device.name}] 子任务: 找到文本 '{target_text}' at {coords}, 调用 click...")
                # --- 调用 click (原子操作) ---
                click_result = self._execute_click(device, coords[0], coords[1])
                device.add_action({"action": f"Subtask: Click Text '{target_text}' at {coords}", "rationale": subtask.get('description')})
                return click_result # 直接返回 click 的结果

            if attempt < max_attempts - 1:
                logger.debug(f"[{device.name}] Subtask FindClickText: 未找到 '{target_text}', 等待 1 秒后重试...")
                time.sleep(1.0) # 短暂等待
            # 检查任务是否已被取消 (如果需要更及时的响应)
            # with self.task_scheduler.lock:
            #     task = self.task_scheduler.running_tasks.get(device.name)
            #     if task and task.status == TaskStatus.CANCELED:
            #         return {"success": False, "error": "在查找文本时任务被取消"}

        logger.warning(f"[{device.name}] 子任务 find_and_click_text: 尝试 {max_attempts} 次后未找到文本 '{target_text}'")
        # --- 失败后尝试返回设备原点 ---
        self._return_to_device_origin(device)
        return {"success": False, "error": f"尝试 {max_attempts} 次后未找到文本 '{target_text}'"}

    def _handle_subtask_template_click(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 template_click 子任务。"""
        template_name = subtask.get("template_name")
        if not template_name: return {"success": False, "error": "缺少 'template_name' 参数"}
        threshold = subtask.get("threshold")
        max_attempts = subtask.get("attempts", 3) # 增加尝试次数参数

        for attempt in range(max_attempts):
            # --- 查找前确保在设备原点 (原子操作) ---
            origin_result = self._return_to_device_origin(device)
            if not origin_result.get("success"): logger.warning(f"[{device.name}] Subtask TemplateClick: 查找前返回原点失败。")

            logger.debug(f"[{device.name}] 子任务 TemplateClick: 尝试 {attempt + 1}/{max_attempts} 查找 '{template_name}'...")
            coords = self._find_template_on_screen(device, template_name, threshold)
            if coords:
                logger.info(f"[{device.name}] 子任务: 找到模板 '{template_name}' at {coords}, 调用 click...")
                # --- 调用 click (原子操作) ---
                click_result = self._execute_click(device, coords[0], coords[1])
                device.add_action({"action": f"Subtask: Click Template '{template_name}' at {coords}", "rationale": subtask.get('description')})
                return click_result

            if attempt < max_attempts - 1:
                logger.debug(f"[{device.name}] Subtask TemplateClick: 未找到 '{template_name}', 等待 0.5 秒后重试...")
                time.sleep(0.5)

        logger.warning(f"[{device.name}] 子任务 template_click: 尝试 {max_attempts} 次后未找到模板 '{template_name}'")
        self._return_to_device_origin(device)
        return {"success": False, "error": f"尝试 {max_attempts} 次后未找到模板 '{template_name}'"}

    def _handle_subtask_swipe(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 swipe 子任务 (包括 scroll)。"""
        start = subtask.get("start"); end = subtask.get("end")
        direction = subtask.get("direction")
        duration = subtask.get("duration", 500)
        action_desc = ""
        swipe_result = {"success": False, "error": "Swipe 参数无效"}

        if direction: # 处理滚动
            action_desc = f"Scroll {direction}"
            # --- 调用 _execute_scroll (原子操作) ---
            swipe_result = self._execute_scroll(device, direction, duration)
        elif start and end and len(start) == 2 and len(end) == 2: # 处理坐标滑动
            action_desc = f"Swipe from {start} to {end}"
            # --- 调用 _execute_swipe (原子操作) ---
            swipe_result = self._execute_swipe(device, start[0], start[1], end[0], end[1], duration)
        else:
            self._return_to_device_origin(device) # 确保返回原点
            return {"success": False, "error": "Swipe 子任务需要 'direction' 或有效的 'start'/'end' 像素坐标"}

        if swipe_result.get("success"):
            device.add_action({"action": f"Subtask: {action_desc}", "rationale": subtask.get('description')})
        return swipe_result #直接返回 swipe/scroll 的结果

    def _handle_subtask_back(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 back 子任务。"""
        logger.info(f"[{device.name}] 子任务: 执行返回操作...")
        # --- 调用 _execute_back (原子操作) ---
        back_result = self._execute_back(device)
        if back_result.get("success"):
            device.add_action({"action": "Subtask: Back", "rationale": subtask.get('description')})
        return back_result # 直接返回 back 的结果

    def _handle_subtask_ai_step(self, device: Device, task: Task, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 ai_step 子任务。"""
        goal = subtask.get("goal", "完成下一个逻辑步骤")
        logger.info(f"[{device.name}] 子任务: 执行 AI 步骤。目标: {goal}")

        # --- 1. 确保在设备原点并获取屏幕上下文 (原子操作 + 截图/分析) ---
        origin_result = self._return_to_device_origin(device)
        if not origin_result.get("success"): logger.warning(f"[{device.name}] Subtask AI Step: 准备前返回原点失败。")
        screenshot, ocr_result, text_positions, matched_templates = self._get_current_screen_context(device)
        if screenshot is None: return {"success": False, "error": "AI 步骤获取截图失败"}

        # --- 2. 构建 AI Prompt (修改 Prompt 指令) ---
        screen_text = "";
        if ocr_result and ocr_result.get("code") == 100: screen_text = "\n".join([item.get("text", "") for item in ocr_result.get("data",[])])
        context_data = self._build_ai_context(device, task, screen_text, text_positions, matched_templates)
        # 修改 Prompt，强调当前子任务目标
        ai_prompt = context_data["prompt"]
        ai_prompt = ai_prompt.replace("## Your Task ##", f"## 当前子任务目标 ##\n{goal}\n\n## Your Task ##", 1)

        # --- 3. 获取 AI 决策 ---
        ai_response = self.ai_analyzer.get_ai_decision(ai_prompt, device.action_history, f"{task.name} - 子任务 AI: {goal}")
        if not ai_response or not ai_response.get("decision"):
            err = ai_response.get("error", "无响应") if ai_response else "无响应"
            logger.error(f"[{device.name}] 子任务 AI 步骤失败: 未收到 AI 决策。错误: {err}")
            return {"success": False, "error": f"AI 步骤未能获取决策: {err}"}

        # --- 4. 解析 AI 决策 ---
        parsed_action = self._parse_ai_decision_only(ai_response["decision"])
        action_name = parsed_action.get("name"); action_args = parsed_action.get("args", [])
        raw_ai_response_str = json.dumps(ai_response.get("raw_response", {}), indent=2, ensure_ascii=False)

        # --- 5. 执行 AI 决策 (调用原子操作或返回等待标记) ---
        action_result = self._execute_parsed_action(
             device=device, task=task, func_name=action_name, args=action_args,
             justification=ai_response.get("justification", f"AI 步骤: {goal}"),
             text_positions=text_positions, matched_templates=matched_templates,
             ai_prompt=ai_prompt, ai_response_raw=raw_ai_response_str
        )

        return action_result # 直接返回 AI 动作的执行结果

    def _handle_subtask_esp_command(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """处理 esp_command 子任务。"""
        command = subtask.get("command_string")
        if not command:
             self._return_to_device_origin(device) # 确保返回原点
             return {"success": False, "error": "缺少 'command_string' 参数"}

        logger.info(f"[{device.name}] 子任务: 执行 ESP 命令 '{command}'...")
        # --- 调用 _execute_esp_command (原子操作) ---
        result = self._execute_esp_command(device, command)
        if result.get("success"):
            device.add_action({"action": f"Subtask: ESP Command '{command}'", "rationale": subtask.get('description')})
        return result #直接返回 esp_command 的结果

    def _handle_subtask_check_text(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 check_text_exists 子任务。
        【已修改】返回 `condition_met` 字段。
        """
        target_text = subtask.get("target_text");
        if not target_text:
            return {"success": False, "error": "缺少 'target_text' 参数 (check_text)"} # 检查本身失败
        expected_result_ignored_for_condition = subtask.get("expected_result", True) # 这个参数现在仅用于用户理解，不影响condition_met
        timeout = subtask.get("timeout", 5.0); partial_match = subtask.get("partial_match", True)
        start_time = time.time(); found_on_screen = False

        origin_result = self._return_to_device_origin(device)
        if not origin_result.get("success"): logger.warning(f"[{device.name}] CheckText: 检查前返回原点失败。")

        try:
            while time.time() - start_time < timeout:
                coords = self._find_text_on_screen(device, target_text, partial_match)
                if coords is not None: found_on_screen = True; break
                logger.debug(f"[{device.name}] CheckText: 未找到 '{target_text}', 等待 0.5s")
                time.sleep(0.5)
                # 可以添加任务取消检查
                # ...

            logger.info(f"[{device.name}] 子任务 check_text: 检查文本 '{target_text}' 完成，实际找到: {found_on_screen}。")
            # "success" 表示检查操作本身是否成功完成（如截图、OCR）
            # "condition_met" 表示文本是否在屏幕上找到
            return {"success": True, "condition_met": found_on_screen}
        except Exception as e:
            logger.error(f"[{device.name}] CheckText 执行时出错: {e}", exc_info=True)
            return {"success": False, "error": f"CheckText 异常: {e}", "exception": True} # 检查本身失败

    def _handle_subtask_check_template(self, device: Device, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 check_template_exists 子任务。
        【已修改】返回 `condition_met` 字段。
        """
        template_name = subtask.get("template_name");
        if not template_name:
            return {"success": False, "error": "缺少 'template_name' 参数 (check_template)"} # 检查本身失败
        expected_result_ignored_for_condition = subtask.get("expected_result", True)
        threshold = subtask.get("threshold"); timeout = subtask.get("timeout", 5.0)
        start_time = time.time(); found_on_screen = False

        origin_result = self._return_to_device_origin(device)
        if not origin_result.get("success"): logger.warning(f"[{device.name}] CheckTemplate: 检查前返回原点失败。")

        try:
            while time.time() - start_time < timeout:
                coords = self._find_template_on_screen(device, template_name, threshold)
                if coords is not None: found_on_screen = True; break
                logger.debug(f"[{device.name}] CheckTemplate: 未找到 '{template_name}', 等待 0.5s")
                time.sleep(0.5)
                # 可以添加任务取消检查
                # ...

            logger.info(f"[{device.name}] 子任务 check_template: 检查模板 '{template_name}' 完成，实际找到: {found_on_screen}。")
            return {"success": True, "condition_met": found_on_screen}
        except Exception as e:
            logger.error(f"[{device.name}] CheckTemplate 执行时出错: {e}", exc_info=True)
            return {"success": False, "error": f"CheckTemplate 异常: {e}", "exception": True} # 检查本身失败

    # --- 原子物理动作函数 ---
    def _execute_click(self, device: Device, x: int, y: int) -> Dict[str, Any]:
        """
        【原子操作 V5】执行点击操作。使用新的坐标转换并进行发送前钳位。
        """
        # 1. 应用设备内部微调（如果需要）
        adj_x, adj_y = self._apply_coordinate_transform(device, x, y)

        # 2. 像素坐标转换为指令坐标
        try:
            # 使用新的转换函数，获取机台指令坐标
            commanded_mx, commanded_my = self._pixel_to_commanded_machine_coords(adj_x, adj_y, device)
        except Exception as coord_err:
            logger.error(f"[{device.name}] 点击: 像素到指令坐标转换失败: {coord_err}", exc_info=True)
            return {"success": False, "error": f"坐标转换失败: {coord_err}"}

        # 3. 钳位指令坐标 (确保发送给 ESP 的是非负值)
        clamped_commanded_mx = max(0.0, commanded_mx)
        clamped_commanded_my = max(0.0, commanded_my)

        # 记录详细的转换和钳位日志
        log_prefix = (f"[{device.name}] 点击计算 (V5): Px({x},{y})->Adj({adj_x},{adj_y})"
                      f"->Cmd({commanded_mx:.2f},{commanded_my:.2f})")
        if clamped_commanded_mx != commanded_mx or clamped_commanded_my != commanded_my:
            logger.warning(f"{log_prefix} -> **钳位后** Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm")
        else:
            logger.debug(f"{log_prefix} -> Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm (无需钳位)")

        # 4. 获取锁并执行物理动作
        result = {"success": False, "error": "锁获取失败"}
        if not self.esp_action_lock.acquire(timeout=10):
            logger.error(f"[{device.name}] 获取 ESP 动作锁超时 (执行 click)。")
            return result
        try:
            # 使用钳位后的指令坐标
            logger.info(f"[{device.name}] [Lock Acquired] 执行点击: 发送钳位后指令坐标 Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm...")
            # 注意：esp_controller.click 接收的是最终的物理坐标
            click_result = self.esp_controller.click(clamped_commanded_mx, clamped_commanded_my)

            if click_result.get("success"):
                logger.debug(f"[{device.name}] 点击命令成功。等待内部延时 {self.INTERNAL_ACTION_DELAY_SECONDS}s...")
                time.sleep(self.INTERNAL_ACTION_DELAY_SECONDS)
                logger.debug(f"[{device.name}] 准备返回设备原点...")
                return_result = self._internal_return_to_device_origin(device) # 返回设备配置原点
                if return_result.get("success"):
                    logger.debug(f"[{device.name}] 返回设备原点成功。等待外部延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS}s...")
                    time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS)
                    logger.debug(f"[{device.name}] 点击序列完成。")
                    result = {"success": True}
                else:
                    error_msg = f"点击成功但返回设备原点失败: {return_result.get('error')}"
                    logger.error(f"[{device.name}] {error_msg}")
                    result = {"success": False, "error": error_msg}
            else:
                error_msg = f"点击命令失败: {click_result.get('error')}"
                logger.error(f"[{device.name}] {error_msg}")
                # 特别检查范围错误，提示用户检查固件或坐标系
                if "OUT_OF_RANGE" in error_msg.upper():
                    logger.critical(f"[{device.name}] 点击命令报告范围错误，即使指令坐标已钳位为 ({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f})！请检查 ESP 固件的坐标限制或坐标转换/偏移设置！")
                result = {"success": False, "error": error_msg}
            logger.info(f"[{device.name}] 完成点击序列处理。")
        finally:
            logger.debug(f"[{device.name}] [Lock Releasing] click")
            self.esp_action_lock.release()
        return result

    # 修改 _execute_long_press
    def _execute_long_press(self, device: Device, x: int, y: int, duration_ms: int = 1000) -> Dict[str, Any]:
        """
        【原子操作 V5】执行长按操作。使用新的坐标转换并进行发送前钳位。
        """
        # 1. 应用设备内部微调
        adj_x, adj_y = self._apply_coordinate_transform(device, x, y)

        # 2. 像素坐标转换为指令坐标
        try:
            commanded_mx, commanded_my = self._pixel_to_commanded_machine_coords(adj_x, adj_y, device)
        except Exception as coord_err:
            logger.error(f"[{device.name}] 长按: 像素到指令坐标转换失败: {coord_err}", exc_info=True)
            return {"success": False, "error": f"坐标转换失败: {coord_err}"}

        # 3. 钳位指令坐标
        clamped_commanded_mx = max(0.0, commanded_mx)
        clamped_commanded_my = max(0.0, commanded_my)

        log_prefix = (f"[{device.name}] 长按计算 (V5): Px({x},{y})->Adj({adj_x},{adj_y})"
                      f"->Cmd({commanded_mx:.2f},{commanded_my:.2f})")
        if clamped_commanded_mx != commanded_mx or clamped_commanded_my != commanded_my:
            logger.warning(f"{log_prefix} -> **钳位后** Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm")
        else:
            logger.debug(f"{log_prefix} -> Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm (无需钳位)")

        # 4. 获取锁并执行物理动作
        result = {"success": False, "error": "锁获取失败"}
        if not self.esp_action_lock.acquire(timeout=10):
            logger.error(f"[{device.name}] 获取 ESP 动作锁超时 (执行 long_press)。")
            return result
        try:
            logger.info(f"[{device.name}] [Lock Acquired] 执行长按: 发送钳位后指令坐标 Cmd=({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f}) mm...")
            # ESP long_press 接收物理坐标
            lp_result = self.esp_controller.long_press(clamped_commanded_mx, clamped_commanded_my, duration_ms)

            if lp_result.get("success"):
                logger.debug(f"[{device.name}] 长按命令成功。等待内部延时 {self.INTERNAL_ACTION_DELAY_SECONDS}s...")
                time.sleep(self.INTERNAL_ACTION_DELAY_SECONDS)
                logger.debug(f"[{device.name}] 准备返回设备原点...")
                return_result = self._internal_return_to_device_origin(device) # 返回设备配置原点
                if return_result.get("success"):
                    logger.debug(f"[{device.name}] 返回设备原点成功。等待外部延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS}s...")
                    time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS)
                    logger.debug(f"[{device.name}] 长按序列完成。")
                    result = {"success": True}
                else:
                    error_msg = f"长按成功但返回设备原点失败: {return_result.get('error')}"
                    logger.error(f"[{device.name}] {error_msg}")
                    result = {"success": False, "error": error_msg}
            else:
                error_msg = f"长按命令失败: {lp_result.get('error')}"
                logger.error(f"[{device.name}] {error_msg}")
                if "OUT_OF_RANGE" in error_msg.upper():
                    logger.critical(f"[{device.name}] 长按命令报告范围错误，即使指令坐标已钳位为 ({clamped_commanded_mx:.2f},{clamped_commanded_my:.2f})！请检查 ESP 固件限制或坐标转换/偏移！")
                result = {"success": False, "error": error_msg}
            logger.info(f"[{device.name}] 完成长按序列处理。")
        finally:
            logger.debug(f"[{device.name}] [Lock Releasing] long_press")
            self.esp_action_lock.release()
        return result

    # 修改 _execute_swipe
    def _execute_swipe(self, device: Device, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 500) -> Dict[str, Any]:
        """
        【原子操作 V5】执行滑动操作。使用新的坐标转换并进行发送前钳位。
        """
        # 1. 应用设备内部微调（起点和终点）
        adj_start_x, adj_start_y = self._apply_coordinate_transform(device, start_x, start_y)
        adj_end_x, adj_end_y = self._apply_coordinate_transform(device, end_x, end_y)

        # 2. 像素坐标转换为指令坐标（起点和终点）
        try:
            commanded_sx, commanded_sy = self._pixel_to_commanded_machine_coords(adj_start_x, adj_start_y, device)
            commanded_ex, commanded_ey = self._pixel_to_commanded_machine_coords(adj_end_x, adj_end_y, device)
        except Exception as coord_err:
            logger.error(f"[{device.name}] 滑动: 像素到指令坐标转换失败: {coord_err}", exc_info=True)
            return {"success": False, "error": f"坐标转换失败: {coord_err}"}

        # 3. 钳位指令坐标（起点和终点）
        clamped_sx = max(0.0, commanded_sx)
        clamped_sy = max(0.0, commanded_sy)
        clamped_ex = max(0.0, commanded_ex)
        clamped_ey = max(0.0, commanded_ey)

        # 记录详细的转换和钳位日志
        log_prefix_start = (f"StartPx({start_x},{start_y})->Adj({adj_start_x},{adj_start_y})"
                            f"->Cmd({commanded_sx:.2f},{commanded_sy:.2f})")
        log_prefix_end = (f"EndPx({end_x},{end_y})->Adj({adj_end_x},{adj_end_y})"
                          f"->Cmd({commanded_ex:.2f},{commanded_ey:.2f})")
        clamp_log = ""
        if clamped_sx != commanded_sx or clamped_sy != commanded_sy:
            clamp_log += f" Start->Clamped({clamped_sx:.2f},{clamped_sy:.2f})"
        if clamped_ex != commanded_ex or clamped_ey != commanded_ey:
            clamp_log += f" End->Clamped({clamped_ex:.2f},{clamped_ey:.2f})"

        if clamp_log:
             logger.warning(f"[{device.name}] 滑动计算 (V5): {log_prefix_start} | {log_prefix_end} ->**钳位后**{clamp_log} mm")
        else:
             logger.debug(f"[{device.name}] 滑动计算 (V5): {log_prefix_start} | {log_prefix_end} -> CmdStart({clamped_sx:.2f},{clamped_sy:.2f}) CmdEnd({clamped_ex:.2f},{clamped_ey:.2f}) mm (无需钳位)")


        # 4. 获取锁并执行物理动作
        result = {"success": False, "error": "锁获取失败"}
        if not self.esp_action_lock.acquire(timeout=10):
            logger.error(f"[{device.name}] 获取 ESP 动作锁超时 (执行 swipe)。")
            return result
        try:
            logger.info(f"[{device.name}] [Lock Acquired] 执行滑动: 从钳位后指令 CmdStart({clamped_sx:.2f},{clamped_sy:.2f}) 到 CmdEnd({clamped_ex:.2f},{clamped_ey:.2f}) mm...")
            # ESP swipe 接收物理坐标
            swipe_result = self.esp_controller.swipe(clamped_sx, clamped_sy, clamped_ex, clamped_ey, duration_ms)

            if swipe_result.get("success"):
                logger.debug(f"[{device.name}] 滑动命令成功。等待内部延时 {self.INTERNAL_ACTION_DELAY_SECONDS}s...")
                time.sleep(self.INTERNAL_ACTION_DELAY_SECONDS)
                logger.debug(f"[{device.name}] 准备返回设备原点...")
                return_result = self._internal_return_to_device_origin(device) # 返回设备配置原点
                if return_result.get("success"):
                    logger.debug(f"[{device.name}] 返回设备原点成功。等待外部延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS}s...")
                    time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS)
                    logger.debug(f"[{device.name}] 滑动序列完成。")
                    result = {"success": True}
                else:
                    error_msg = f"滑动成功但返回设备原点失败: {return_result.get('error')}"
                    logger.error(f"[{device.name}] {error_msg}")
                    result = {"success": False, "error": error_msg}
            else:
                error_msg = f"滑动命令失败: {swipe_result.get('error')}"
                logger.error(f"[{device.name}] {error_msg}")
                if "OUT_OF_RANGE" in error_msg.upper():
                    logger.critical(f"[{device.name}] 滑动命令报告范围错误，即使指令坐标已钳位！请检查 ESP 固件限制或坐标转换/偏移！Start:({clamped_sx:.2f},{clamped_sy:.2f}) End:({clamped_ex:.2f},{clamped_ey:.2f})")
                result = {"success": False, "error": error_msg}
            logger.info(f"[{device.name}] 完成滑动序列处理。")
        finally:
            logger.debug(f"[{device.name}] [Lock Releasing] swipe")
            self.esp_action_lock.release()
        return result

    def _execute_scroll(self, device: Device, direction: str, duration_ms: int = 500) -> Dict[str, Any]:
        """
        【原子操作】执行滚动操作（通过计算像素坐标调用 _execute_swipe 实现）。
        """
        # --- 计算起点终点像素坐标 (锁外) ---
        crop_w, crop_h = device.get_config("CROPPED_RESOLUTION", CONFIG["CROPPED_RESOLUTION"])
        center_x, center_y = crop_w // 2, crop_h // 2
        dist_y = int(crop_h * 0.30) # 增加垂直滚动距离
        dist_x = int(crop_w * 0.35) # 增加水平滚动距离
        start_x, start_y, end_x, end_y = 0, 0, 0, 0
        direction = direction.lower()

        if direction == "down":   start_x, start_y, end_x, end_y = center_x, center_y + dist_y, center_x, center_y - dist_y
        elif direction == "up": start_x, start_y, end_x, end_y = center_x, center_y - dist_y, center_x, center_y + dist_y
        elif direction == "left":  start_x, start_y, end_x, end_y = center_x + dist_x, center_y, center_x - dist_x, center_y
        elif direction == "right": start_x, start_y, end_x, end_y = center_x - dist_x, center_y, center_x + dist_x, center_y
        else: return {"success": False, "error": f"无效的滚动方向: {direction}"}

        logger.info(f"[{device.name}] 执行滚动 '{direction}' (调用 _execute_swipe)...")
        # --- 调用 _execute_swipe (原子操作) ---
        return self._execute_swipe(device, start_x, start_y, end_x, end_y, duration_ms)

    def _execute_wait(self, device: Device, seconds: Union[int, float]) -> Dict[str, Any]:
        """处理 wait 动作（返回等待标记给调度器）。"""
        try:
            wait_seconds = float(seconds)
            if wait_seconds > 0:
                 logger.info(f"[{device.name}] 请求等待 {wait_seconds:.1f} 秒。")
                 return {"success": True, "waiting": True, "wait_duration": wait_seconds}
            else: return {"success": False, "error": "等待时长必须为正数"}
        except ValueError: return {"success": False, "error": f"无效的等待时长: {seconds}"}

    def _execute_analyze_screen(self, device: Device) -> Dict[str, Any]:
        """处理 analyze_screen 动作 (AI 强制重新评估，无物理动作)。"""
        logger.info(f"[{device.name}] AI 请求重新分析屏幕。");
        # 返回成功，调度器会继续下一轮（届时会重新截图分析）
        return {"success": True, "message": "请求屏幕分析"}

    def _execute_search_text(self, device: Device, text: str) -> Dict[str, Any]:
        """处理 search_text 动作 (查找文本，无物理动作)。"""
        logger.info(f"[{device.name}] AI 请求搜索文本: '{text}'")
        # --- 查找前确保在设备原点 ---
        origin_result = self._return_to_device_origin(device)
        if not origin_result.get("success"): logger.warning(f"[{device.name}] SearchText: 查找前返回原点失败。")

        coords = self._find_text_on_screen(device, text, use_partial_match=True)
        if coords:
             logger.info(f"[{device.name}] 找到文本 '{text}' at {coords}");
             return {"success": True, "found": True, "center": coords}
        else:
             logger.info(f"[{device.name}] 未找到文本 '{text}'");
             return {"success": True, "found": False}

    def _execute_esp_command(self, device: Device, command_string: str) -> Dict[str, Any]:
        """
        【原子操作】处理 esp_command 动作 (直接发送命令)。
        移动命令会自动返回原点和延时。
        """
        logger.info(f"[{device.name}] 请求直接发送 ESP 命令: '{command_string}'")
        # 粗略判断是否是移动命令
        is_move_command = command_string.upper().startswith(('G ', 'X ', 'Y ', 'HOME', 'M1', 'M2', 'M3'))

        result = {"success": False, "error": "锁获取失败"}
        if not self.esp_action_lock.acquire(timeout=10):
            logger.error(f"[{device.name}] 获取 ESP 动作锁超时 (执行 esp_command)。")
            return result
        try:
            logger.info(f"[{device.name}] [Lock Acquired] 发送自定义 ESP 命令: '{command_string}'...")
            cmd_result = self.esp_controller.send_command(command_string, wait_for_response=True)

            if cmd_result.get("success"):
                if is_move_command:
                    logger.debug(f"[{device.name}] 自定义移动命令成功。等待内部延时 {self.INTERNAL_ACTION_DELAY_SECONDS}s...")
                    time.sleep(self.INTERNAL_ACTION_DELAY_SECONDS) # 等待动作完成
                    logger.debug(f"[{device.name}] 准备返回设备原点...")
                    return_result = self._internal_return_to_device_origin(device)
                    if return_result.get("success"):
                        logger.debug(f"[{device.name}] 返回设备原点成功。等待外部延时 {self.POST_ACTION_PHYSICAL_DELAY_SECONDS}s...")
                        time.sleep(self.POST_ACTION_PHYSICAL_DELAY_SECONDS)
                        logger.debug(f"[{device.name}] 自定义移动命令序列完成。")
                        result = {"success": True, "response": cmd_result.get("response")}
                    else:
                        error_msg = f"自定义命令成功但返回设备原点失败: {return_result.get('error')}"
                        logger.error(f"[{device.name}] {error_msg}")
                        result = {"success": False, "error": error_msg, "response": cmd_result.get("response")}
                else:
                    logger.debug(f"[{device.name}] 自定义非移动命令成功。")
                    result = {"success": True, "response": cmd_result.get("response")}
            else:
                error_msg = f"自定义命令失败: {cmd_result.get('error')}"
                logger.error(f"[{device.name}] {error_msg}")
                result = {"success": False, "error": error_msg, "response": cmd_result.get("response")}
            logger.info(f"[{device.name}] 完成自定义 ESP 命令处理。")
        finally:
            logger.debug(f"[{device.name}] [Lock Releasing] esp_command")
            self.esp_action_lock.release()
        return result

    def _execute_complete_task(self, device: Device) -> Dict[str, Any]:
        """处理 complete_task 动作 (AI 认为任务完成)。"""
        logger.info(f"[{device.name}] AI 指示任务完成。")
        return {"success": True, "completed": True} # 返回完成标记

    def _execute_back(self, device: Device) -> Dict[str, Any]:
        """
        【原子操作】执行返回操作（尝试模板/文本点击，最后使用默认坐标）。
        【修改】调整默认 Y 坐标，使其更安全。
        """
        logger.info(f"[{device.name}] 请求执行 'back' 操作...")

        # --- 查找模板/文本 (锁外，但需要先回原点获取最新截图) ---
        origin_result = self._return_to_device_origin(device)
        if not origin_result.get("success"):
            logger.warning(f"[{device.name}] Back: 查找前返回原点失败。")
            # 即使返回原点失败，仍然尝试查找，可能只是 ESP 响应问题

        # 尝试模板匹配
        template_coords = self._find_template_on_screen(device, "back_button", threshold=0.7)
        if template_coords:
            logger.info(f"[{device.name}] 找到 'back_button' 模板，调用 click...")
            # _execute_click 内部会处理转换和钳位
            return self._execute_click(device, template_coords[0], template_coords[1])

        # 尝试文本匹配
        text_coords = self._find_text_on_screen(device, "返回", use_partial_match=False)
        if text_coords:
            logger.info(f"[{device.name}] 找到 '返回' 文本，调用 click...")
            # _execute_click 内部会处理转换和钳位
            return self._execute_click(device, text_coords[0], text_coords[1])

        # --- 使用调整后的默认坐标 ---
        crop_w, crop_h = device.get_config("CROPPED_RESOLUTION", CONFIG["CROPPED_RESOLUTION"])
        # 默认 X 坐标：靠近左边缘，但留有边距
        back_x = max(20, int(crop_w * 0.05))
        # 默认 Y 坐标：调整为屏幕高度的 90% 左右，避开最底部
        # 之前 1368 / 1440 ≈ 95%，可能太靠下了
        back_y = int(crop_h * 0.90) # 尝试 90% 高度
        logger.warning(f"[{device.name}] 未找到返回按钮的模板或文本，使用调整后的默认像素坐标 ({back_x}, {back_y}) 调用 click...")
        # _execute_click 内部会处理转换和钳位
        return self._execute_click(device, back_x, back_y)

    # --- Recovery Logic (保持不变，依赖原子操作) ---
    def _attempt_recovery(self, device: Device) -> Dict[str, Any]:
        """尝试将设备恢复到已知状态（例如主屏幕）。"""
        logger.warning(f"[{device.name}] 任务执行出错，尝试自动恢复...")
        max_recovery_attempts = 2 # 减少恢复尝试次数

        for i in range(max_recovery_attempts):
            progress_msg = f"恢复尝试 {i + 1}/{max_recovery_attempts}"
            device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
            logger.info(f"[{device.name}] 恢复尝试 {i + 1}")

            # 1. 执行返回操作 (原子操作)
            logger.info(f"[{device.name}] 恢复: 执行 'back' 操作...")
            back_result = self._execute_back(device)
            device.add_action({"action": "Recovery: back()", "rationale": f"Attempt {i + 1}"})
            if not back_result.get("success"):
                logger.warning(f"[{device.name}] 恢复尝试 {i + 1}: 'back' 操作失败: {back_result.get('error')}")
                time.sleep(1.0) # 失败后等待
                continue

            # 2. 检查屏幕状态 (back 后已返回原点并延时)
            logger.debug(f"[{device.name}] 恢复: 检查当前屏幕状态...")
            screenshot, _, text_positions_after, templates_after = self._get_current_screen_context(device)
            if screenshot is None:
                logger.warning(f"[{device.name}] 恢复尝试 {i + 1}: 检查状态时截图失败。")
                time.sleep(1.5)
                continue

            # --- 验证是否恢复到主屏幕 (逻辑与 _navigate_to_home_screen 类似) ---
            is_recovered = False
            home_template_name = device.get_config("HOME_SCREEN_TEMPLATE_NAME")
            # ... (省略重复的验证代码, 参考 _navigate_to_home_screen) ...
            # 这里简化：如果找到了主屏幕模板或足够多的锚点文本，则认为恢复
            home_anchor_texts = device.get_config("HOME_SCREEN_ANCHOR_TEXTS", [])
            min_anchor_count = device.get_config("HOME_SCREEN_MIN_ANCHORS", 0)
            home_template_threshold = device.get_config("HOME_SCREEN_TEMPLATE_THRESHOLD")

            template_matched = False
            if home_template_name and home_template_threshold is not None:
                 match_res = self.ai_analyzer.template_matching(screenshot, home_template_name, threshold=home_template_threshold)
                 if match_res.get("match"): template_matched = True

            ocr_verified = False
            if home_anchor_texts and min_anchor_count > 0 and text_positions_after:
                 found_anchors = {a.strip() for a in home_anchor_texts if a.strip() and any(a.strip() == item.get("text","").strip() for item in text_positions_after)}
                 if len(found_anchors) >= min_anchor_count: ocr_verified = True

            if template_matched or ocr_verified:
                 logger.info(f"[{device.name}] 恢复检查：检测到主屏幕特征。")
                 is_recovered = True
            # --- 结束简化验证 ---

            if is_recovered:
                logger.info(f"[{device.name}] 自动恢复成功 (尝试 {i + 1})。")
                progress_msg = "自动恢复成功。"; device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
                self._return_to_device_origin(device) # 确保最后在原点
                return {"success": True}

            logger.debug(f"[{device.name}] 恢复尝试 {i + 1} 未成功，将进行下一次尝试。")
            time.sleep(1.0 + i)

        logger.error(f"[{device.name}] 自动恢复失败（尝试 {max_recovery_attempts} 次）。")
        progress_msg = "自动恢复失败。"; device.update_progress(progress_msg); self._emit_progress(device.name, progress_msg)
        self._return_to_device_origin(device)
        return {"success": False, "error": "自动恢复失败"}

    def _apply_coordinate_transform(self, device: Device, x: int, y: int) -> Tuple[int, int]:
        """应用设备特定的像素坐标内部微调。(保持不变)"""
        coord_map = device.get_config("COORDINATE_MAP")
        if coord_map and isinstance(coord_map, dict):
            try:
                scale_x = float(coord_map.get("scale_x", 1.0)); offset_x = float(coord_map.get("offset_x", 0))
                scale_y = float(coord_map.get("scale_y", 1.0)); offset_y = float(coord_map.get("offset_y", 0))
                transformed_x = int(round(x * scale_x + offset_x)); transformed_y = int(round(y * scale_y + offset_y))
                if (transformed_x, transformed_y) != (x, y): logger.debug(f"设备内坐标映射 '{device.name}': 像素 ({x},{y}) -> 微调后 ({transformed_x},{transformed_y})")
                return transformed_x, transformed_y
            except (TypeError, ValueError) as e: logger.warning(f"设备 '{device.name}' 的坐标映射配置无效: {coord_map}。错误: {e}。使用原始像素坐标。"); return x, y
        return x, y

class TaskScheduler(QObject):
    """
    管理任务队列并以串行轮询方式将任务步骤分配给设备执行器。
    """
    # --- 定义信号 ---
    device_update_required = pyqtSignal()
    task_update_required = pyqtSignal()

    def __init__(self, config: Dict[str, Any], task_executor: Optional['TaskExecutor']):
        super().__init__() # 调用 QObject 的构造函数
        self.config = config
        self.task_executor = task_executor
        self.devices: Dict[str, Device] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.lock = threading.Lock()
        # ===> 关键改动：确保 stop_event 初始状态为 False <===
        self.stop_event = threading.Event()
        self.stop_event.clear() # 明确设置为 False
        logger.debug("TaskScheduler __init__: stop_event 初始化并明确设置为 False")
        # ===> 结束改动 <===
        self.scheduler_thread: Optional[threading.Thread] = None

        logger.info("TaskScheduler 初始化 (串行模式，使用信号/槽)。")


    def set_executor(self, executor: 'TaskExecutor'):
        """设置 TaskExecutor 实例 (如果在初始化时未提供)。"""
        if not self.task_executor:
            self.task_executor = executor
            logger.info("TaskExecutor 已设置到 TaskScheduler。")
        else:
            logger.warning("TaskExecutor 已设置，忽略新的设置请求。")

    def _update_task_status_in_config(self, task_id: str, new_status: TaskStatus):
        """
        【内部辅助】更新 self.config["USER_TASKS"] 中任务的状态。
        必须在持有 self.lock 的情况下调用。
        """
        if "USER_TASKS" not in self.config or not isinstance(self.config["USER_TASKS"], list):
            return False # 配置中没有用户任务，无需更新
        found = False
        for task_dict in self.config["USER_TASKS"]:
            if task_dict.get("task_id") == task_id:
                old_status_str = task_dict.get("status")
                new_status_str = new_status.value
                if old_status_str != new_status_str:
                    task_dict["status"] = new_status_str
                    logger.info(f"已更新配置文件(内存)中任务 ID {task_id} 的状态为: {new_status_str}")
                found = True
                break
        if not found: logger.warning(f"尝试更新配置状态失败：未在 config['USER_TASKS'] 中找到任务 ID {task_id}")
        return found

    def _load_all_predefined_tasks(self):
        """从 config['USER_TASKS'] 加载任务定义，并尊重其保存的状态。"""
        loaded_count = 0; queued_count = 0; completed_loaded_count = 0
        user_tasks_data = self.config.get("USER_TASKS", [])
        if not isinstance(user_tasks_data, list): logger.warning("配置中的 'USER_TASKS' 不是列表格式。"); return
        tasks_to_add_to_queue = []; tasks_to_add_to_completed = []
        for task_dict in user_tasks_data:
            try:
                task = Task.from_dict(task_dict); loaded_count += 1
                if task.status == TaskStatus.PENDING: tasks_to_add_to_queue.append(task); queued_count += 1
                elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]: tasks_to_add_to_completed.append(task); completed_loaded_count += 1
                elif task.status == TaskStatus.RUNNING:
                    logger.warning(f"加载到运行中状态的任务 '{task.name}'，重置为 PENDING。"); task.status = TaskStatus.PENDING; task.task_stage = "PENDING"; task.retry_count = 0; task.error = "任务因程序重启中断"; tasks_to_add_to_queue.append(task); queued_count += 1
                    with self.lock: self._update_task_status_in_config(task.task_id, TaskStatus.PENDING)
            except ValueError as e: logger.error(f"从配置加载任务时值无效: {task_dict.get('name')}, 错误: {e}")
            except Exception as e: logger.error(f"从配置加载用户任务时出错: {task_dict.get('name')} - {e}", exc_info=True)
        with self.lock:
            for task in tasks_to_add_to_queue:
                 if not hasattr(task, 'task_id') or not task.task_id: task.task_id = uuid.uuid4().hex
                 task.status = TaskStatus.PENDING; task.task_stage = "PENDING"
                 self.task_queue.put((-task.priority, datetime.now(), task.task_id, task))
            existing_completed_ids = {t.task_id for t in self.completed_tasks}
            for task in tasks_to_add_to_completed:
                 if task.task_id not in existing_completed_ids: self.completed_tasks.append(task)
            self.completed_tasks = self.completed_tasks[-100:]
        logger.info(f"从配置加载了 {loaded_count} 个用户任务定义。{queued_count} 个加入队列，{completed_loaded_count} 个加入完成列表。")
        # 不再在这里触发回调

    def add_device(self, device: Device) -> None:
        """添加或更新设备信息。"""
        with self.lock:
            is_update = device.name in self.devices; self.devices[device.name] = device
            log_action = "更新" if is_update else "添加"; logger.info(f"设备已{log_action}: {device.name}")
        # 主循环会检测到变化并触发信号

    def remove_device(self, device_name: str) -> None:
        """移除设备，处理相关任务和配置。"""
        task_to_cancel = None; task_updated = False
        with self.lock:
            if device_name in self.devices:
                device_to_remove = self.devices[device_name]
                if device_name in self.running_tasks:
                    task_to_cancel = self.running_tasks.pop(device_name); task_updated = True
                    logger.warning(f"设备 '{device_name}' 正在运行任务 '{task_to_cancel.name}'，将取消。")
                    task_to_cancel.cancel(); self._update_task_status_in_config(task_to_cancel.task_id, TaskStatus.CANCELED)
                    if task_to_cancel not in self.completed_tasks: self.completed_tasks.append(task_to_cancel)
                    self.completed_tasks = self.completed_tasks[-100:]
                    device_to_remove.status = DeviceStatus.IDLE; device_to_remove.current_task = None; device_to_remove.task_progress = "设备移除，任务取消"
                if "DEVICE_CONFIGS" in self.config and device_name in self.config["DEVICE_CONFIGS"]: del self.config["DEVICE_CONFIGS"][device_name]; logger.info(f"已从内存配置移除设备 '{device_name}'。")
                del self.devices[device_name]; logger.info(f"已移除设备: {device_name}")
                # 触发 UI 更新在循环中完成
            else: logger.warning(f"无法移除设备: 未找到 '{device_name}'。")


    def get_device(self, device_name: str) -> Optional[Device]:
        """获取指定名称的设备对象。"""
        with self.lock:
            return self.devices.get(device_name)

    def add_task(self, task: Task, trigger_update: bool = True) -> bool: # trigger_update 参数现在可以忽略
        """将任务添加到优先级队列，并更新 config。"""
        with self.lock:
            if not hasattr(task, 'task_id') or not task.task_id: task.task_id = uuid.uuid4().hex; logger.warning(f"任务 '{task.name}' 分配新 ID: {task.task_id}")
            task.status = TaskStatus.PENDING; task.task_stage = "PENDING"; task.retry_count = 0; task.error = None
            task_dict = task.to_dict()
            if "USER_TASKS" not in self.config or not isinstance(self.config["USER_TASKS"], list): self.config["USER_TASKS"] = []
            updated_in_config = False
            for i, existing_task_dict in enumerate(self.config["USER_TASKS"]):
                if existing_task_dict.get("task_id") == task.task_id: self.config["USER_TASKS"][i] = task_dict; updated_in_config = True; break
            if not updated_in_config: self.config["USER_TASKS"].append(task_dict)
            self.task_queue.put((-task.priority, datetime.now(), task.task_id, task))
            logger.info(f"任务已添加到队列: '{task.name}' (ID: {task.task_id})")
        # 不再直接触发 UI 更新
        return True

    def cancel_task(self, task_id: str) -> bool:
        """将等待中或运行中任务标记为已取消，并更新 config 状态。"""
        canceled = False; task_name = f"ID {task_id}"; task_found = False
        with self.lock:
            updated_queue_items = []; original_queue_size = self.task_queue.qsize()
            while not self.task_queue.empty():
                try: item = self.task_queue.get_nowait(); prio, ts, tid, task_obj = item
                except queue.Empty: break
                if tid == task_id:
                    task_name = task_obj.name; task_found = True
                    if task_obj.status == TaskStatus.PENDING:
                        task_obj.cancel(); canceled = True
                        if task_obj not in self.completed_tasks: self.completed_tasks.append(task_obj)
                        self.completed_tasks = self.completed_tasks[-100:]
                        self._update_task_status_in_config(task_id, TaskStatus.CANCELED)
                        logger.info(f"已取消等待中任务: '{task_name}' (ID: {task_id})。")
                    else: updated_queue_items.append(item)
                else: updated_queue_items.append(item)
            for item in updated_queue_items: self.task_queue.put(item)
            if not canceled:
                for dev_name, running_task in self.running_tasks.items():
                    if running_task.task_id == task_id:
                        task_name = running_task.name; task_found = True
                        if running_task.status == TaskStatus.RUNNING:
                            running_task.cancel(); canceled = True
                            self._update_task_status_in_config(task_id, TaskStatus.CANCELED)
                            logger.info(f"标记运行中任务 '{task_name}' (ID: {task_id}) 为取消状态。")
                        else: logger.warning(f"尝试取消的任务 '{task_name}' 状态不是 RUNNING ({running_task.status.value})。")
                        break
        # 不再直接触发 UI 更新
        if not task_found: logger.warning(f"无法取消任务: ID {task_id} 未在等待或运行中找到。")
        return canceled

    def get_task_lists(self) -> Dict[str, List[Task]]:
        """获取等待中、运行中和最近完成的任务列表。"""
        pending = []; running = []; completed = []
        with self.lock:
            temp_list = list(self.task_queue.queue)
            pending_tasks_with_meta = [(task, ts) for prio, ts, tid, task in temp_list if task.status == TaskStatus.PENDING]
            pending_tasks_with_meta.sort(key=lambda item: (item[0].priority, item[1]), reverse=True)
            pending = [task for task, ts in pending_tasks_with_meta]
            running = list(self.running_tasks.values())
            completed = self.completed_tasks[-100:]
        return {"pending": pending, "running": running, "completed": completed}


    def start_scheduler(self) -> None:
        """启动任务调度器线程。"""
        logger.info("请求启动任务调度器...") # <<< 修改日志
        if not self.task_executor:
            logger.error("无法启动调度器：TaskExecutor 未设置。")
            QMessageBox.critical(None, "启动错误", "TaskScheduler 未关联 TaskExecutor！")
            return
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("任务调度器已在运行。")
            return

        # <<< 在启动线程前再次检查并清除 stop_event >>>
        if self.stop_event.is_set():
            logger.warning("启动调度器时发现 stop_event 已被设置，强制清除！")
            self.stop_event.clear()
        else:
            logger.debug("启动调度器前检查: stop_event 未被设置。")

        logger.info("准备创建并启动调度器线程 (TaskSchedulerThread)...")
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="TaskSchedulerThread")
        # ===> 关键改动：添加 try...except 围绕 start() <===
        try:
            self.scheduler_thread.start()
            logger.info("调度器线程 start() 方法已调用。")
        except Exception as start_err:
            logger.critical(f"启动调度器线程时发生严重错误: {start_err}", exc_info=True)
            self.scheduler_thread = None # 启动失败，重置线程对象
            QMessageBox.critical(self, "线程启动错误", f"无法启动调度器线程:\n{start_err}")
            return
        # ===> 结束改动 <===

        # <<< 线程启动后短暂延时并检查 >>>
        time.sleep(0.1) # 给线程一点启动时间
        if self.scheduler_thread and self.scheduler_thread.is_alive(): # 增加 self.scheduler_thread 检查
            logger.info("任务调度器线程已成功启动 (is_alive() == True)。")
        elif self.scheduler_thread: # 线程对象存在但未运行
            logger.error("任务调度器线程启动后未能运行或立即结束 (is_alive() == False)！")
            # 尝试获取线程退出码等信息（如果平台支持）
            # exit_code = getattr(self.scheduler_thread, '_exitcode', '未知') # 不可靠
            # logger.error(f"线程退出码 (如果可用): {exit_code}")
        else: # 线程对象都不存在了（可能在 start() 时就失败了）
             logger.error("任务调度器线程对象在启动尝试后为 None。")


    def stop_scheduler(self) -> None:
        """停止任务调度器线程。"""
        logger.info("请求停止任务调度器...") # <<< 修改日志
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.info("任务调度器未在运行。")
            # <<< 即使线程不在运行，也确保设置 stop_event >>>
            logger.debug("确保 stop_event 被设置...")
            self.stop_event.set()
            # <<< 结束改动 >>>
            return
        logger.info("正在设置 stop_event 以停止调度器循环...")
        self.stop_event.set()
        logger.debug("正在等待调度器线程结束 (最多 5 秒)...")
        self.scheduler_thread.join(timeout=5.0)
        if self.scheduler_thread.is_alive():
            logger.warning("调度器线程在等待 5 秒后未能优雅停止。")
        else:
            logger.info("调度器线程已成功停止。")
        self.scheduler_thread = None # 清理线程引用


    def _scheduler_loop(self) -> None:
        """任务调度器的主循环 (串行轮询模型)。"""
        loop_count = -1

        while not self.stop_event.is_set():
            loop_count += 1
            # 使用 print 和 logger 记录循环开始
            # --- 为了绝对确定循环是否卡死，暂时只放最简单的日志和 sleep ---
            try:
                # <<<<<<< 暂时注释掉所有实际逻辑 >>>>>>>>>
                # 1. 更新设备状态
                devices_updated = self._update_device_status()
                # # 2. 分配新任务
                tasks_assigned = self._assign_tasks_to_idle_devices()

                # # 3. 处理运行中任务
                tasks_processed = self._process_running_tasks()
                # 4. UI 更新触发
                if devices_updated:
                    try: self.device_update_required.emit()
                    except Exception as emit_err: logger.error(f"发射 device_update_required 信号时出错: {emit_err}", exc_info=True)

                if tasks_assigned or tasks_processed:
                    try: self.task_update_required.emit()
                    except Exception as emit_err: logger.error(f"发射 task_update_required 信号时出错: {emit_err}", exc_info=True)

                # 5. 控制循环速率 (保留 sleep)
                sleep_interval = self.config.get("TASK_POLLING_INTERVAL", 1.0) # 使用 1 秒间隔
                time.sleep(sleep_interval)

            except SystemExit: # Allow SystemExit to propagate (e.g., from sys.exit())
                logger.info("Scheduler loop received SystemExit, setting stop_event and re-raising.")
                self.stop_event.set() # Ensure stop_event is set for a clean exit
                raise # Re-raise to allow Python to exit
            except KeyboardInterrupt: # Allow KeyboardInterrupt to propagate
                logger.info("Scheduler loop interrupted by KeyboardInterrupt, setting stop_event and breaking.")
                self.stop_event.set() # Ensure graceful shutdown on Ctrl+C
                break # Exit the loop
            except Exception as e:
                # Log critical error but continue loop for most exceptions.
                logger.critical("Unhandled exception in scheduler loop iteration %s: %s", loop_count, e, exc_info=True)
                # Optional: Add a slightly longer delay here if errors are very rapid,
                # to prevent log spamming and high CPU, but continue the loop.
                # time.sleep(self.config.get("TASK_POLLING_INTERVAL", 1.0) * 5) # e.g., wait 5x polling interval
                continue # Continue to the next iteration

    def _process_running_tasks(self) -> bool:
        """按顺序轮询执行每个运行中任务的下一步。返回是否有任务被处理。"""

        if not self.task_executor:
            logger.error("_process_running_tasks: TaskExecutor 未设置!")
            return False

        processed_task = False
        running_items = []

        items_count = -1
        try:
            if self.lock.acquire(timeout=5.0):
                try:
                    running_items = list(self.running_tasks.items())
                    items_count = len(running_items)
                finally:
                    self.lock.release()
            else:
                return False
        except Exception as lock_err:
             return False

        if items_count <= 0:
            logger.debug(f"--- PRINT DEBUG: _process_running_tasks Skipping FOR loop (items_count={items_count}). ({datetime.now()}) ---", flush=True)
        else:
            logger.debug(f"_process_running_tasks: running_items 包含 {items_count} 个任务，准备进入 for 循环...")

            # --- 后续的任务处理逻辑 ---
            for device_name, task in running_items:
                logger.debug(f"_process_running_tasks: 正在处理设备 '{device_name}' 的任务 '{task.name}' ...")

                should_process = False
                device_status_for_log = "None" # 用于日志记录
                try:
                    if self.lock.acquire(timeout=1.0):
                        try:
                            if device_name in self.running_tasks and self.running_tasks[device_name] == task:
                                device = self.devices.get(device_name)
                                # ===> 关键修改：允许处理 INITIALIZING 或 BUSY 状态 <===
                                if device and device.status in [DeviceStatus.BUSY, DeviceStatus.INITIALIZING]:
                                    should_process = True
                                    device_status_for_log = device.status.value # 记录允许处理的状态
                                    logger.debug(f"_process_running_tasks: 设备 '{device_name}' 状态为 {device_status_for_log}，将处理任务 '{task.name}'.")
                                else:
                                    device_status_for_log = device.status.value if device else "None"
                                    logger.debug(f"_process_running_tasks: 设备 '{device_name}' 状态为 {device_status_for_log} (非 BUSY/INITIALIZING) 或不存在，跳过 '{task.name}'.")
                            else:
                                logger.debug(f"_process_running_tasks: 任务 '{task.name}' 在设备 '{device_name}' 上已不再运行，跳过。")
                        finally:
                             self.lock.release()
                    else:
                        logger.warning(f"_process_running_tasks: 检查任务状态时获取锁超时，跳过 '{task.name}'.")
                except Exception as check_lock_err:
                    logger.error(f"_process_running_tasks: 检查任务状态获取锁时异常: {check_lock_err}", exc_info=True)

                if not should_process:
                     print(f"--- PRINT DEBUG: _process_running_tasks Skipping task '{task.name}' (Status={device_status_for_log}, should_process=False). ({datetime.now()}) ---", flush=True) # 日志中加入状态
                     continue # 如果不应处理，跳到下一个任务

                # --- 任务取消检查, 调用 execute_next_step, 处理结果等保持不变 ---
                if task.status == TaskStatus.CANCELED:
                    # ... (处理取消)
                    continue

                logger.info(f"_process_running_tasks: 准备调用 TaskExecutor.execute_next_step 为任务 '{task.name}' (设备 '{device_name}', 状态: {device_status_for_log})...") # 日志加入状态
                step_result = None
                try:
                    step_result = self.task_executor.execute_next_step(device, task)
                    logger.debug(f"_process_running_tasks: TaskExecutor.execute_next_step 为任务 '{task.name}' 返回: {step_result}")
                    processed_task = True
                except Exception as exec_err:
                    logger.error(f"_process_running_tasks: 调用 execute_next_step 处理任务 '{task.name}' 时捕获到异常: {exec_err}", exc_info=True)
                    step_result = {"success": False, "error": f"调度器在调用执行器时捕获异常: {exec_err}", "exception": True}
                    processed_task = True

                if step_result is not None:
                    logger.debug(f"_process_running_tasks: 准备调用 _handle_step_result 处理任务 '{task.name}' 的结果...")
                    self._handle_step_result(device, task, step_result)
                else:
                     logger.error(f"_process_running_tasks: TaskExecutor.execute_next_step 意外返回 None，任务 '{task.name}'。")

        return processed_task

    def _handle_step_result(self, device: Device, task: Task, result: Dict[str, Any]):
        """
        根据执行步骤的结果更新任务和设备状态。
        【已修改 V5】处理 wait 子任务时立即推进索引，并处理因此完成的情况。
        """
        success = result.get("success", False)
        error_msg = result.get("error")
        is_completed_by_action = result.get("completed", False) # AI 或子任务要求直接完成
        is_canceled = result.get("canceled", False) or task.status == TaskStatus.CANCELED
        is_waiting = result.get("waiting", False)
        wait_seconds = result.get("wait_duration", 0)
        force_fail_task = result.get("force_fail", False) # 步骤强制任务失败
        exception_occurred = result.get("exception", False)
        step_skipped_for_internal_reason = result.get("skipped", False)

        with self.lock: # 确保对任务和设备状态的修改是线程安全的
            # 检查任务是否仍然由该设备运行 (可能在步骤执行期间被取消或重新分配)
            if device.name not in self.running_tasks or self.running_tasks[device.name] != task:
                logger.debug(f"_handle_step_result: 任务 '{task.name}' 在设备 '{device.name}' 上已不再运行，忽略结果。")
                return

            # 1. 处理内部跳过 (例如 LOOP_START/END, PREPARING 完成)
            if step_skipped_for_internal_reason:
                reason = result.get("reason", "内部处理")
                logger.info(f"_handle_step_result: 任务 '{task.name}' 步骤在索引 {task.current_subtask_index} 被跳过 ({reason})，继续下一个调度循环。")
                # 通常跳过意味着索引已在 task_executor 中处理，这里直接返回
                return

            # 2. 处理任务取消
            if is_canceled:
                logger.info(f"任务 '{task.name}' (ID: {task.task_id}) 已被取消，从运行列表移除。")
                self.running_tasks.pop(device.name, None)
                task.cancel() # 确保任务对象状态更新
                if task not in self.completed_tasks: self.completed_tasks.append(task)
                self.completed_tasks = self.completed_tasks[-100:] # 限制完成列表大小
                device.complete_task(success=False) # 设备状态 IDLE 或 ERROR
                device.task_progress = "任务已取消"
                self._update_task_status_in_config(task.task_id, TaskStatus.CANCELED)
                return

            # 3. 处理等待状态 (核心修改点)
            if is_waiting and wait_seconds > 0:
                # 只有当任务当前处于 RUNNING 阶段时才应该进入等待
                # 如果已经是 WAITING (例如，某些意外情况导致重复调用)，则仅更新等待时间
                if task.task_stage == "RUNNING" or task.task_stage == "WAITING":
                    device.set_waiting(wait_seconds) # 设置设备等待
                    task.task_stage = "WAITING"      # 设置任务阶段为等待
                    logger.info(f"任务 '{task.name}' (ID: {task.task_id}) 进入/更新等待状态 ({wait_seconds:.1f}秒)。")

                    # --- 关键修复：在启动等待后，立即推进子任务索引 ---
                    if not task.use_ai_driver: # 仅对子任务模式有效
                        task.current_subtask_index += 1
                        logger.info(f"等待开始后，任务 '{task.name}' 的子任务索引前进到 {task.current_subtask_index} (总共 {len(task.subtasks)} 个子任务)。")

                        # 检查在推进索引后，任务是否已完成所有子任务
                        if task.current_subtask_index >= len(task.subtasks):
                            logger.info(f"任务 '{task.name}' (ID: {task.task_id}) 因等待后到达子任务末尾而完成。")
                            task.complete(success=True) # 标记任务完成
                            self.running_tasks.pop(device.name, None) # 从运行列表移除
                            if task not in self.completed_tasks: self.completed_tasks.append(task)
                            self.completed_tasks = self.completed_tasks[-100:]
                            # 注意：设备状态仍然是 WAITING。当等待结束后，_update_device_status 会将其变为 BUSY。
                            # 由于任务已完成，设备最终会变回 IDLE。
                            # 这里不需要 device.complete_task()，因为设备仍在物理等待。
                            self._update_task_status_in_config(task.task_id, TaskStatus.COMPLETED)
                else:
                    logger.warning(f"任务 '{task.name}' 在非 RUNNING/WAITING 阶段 ({task.task_stage}) 收到等待指令，已忽略。")
                return # 处理完等待后直接返回，等待计时器触发后续

            # 4. 处理由动作直接请求的任务完成
            if success and is_completed_by_action:
                logger.info(f"任务 '{task.name}' (ID: {task.task_id}) 通过步骤指令成功完成。")
                task.complete(success=True)
                self.running_tasks.pop(device.name, None)
                if task not in self.completed_tasks: self.completed_tasks.append(task)
                self.completed_tasks = self.completed_tasks[-100:]
                device.complete_task(success=True)
                self._update_task_status_in_config(task.task_id, TaskStatus.COMPLETED)
                return

            # 5. 处理步骤执行失败
            if not success:
                current_subtask_index_for_log = task.current_subtask_index if not task.use_ai_driver else -1
                is_subtask_mode = not task.use_ai_driver

                # 5.1 子任务失败重试逻辑 (非强制失败或异常)
                if is_subtask_mode and not force_fail_task and not exception_occurred:
                    subtask_max_retries = self.config.get("SUBTASK_RETRY_COUNT", 1)
                    if task.current_subtask_retry_count < subtask_max_retries:
                        task.current_subtask_retry_count += 1
                        logger.warning(f"任务 '{task.name}': 子任务 {current_subtask_index_for_log + 1} 执行失败，将重试第 {task.current_subtask_retry_count}/{subtask_max_retries} 次。错误: {error_msg}")
                        progress_msg = task.get_progress_display() # 获取包含重试信息的进度
                        device.update_progress(progress_msg)
                        if self.task_executor: self.task_executor._emit_progress(device.name, progress_msg)
                        return # 等待下一次调度执行此子任务（不推进索引）
                    else: # 子任务重试次数用尽，标记此子任务为"跳过"并继续
                        logger.error(f"任务 '{task.name}': 子任务 {current_subtask_index_for_log + 1} 重试失败 (已尝试 {task.current_subtask_retry_count}/{subtask_max_retries})，将跳过此子任务。错误: {error_msg}")
                        task.current_subtask_retry_count = 0 # 为下一个子任务重置
                        task.current_subtask_index += 1      # 跳过当前失败的子任务
                        # 将此视为"成功处理"了当前子任务（通过跳过），以便后续逻辑能判断任务是否完成
                        success = True # 强制为 True
                        error_msg = f"子任务 {current_subtask_index_for_log + 1} 因重试失败被跳过。原错误: {error_msg}" # 更新错误，标记为跳过
                        # 注意：这里的 success = True 会让流程进入下面的 "步骤成功" 部分
                else: # AI 步骤失败 或 子任务强制失败/异常 (触发任务级别重试或最终失败)
                    should_retry_task_globally = False
                    if not force_fail_task and not exception_occurred and task.max_retries > 0 and task.retry_count < task.max_retries:
                        should_retry_task_globally = True

                    if should_retry_task_globally:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        task.task_stage = "PENDING" # 重置阶段
                        task.error = f"失败等待重试({task.retry_count}/{task.max_retries}): {error_msg}"
                        self.running_tasks.pop(device.name, None)
                        retry_delay_seconds = 2 * (task.retry_count) # 简单的指数退避
                        retry_time = datetime.now() + timedelta(seconds=retry_delay_seconds)
                        self.task_queue.put((-task.priority, retry_time, task.task_id, task)) # 重新加入队列
                        logger.warning(f"任务 '{task.name}' (ID: {task.task_id}) 失败，将在 {retry_delay_seconds} 秒后重试 ({task.retry_count}/{task.max_retries})。错误: {error_msg}")
                        device.status = DeviceStatus.IDLE # 设备变为空闲
                        device.current_task = None
                        device.task_progress = f"失败等待重试({task.retry_count})"
                        self._update_task_status_in_config(task.task_id, TaskStatus.PENDING)
                    else: # 任务最终失败
                        reason = "达到最大任务重试次数" if task.max_retries > 0 and task.retry_count >= task.max_retries else \
                                 "被强制失败" if force_fail_task else \
                                 "发生执行异常" if exception_occurred else \
                                 "步骤失败且无更多重试"
                        final_error_msg = f"{reason}。最后错误: {error_msg}"
                        logger.error(f"任务 '{task.name}' (ID: {task.task_id}) 最终失败。{final_error_msg}")
                        task.complete(success=False, error=final_error_msg)
                        self.running_tasks.pop(device.name, None)
                        if task not in self.completed_tasks: self.completed_tasks.append(task)
                        self.completed_tasks = self.completed_tasks[-100:]
                        device.complete_task(success=False) # 设备标记错误或空闲
                        self._update_task_status_in_config(task.task_id, TaskStatus.FAILED)
                    return # 任务失败处理完毕

            # 6. 处理步骤成功 (或子任务跳过后的 "伪成功")
            # (is_completed_by_action 已在前面处理)
            if success and not is_completed_by_action:
                if device.error_count > 0: device.error_count = 0 # 成功则清零设备错误计数

                # 检查任务总时长是否超时
                max_task_time = self.config.get("MAX_TASK_TIME", 3600)
                if task.start_time and (datetime.now() - task.start_time).total_seconds() > max_task_time:
                    logger.error(f"任务 '{task.name}' (ID: {task.task_id}) 执行超时 ({max_task_time}s)，任务失败。")
                    task.complete(success=False, error="任务执行总时长超时")
                    self.running_tasks.pop(device.name, None)
                    if task not in self.completed_tasks: self.completed_tasks.append(task)
                    self.completed_tasks = self.completed_tasks[-100:]
                    device.complete_task(success=False)
                    self._update_task_status_in_config(task.task_id, TaskStatus.FAILED)
                    return

                # --- 子任务模式下的处理 ---
                if not task.use_ai_driver:
                    task.current_subtask_retry_count = 0 # 子任务成功或被跳过后，重置其重试计数
                    old_subtask_index = task.current_subtask_index # 这个索引对应的子任务刚执行完/跳过

                    # --- 处理条件跳转 (仅当子任务真正成功执行时，跳过的不算) ---
                    # result.get("condition_met") 应该由 CHECK_TEXT/TEMPLATE_EXISTS 等子任务返回
                    # 并且，只有当这些检查子任务本身执行成功 (result["success"] is True 且不是因为跳过)
                    # 才应该考虑其 "condition_met" 结果。
                    # 然而，如果子任务因重试失败而被跳过，我们已经将 success 强制为 True，
                    # 这时我们不应该进行条件跳转。
                    # 因此，我们需要一个原始的成功标记。
                    # 为了简化，我们假设如果 error_msg 包含 "被跳过"，则不进行条件跳转。
                    performed_jump = False
                    original_step_success = not ("被跳过" in (error_msg or "")) # 检查是否是真成功

                    if original_step_success and 0 <= old_subtask_index < len(task.subtasks):
                        current_subtask_def = task.subtasks[old_subtask_index]
                        condition_met_from_result = result.get("condition_met") # 来自检查类子任务

                        target_jump_label = None
                        if condition_met_from_result is True and current_subtask_def.get('on_success_jump_to_label'):
                            target_jump_label = current_subtask_def['on_success_jump_to_label']
                            logger.info(f"任务 '{task.name}', 子任务 {old_subtask_index + 1} 成功且满足条件，尝试跳转到标签: '{target_jump_label}'")
                        elif condition_met_from_result is False and current_subtask_def.get('on_failure_jump_to_label'):
                            target_jump_label = current_subtask_def['on_failure_jump_to_label']
                            logger.info(f"任务 '{task.name}', 子任务 {old_subtask_index + 1} 成功但未满足条件，尝试跳转到标签: '{target_jump_label}'")
                        # 如果 condition_met_from_result is None (非检查类子任务)，则不进行条件跳转

                        if target_jump_label:
                            found_target_index = -1
                            for idx, sub_def in enumerate(task.subtasks):
                                if sub_def.get('label') == target_jump_label:
                                    found_target_index = idx
                                    break
                            if found_target_index != -1:
                                task.current_subtask_index = found_target_index # 直接设置跳转目标索引
                                performed_jump = True
                                logger.info(f"任务 '{task.name}' 跳转到子任务索引 {found_target_index + 1} (标签 '{target_jump_label}')")
                            else:
                                logger.warning(f"任务 '{task.name}' 无法找到跳转标签 '{target_jump_label}'，将顺序执行下一个子任务。")
                    # --- 条件跳转处理结束 ---

                    if not performed_jump and not ("被跳过" in (error_msg or "")): # 如果没有发生跳转，并且不是因为跳过而到这里
                        task.current_subtask_index += 1 # 正常递增索引
                    # (else: task.current_subtask_index 要么被跳转设置了, 要么在子任务跳过时已经增加了)

                    total_subtasks = len(task.subtasks)
                    # 记录日志时使用 +1 显示给用户
                    logger.info(f"任务 '{task.name}' 子任务 {old_subtask_index + 1} 处理完成。下一个索引: {task.current_subtask_index + 1 if task.current_subtask_index < total_subtasks else '结束'} / {total_subtasks}")

                    if task.current_subtask_index >= total_subtasks: # 所有子任务完成
                        final_task_error_msg = error_msg if "被跳过" in (error_msg or "") else None # 如果是因为跳过失败子任务而完成
                        logger.info(f"任务 '{task.name}' (ID: {task.task_id}) 所有子任务执行完毕，任务完成。")
                        task.complete(success=True, error=final_task_error_msg) # 记录跳过信息
                        self.running_tasks.pop(device.name, None)
                        if task not in self.completed_tasks: self.completed_tasks.append(task)
                        self.completed_tasks = self.completed_tasks[-100:]
                        device.complete_task(success=True)
                        self._update_task_status_in_config(task.task_id, TaskStatus.COMPLETED)
                    else: # 任务继续，准备下一个子任务
                        next_subtask_def = task.subtasks[task.current_subtask_index]
                        subtask_desc = next_subtask_def.get('description', next_subtask_def.get('type', 'N/A'))
                        progress_msg = f"准备子任务 {task.current_subtask_index + 1}/{total_subtasks}: {subtask_desc[:30]}"
                        device.update_progress(progress_msg)
                        if self.task_executor: self.task_executor._emit_progress(device.name, progress_msg)

                else: # AI 驱动模式
                    old_ai_step = task.current_step
                    task.current_step += 1
                    logger.info(f"任务 '{task.name}' AI 步骤 {old_ai_step + 1} 成功，前进到步骤 {task.current_step + 1}")
                    progress_msg = f"准备 AI 步骤 {task.current_step + 1}"
                    device.update_progress(progress_msg)
                    if self.task_executor: self.task_executor._emit_progress(device.name, progress_msg)
                return # 步骤成功，任务继续或已完成

            # 如果代码能执行到这里，说明逻辑有遗漏，是个不期望的状态
            logger.error(f"任务 '{task.name}' 在 _handle_step_result 中到达了未处理的逻辑分支。Result: {result}")

    def _update_device_status(self) -> bool:
        """检查并更新所有设备的状态 (主要是等待状态)。返回是否有状态变化。"""
        status_changed = False
        with self.lock:
            current_time = datetime.now()
            for device in self.devices.values():
                old_status = device.status
                if device.status == DeviceStatus.WAITING and device.waiting_until and current_time >= device.waiting_until:
                    device.status = DeviceStatus.BUSY
                    device.waiting_until = None
                    if device.name in self.running_tasks: task = self.running_tasks[device.name]; task.task_stage = "RUNNING"
                    logger.info(f"设备 '{device.name}' 等待结束，恢复 BUSY 状态。")
                    device.update_progress("等待完成，准备下一步...")
                    status_changed = True
        return status_changed

    def _assign_tasks_to_idle_devices(self) -> bool:
        """查找空闲设备并从队列中分配任务。"""
        # logger.debug("进入 _assign_tasks_to_idle_devices") # 此日志级别可能太低
        if not self.task_executor: return False
        tasks_assigned_this_cycle = False
        with self.lock:
            idle_devices = [dev for dev in self.devices.values() if dev.status == DeviceStatus.IDLE]
            # logger.debug(f"_assign_tasks_to_idle_devices: 找到 {len(idle_devices)} 个空闲设备。任务队列大小: {self.task_queue.qsize()}") # 此日志级别可能太低
            if not idle_devices or self.task_queue.empty(): return False

            tasks_to_requeue = []
            while not self.task_queue.empty() and idle_devices:
                try:
                    priority, timestamp, task_id, task = self.task_queue.get_nowait()
                except queue.Empty:
                    break

                if task.status != TaskStatus.PENDING:
                    logger.warning(
                        f"_assign_tasks_to_idle_devices: 从队列取出非 PENDING 任务 '{task.name}' (状态: {task.status.value})，跳过。")
                    continue  # 跳过非等待任务
                device_to_assign = None
                if task.assigned_device_name:
                    specific_device = self.devices.get(task.assigned_device_name)
                    if specific_device and specific_device in idle_devices: device_to_assign = specific_device
                    else: tasks_to_requeue.append((priority, timestamp, task_id, task)); continue
                else:
                    for dev in idle_devices:
                        if not task.app_name or task.app_name in dev.apps: device_to_assign = dev; break
                if device_to_assign:
                    logger.info(f"分配任务 '{task.name}' (ID: {task_id}) 到设备 '{device_to_assign.name}'。")

                    # ===> 关键改动：在更新 self.running_tasks 前后添加日志 <===
                    logger.info(f"_assign_tasks_to_idle_devices: 准备将任务 '{task.name}' 添加到 running_tasks (设备: '{device_to_assign.name}')...")

                    # !!! 执行状态更新 !!!
                    device_to_assign.start_task(task) # 设置设备状态为 INITIALIZING
                    task.start(device_to_assign)      # 设置任务状态为 RUNNING
                    self.running_tasks[device_to_assign.name] = task # 加入运行字典

                    logger.info(f"_assign_tasks_to_idle_devices: 任务 '{task.name}' 已添加到 running_tasks。当前 running_tasks 数量: {len(self.running_tasks)}")
                    # ===> 结束改动 <===

                    idle_devices.remove(device_to_assign); tasks_assigned_this_cycle = True
                    self._update_task_status_in_config(task.task_id, TaskStatus.RUNNING) # 这个更新配置似乎没问题

                else:
                    # logger.debug(f"_assign_tasks_to_idle_devices: 任务 '{task.name}' 未找到合适的空闲设备，放回队列。") # 可能日志过多
                    tasks_to_requeue.append((priority, timestamp, task_id, task)) # 未找到设备，放回队列

            # 将未能分配的任务重新放回队列
            for item in tasks_to_requeue:
                 self.task_queue.put(item)

        # logger.debug("离开 _assign_tasks_to_idle_devices") # 可能日志过多
        return tasks_assigned_this_cycle

# --- UI Classes (with minor adjustments for new features) ---

class DeviceEditDialog(QDialog):
    def __init__(self, parent=None, device=None):
        super().__init__(parent)
        self.device = device
        self.setWindowTitle("设备信息" if device else "添加设备")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        if self.device: self.name_edit.setText(self.device.name)
        form_layout.addRow("设备名称 (*):", self.name_edit)

        self.apps_edit = QLineEdit()
        if self.device: self.apps_edit.setText(", ".join(self.device.apps))
        form_layout.addRow("已安装应用 (逗号分隔):", self.apps_edit)

        self.position_edit = QLineEdit()
        if self.device: self.position_edit.setText(self.device.position)
        form_layout.addRow("机位信息:", self.position_edit)

        # --- 设备物理原点输入 ---
        origin_group = QGroupBox("设备物理原点 (机器人坐标系)")
        origin_layout = QFormLayout()
        self.origin_x_edit = QLineEdit()
        self.origin_y_edit = QLineEdit()
        if self.device:
            # 使用 get_config 获取，确保能读取全局或设备配置
            self.origin_x_edit.setText(str(self.device.get_config("machine_origin_x", 0)))
            self.origin_y_edit.setText(str(self.device.get_config("machine_origin_y", 0)))
        else:
            self.origin_x_edit.setText("0") # 新设备默认为 0
            self.origin_y_edit.setText("0")
        self.origin_x_edit.setPlaceholderText("例如: 100.0") # 提示可以是浮点数
        self.origin_y_edit.setPlaceholderText("例如: 0.0")
        origin_layout.addRow("X 坐标 (mm):", self.origin_x_edit) # 明确单位
        origin_layout.addRow("Y 坐标 (mm):", self.origin_y_edit) # 明确单位
        origin_group.setLayout(origin_layout)
        form_layout.addRow(origin_group)
        # --- 设备物理原点输入结束 ---

        # --- 主屏幕验证相关字段 ---
        home_group = QGroupBox("主屏幕验证设置 (可选)") # 标记为可选
        home_layout = QFormLayout()
        self.home_template_edit = QLineEdit()
        self.home_template_threshold_edit = QLineEdit()
        self.home_anchors_edit = QLineEdit()
        self.home_min_anchors_spin = QSpinBox()
        self.home_min_anchors_spin.setRange(0, 20)

        if self.device:
            # 从配置获取，使用 get_config 以支持全局默认值
            self.home_template_edit.setText(self.device.get_config("HOME_SCREEN_TEMPLATE_NAME", ""))
            self.home_template_threshold_edit.setText(
                str(self.device.get_config("HOME_SCREEN_TEMPLATE_THRESHOLD", ""))) # 可能为空字符串
            self.home_anchors_edit.setText(", ".join(self.device.get_config("HOME_SCREEN_ANCHOR_TEXTS", [])))
            self.home_min_anchors_spin.setValue(self.device.get_config("HOME_SCREEN_MIN_ANCHORS", 0))
        else:
            # 使用全局默认值作为占位符/初始值
            self.home_template_edit.setPlaceholderText(CONFIG.get("HOME_SCREEN_TEMPLATE_NAME", "") or "(可选) 例如: home_screen1")
            self.home_template_threshold_edit.setPlaceholderText(str(CONFIG.get("HOME_SCREEN_TEMPLATE_THRESHOLD", 0.85)) or "(可选, 0.0-1.0) 例如: 0.85")
            self.home_anchors_edit.setPlaceholderText(", ".join(CONFIG.get("HOME_SCREEN_ANCHOR_TEXTS", [])) or "(推荐) 例如: 电话, 短信")
            self.home_min_anchors_spin.setValue(CONFIG.get("HOME_SCREEN_MIN_ANCHORS", 2))

        home_layout.addRow("模板名称:", self.home_template_edit)
        home_layout.addRow("模板阈值:", self.home_template_threshold_edit)
        home_layout.addRow("OCR锚点文本 (逗号分隔):", self.home_anchors_edit)
        home_layout.addRow("最少需要锚点数:", self.home_min_anchors_spin)
        home_group.setLayout(home_layout)
        form_layout.addRow(home_group)
        # --- 主屏幕验证结束 ---

        self.coord_map_edit = QLineEdit()
        coord_map_str = ""
        if self.device:
            coord_map = self.device.get_config("COORDINATE_MAP") # 使用 get_config
            if coord_map: coord_map_str = json.dumps(coord_map)
        self.coord_map_edit.setText(coord_map_str)
        self.coord_map_edit.setPlaceholderText('(可选) 例如: {"scale_x": 1.0, "offset_y": 10}') # 括号改为可选
        form_layout.addRow("坐标映射 (JSON, 内部像素微调):", self.coord_map_edit)

        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setMinimumWidth(450)
        self.adjustSize()

    def get_device_info(self) -> Optional[Dict[str, Any]]:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "输入错误", "设备名称不能为空。")
            return None

        apps = [app.strip() for app in self.apps_edit.text().split(",") if app.strip()]
        position = self.position_edit.text().strip()

        origin_x_str = self.origin_x_edit.text().strip()
        origin_y_str = self.origin_y_edit.text().strip()
        try:
            machine_origin_x = float(origin_x_str) if origin_x_str else 0.0
            machine_origin_y = float(origin_y_str) if origin_y_str else 0.0
        except ValueError:
            QMessageBox.warning(self, "输入错误", "设备物理原点坐标必须是有效的数字 (例如 100 或 100.5)。")
            return None

        home_template_name = self.home_template_edit.text().strip()
        if '.' in home_template_name:
             home_template_name = os.path.splitext(home_template_name)[0]
        home_template_threshold_str = self.home_template_threshold_edit.text().strip()
        home_template_threshold = None
        if home_template_threshold_str:
            try:
                home_template_threshold = float(home_template_threshold_str)
                if not (0.0 <= home_template_threshold <= 1.0):
                    raise ValueError("阈值必须在 0.0 到 1.0 之间")
            except ValueError as e:
                QMessageBox.warning(self, "输入错误", f"无效的主屏幕模板阈值: {e}")
                return None
        home_anchors = [anchor.strip() for anchor in self.home_anchors_edit.text().split(",") if anchor.strip()]
        home_min_anchors = self.home_min_anchors_spin.value()

        if home_template_name and home_template_threshold is None:
            QMessageBox.warning(self, "输入错误", "如果设置了主屏幕模板名称，则必须设置有效的模板阈值 (0.0-1.0)。")
            return None
        if not home_template_name:
            home_template_threshold = None

        if home_anchors and home_min_anchors <= 0:
            QMessageBox.warning(self, "输入错误", "如果设置了 OCR 锚点文本，最少需要锚点数必须大于 0。")
            return None
        if not home_anchors:
            home_min_anchors = 0

        coord_map = None
        coord_map_str = self.coord_map_edit.text().strip()
        if coord_map_str:
            try:
                coord_map = json.loads(coord_map_str)
                if not isinstance(coord_map, dict): raise ValueError("JSON is not an object")
            except json.JSONDecodeError:
                QMessageBox.warning(self, "输入错误", "坐标映射不是有效的JSON格式。")
                return None
            except ValueError as ve:
                QMessageBox.warning(self, "输入错误", f"坐标映射JSON格式错误: {ve}")
                return None

        # --- 修改开始: 返回大写键名的字典 ---
        return {
            "name": name,
            "apps": apps,  # 这个键大小写不敏感，保持原样
            "position": position, # 这个键大小写不敏感，保持原样
            # --- 返回规范化的大写键 ---
            "MACHINE_ORIGIN_X": machine_origin_x, # 使用大写
            "MACHINE_ORIGIN_Y": machine_origin_y, # 使用大写
            "HOME_SCREEN_TEMPLATE_NAME": home_template_name if home_template_name else None, # 使用大写
            "HOME_SCREEN_TEMPLATE_THRESHOLD": home_template_threshold, # 使用大写
            "HOME_SCREEN_ANCHOR_TEXTS": home_anchors, # 使用大写
            "HOME_SCREEN_MIN_ANCHORS": home_min_anchors, # 使用大写
            "COORDINATE_MAP": coord_map # 使用大写
            # --- 修改结束 ---
        }

class TaskEditDialog(QDialog):

    def __init__(self, parent=None, task=None, devices=None):
        # --- 使用 print 确认初始化开始 ---
        super().__init__(parent)
        self.task = task
        self.devices = devices or {}
        self.setWindowTitle("任务信息" if task else "添加任务")
        self._stacked_widget_map: Dict[int, SubtaskType] = {}
        self._subtask_widgets = {}
        self._is_loading_data = False # 加载状态标志
        # --- 初始化 UI ---
        self.init_ui()
        # --- UI 初始化之后再连接信号 ---
        self._connect_subtask_param_signals_once() # 连接初始信号
        # --- 填充数据 ---
        self._populate_existing_subtasks() # 填充现有数据

        # 添加复制粘贴按钮的连接
        if hasattr(self, 'copy_subtasks_btn'):
            self.copy_subtasks_btn.clicked.connect(self.on_copy_subtasks)
        if hasattr(self, 'paste_subtasks_btn'):
            self.paste_subtasks_btn.clicked.connect(self.on_paste_subtasks)

        # 初始化按钮状态
        self.on_subtask_multi_selection_changed()
        # logger.info("TaskEditDialog 初始化完成。") # logger 可能仍有问题

    def init_ui(self):
        # --- 布局和基本信息部分 (与之前代码相同) ---
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # --- 基本任务信息 ---
        self.name_edit = QLineEdit()
        if self.task: self.name_edit.setText(self.task.name)
        form_layout.addRow("任务名称 (*):", self.name_edit)

        self.task_id_label = QLabel()
        if self.task: self.task_id_label.setText(self.task.task_id)
        else: self.task_id_label.setText("(自动生成)")
        self.task_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addRow("任务 ID:", self.task_id_label)

        self.type_combo = QComboBox()
        self.type_combo.addItems([t.value for t in TaskType])
        if self.task: self.type_combo.setCurrentText(self.task.type.value)
        form_layout.addRow("任务类型:", self.type_combo)
        self.app_edit = QLineEdit(self.task.app_name if self.task else "")
        form_layout.addRow("目标应用 (可选):", self.app_edit)
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(-100, 100)
        self.priority_spin.setValue(self.task.priority if self.task else 0)
        form_layout.addRow("优先级 (高->低):", self.priority_spin)
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(0, 10)
        self.max_retries_spin.setValue(self.task.max_retries if self.task else CONFIG.get('RETRY_COUNT', 3))  # 使用全局默认重试
        form_layout.addRow("任务总重试次数:", self.max_retries_spin)  # 明确是任务级别的
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动分配", None)
        for name in sorted(self.devices.keys()): self.device_combo.addItem(name, name)
        if self.task and self.task.assigned_device_name:
            index = self.device_combo.findData(self.task.assigned_device_name)
            if index >= 0: self.device_combo.setCurrentIndex(index)
        form_layout.addRow("指定设备 (可选):", self.device_combo)

        # --- 执行模式选择 ---
        self.execution_mode_group = QGroupBox("执行模式")
        mode_layout = QHBoxLayout() # 改为水平布局，更紧凑
        self.ai_driven_radio = QRadioButton("AI 自动驱动")
        self.subtask_driven_radio = QRadioButton("执行预定义子任务")
        mode_layout.addWidget(self.ai_driven_radio)
        mode_layout.addWidget(self.subtask_driven_radio)
        self.execution_mode_group.setLayout(mode_layout)
        form_layout.addRow(self.execution_mode_group) # 直接添加到 FormLayout

        # --- 子任务编辑器 GUI ---
        self.subtask_editor_widget = QWidget()
        subtask_editor_layout = QVBoxLayout(self.subtask_editor_widget)
        subtask_editor_layout.setContentsMargins(0, 0, 0, 0)

        subtask_list_group = QGroupBox("子任务序列")
        subtask_list_layout = QHBoxLayout()

        self.subtask_list_widget = QListWidget()
        self.subtask_list_widget.setAlternatingRowColors(True)
        # 【新增】允许扩展选择，用于复制粘贴
        self.subtask_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.subtask_list_widget.currentItemChanged.connect(self.on_subtask_selection_changed)  # 单选变化加载参数
        # selectionChanged 信号用于更新多选相关的按钮状态
        self.subtask_list_widget.itemSelectionChanged.connect(self.on_subtask_multi_selection_changed)
        subtask_list_layout.addWidget(self.subtask_list_widget, 3)

        subtask_button_layout = QVBoxLayout()
        self.add_subtask_btn = QPushButton("➕ 添加")
        self.remove_subtask_btn = QPushButton("➖ 移除选中")  # 文本更清晰
        self.move_up_btn = QPushButton("🔼 上移选中")
        self.move_down_btn = QPushButton("🔽 下移选中")
        # 【新增】复制粘贴按钮
        self.copy_subtasks_btn = QPushButton("📋 复制选中")
        self.paste_subtasks_btn = QPushButton("📄 粘贴")
        # --- 连接 ---
        self.add_subtask_btn.clicked.connect(self.on_add_subtask)
        self.remove_subtask_btn.clicked.connect(self.on_remove_subtask)
        self.move_up_btn.clicked.connect(self.on_move_subtask_up)
        self.move_down_btn.clicked.connect(self.on_move_subtask_down)
        # copy/paste 的连接在 __init__ 中完成
        # --- 添加到布局 ---
        subtask_button_layout.addWidget(self.add_subtask_btn)
        subtask_button_layout.addWidget(self.remove_subtask_btn)
        subtask_button_layout.addSpacing(10)
        subtask_button_layout.addWidget(self.move_up_btn)
        subtask_button_layout.addWidget(self.move_down_btn)
        subtask_button_layout.addSpacing(10)
        subtask_button_layout.addWidget(self.copy_subtasks_btn)
        subtask_button_layout.addWidget(self.paste_subtasks_btn)
        subtask_button_layout.addStretch()
        subtask_list_layout.addLayout(subtask_button_layout, 1)
        subtask_list_group.setLayout(subtask_list_layout)
        subtask_editor_layout.addWidget(subtask_list_group)

        # 子任务参数配置区域
        self.subtask_param_group = QGroupBox("选中子任务参数 (仅编辑第一个选中项)") # 保存引用
        subtask_param_layout = QVBoxLayout()
        param_type_layout = QHBoxLayout()
        param_type_layout.addWidget(QLabel("类型:"))
        self.subtask_type_combo = QComboBox()
        for st_type in SubtaskType: self.subtask_type_combo.addItem(st_type.value, st_type)
        self.subtask_type_combo.currentIndexChanged.connect(self.on_subtask_type_changed)
        param_type_layout.addWidget(self.subtask_type_combo, 1)
        subtask_param_layout.addLayout(param_type_layout)
        self.subtask_param_stack = QStackedWidget()
        self._create_subtask_param_widgets()  # 创建页面
        subtask_param_layout.addWidget(self.subtask_param_stack)
        self.subtask_desc_edit = QLineEdit()
        self.subtask_desc_edit.setPlaceholderText("可选的描述信息，方便理解")
        desc_layout = QFormLayout();
        desc_layout.addRow("描述:", self.subtask_desc_edit)
        subtask_param_layout.addLayout(desc_layout)
        self.subtask_param_group.setLayout(subtask_param_layout)
        subtask_editor_layout.addWidget(self.subtask_param_group) # 使用保存的引用

        form_layout.addRow(self.subtask_editor_widget)
        self.subtask_editor_widget.setVisible(False) # 默认隐藏

        self.ai_driven_radio.toggled.connect(self.toggle_subtask_editor)
        if self.task and not self.task.use_ai_driver:
            self.subtask_driven_radio.setChecked(True);
            self.subtask_editor_widget.setVisible(True)
        else:
            self.ai_driven_radio.setChecked(True);
            self.subtask_editor_widget.setVisible(False)

        # --- 底部按钮 (不变) ---
        layout.addLayout(form_layout)
        button_layout = QHBoxLayout()
        save_btn = QPushButton("保存");
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消");
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch();
        button_layout.addWidget(save_btn);
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.setMinimumWidth(700)  # 可能需要更宽
        #self.adjustSize() #注释掉，否则对话框可能太小


    def _create_subtask_param_widgets(self):
        """为 QStackedWidget 创建所有子任务类型的参数页面。
        【已修改】为所有子任务添加 'label' 字段。为 check_* 类型添加跳转字段。
        """
        self._subtask_widgets = {}
        self._stacked_widget_map = {}

        for index, st_type in enumerate(SubtaskType):
            page_widget = QWidget()
            page_layout = QFormLayout(page_widget)
            page_layout.setContentsMargins(5, 5, 5, 5)
            widgets = {}

            # --- 通用标签字段 ---
            widgets['label'] = QLineEdit()
            widgets['label'].setPlaceholderText("(可选) 用于跳转目标，例如: step1_success")
            page_layout.addRow("标签 (Label):", widgets['label'])

            # --- 特定类型控件 ---
            if st_type == SubtaskType.COMMENT:
                widgets['text'] = QLineEdit()
                widgets['text'].setPlaceholderText("输入注释内容")
                page_layout.addRow("注释内容:", widgets['text'])
            # ... (existing cases for LOOP_START, LOOP_END, WAIT, etc. remain largely the same) ...
            elif st_type == SubtaskType.LOOP_START:
                widgets['count'] = QSpinBox(); widgets['count'].setRange(1, CONFIG.get("MAX_LOOP_ITERATIONS", 1000)); widgets['count'].setValue(3)
                page_layout.addRow("循环次数 (*):", widgets['count']); page_layout.addRow(QLabel("提示: 下方的 LOOP_END 标记此循环结束。"))
            elif st_type == SubtaskType.LOOP_END:
                page_layout.addRow(QLabel("标记上方最近的 LOOP_START 的结束位置。"))
            elif st_type == SubtaskType.WAIT:
                widgets['duration'] = QDoubleSpinBox(); widgets['duration'].setRange(0.1, 3600.0); widgets['duration'].setDecimals(1); widgets['duration'].setValue(1.0)
                page_layout.addRow("等待时长 (秒):", widgets['duration'])
            elif st_type == SubtaskType.FIND_AND_CLICK_TEXT:
                widgets['target_text'] = QLineEdit(); widgets['partial_match'] = QCheckBox("允许部分匹配"); widgets['partial_match'].setChecked(True); widgets['attempts'] = QSpinBox(); widgets['attempts'].setRange(1, 10); widgets['attempts'].setValue(3); widgets['timeout'] = QDoubleSpinBox(); widgets['timeout'].setRange(1.0, 60.0); widgets['timeout'].setValue(10.0); widgets['timeout'].setSuffix(" 秒")
                page_layout.addRow("目标文本 (*):", widgets['target_text']); page_layout.addRow(widgets['partial_match']); page_layout.addRow("尝试次数:", widgets['attempts']); page_layout.addRow("查找超时:", widgets['timeout'])
            elif st_type == SubtaskType.TEMPLATE_CLICK:
                widgets['template_name'] = QLineEdit(); widgets['threshold'] = QDoubleSpinBox(); widgets['threshold'].setRange(0.1, 1.0); widgets['threshold'].setDecimals(2); widgets['threshold'].setValue(0.75); widgets['attempts'] = QSpinBox(); widgets['attempts'].setRange(1, 10); widgets['attempts'].setValue(3); widgets['timeout'] = QDoubleSpinBox(); widgets['timeout'].setRange(1.0, 60.0); widgets['timeout'].setValue(10.0); widgets['timeout'].setSuffix(" 秒")
                page_layout.addRow("模板名称 (*):", widgets['template_name']); page_layout.addRow("匹配阈值:", widgets['threshold']); page_layout.addRow("尝试次数:", widgets['attempts']); page_layout.addRow("查找超时:", widgets['timeout'])
            elif st_type == SubtaskType.SWIPE: # (Swipe UI unchanged from your provided code)
                widgets['mode_radio_group'] = QGroupBox("模式"); mode_layout = QHBoxLayout(); widgets['direction_radio'] = QRadioButton("方向"); widgets['coords_radio'] = QRadioButton("坐标"); mode_layout.addWidget(widgets['direction_radio']); mode_layout.addWidget(widgets['coords_radio']); widgets['mode_radio_group'].setLayout(mode_layout); widgets['direction_radio'].setChecked(True)
                widgets['direction_combo'] = QComboBox(); widgets['direction_combo'].addItems(['up', 'down', 'left', 'right'])
                widgets['start_x'] = QLineEdit(); widgets['start_y'] = QLineEdit(); widgets['end_x'] = QLineEdit(); widgets['end_y'] = QLineEdit(); start_layout = QHBoxLayout(); start_layout.setContentsMargins(0,0,0,0); start_layout.addWidget(QLabel("X:")); start_layout.addWidget(widgets['start_x']); start_layout.addWidget(QLabel("Y:")); start_layout.addWidget(widgets['start_y']); start_widget = QWidget(); start_widget.setLayout(start_layout); widgets['_start_widget_ref'] = start_widget
                end_layout = QHBoxLayout(); end_layout.setContentsMargins(0,0,0,0); end_layout.addWidget(QLabel("X:")); end_layout.addWidget(widgets['end_x']); end_layout.addWidget(QLabel("Y:")); end_layout.addWidget(widgets['end_y']); end_widget = QWidget(); end_widget.setLayout(end_layout); widgets['_end_widget_ref'] = end_widget
                widgets['duration'] = QSpinBox(); widgets['duration'].setRange(100, 5000); widgets['duration'].setValue(500); widgets['duration'].setSuffix(" ms")
                page_layout.addRow(widgets['mode_radio_group']); page_layout.addRow("方向:", widgets['direction_combo']); page_layout.addRow("起始像素:", start_widget); page_layout.addRow("结束像素:", end_widget); page_layout.addRow("持续时间:", widgets['duration'])
                widgets['_page_layout_ref'] = page_layout
                self._toggle_swipe_widgets(True, widgets['direction_combo'], start_widget, end_widget, page_layout)
                widgets['direction_radio'].toggled.connect(lambda checked, dc=widgets['direction_combo'], sw=start_widget, ew=end_widget, pl=page_layout: self._toggle_swipe_widgets(checked, dc, sw, ew, pl))
            elif st_type == SubtaskType.BACK:
                page_layout.addRow(QLabel("模拟按下设备的返回键。"))
            elif st_type == SubtaskType.AI_STEP:
                widgets['goal'] = QLineEdit(); widgets['goal'].setPlaceholderText("例如：点击“下一步”按钮")
                page_layout.addRow("目标/指令 (*):", widgets['goal'])
            elif st_type == SubtaskType.ESP_COMMAND:
                widgets['command_string'] = QLineEdit(); widgets['command_string'].setPlaceholderText("例如: G X50 Y50 M1")
                page_layout.addRow("原始 ESP 命令 (*):", widgets['command_string'])

            # --- 修改 CHECK_* 类型以包含跳转标签 ---
            elif st_type == SubtaskType.CHECK_TEXT_EXISTS:
                widgets['target_text'] = QLineEdit(); widgets['partial_match'] = QCheckBox("允许部分匹配"); widgets['partial_match'].setChecked(True)
                # widgets['expected_result'] is now just for user info, not direct logic
                widgets['timeout'] = QDoubleSpinBox(); widgets['timeout'].setRange(0.5, 60.0); widgets['timeout'].setValue(5.0); widgets['timeout'].setSuffix(" 秒")
                page_layout.addRow("目标文本 (*):", widgets['target_text']); page_layout.addRow(widgets['partial_match']); page_layout.addRow("检查超时:", widgets['timeout'])
                # Jump labels
                widgets['on_success_jump_to_label'] = QLineEdit(); widgets['on_success_jump_to_label'].setPlaceholderText("(可选) 文本找到时跳转")
                widgets['on_failure_jump_to_label'] = QLineEdit(); widgets['on_failure_jump_to_label'].setPlaceholderText("(可选) 文本未找到时跳转")
                page_layout.addRow("成功跳转标签:", widgets['on_success_jump_to_label'])
                page_layout.addRow("失败跳转标签:", widgets['on_failure_jump_to_label'])

            elif st_type == SubtaskType.CHECK_TEMPLATE_EXISTS:
                widgets['template_name'] = QLineEdit(); widgets['threshold'] = QDoubleSpinBox(); widgets['threshold'].setRange(0.1, 1.0); widgets['threshold'].setDecimals(2); widgets['threshold'].setValue(0.75)
                widgets['timeout'] = QDoubleSpinBox(); widgets['timeout'].setRange(0.5, 60.0); widgets['timeout'].setValue(5.0); widgets['timeout'].setSuffix(" 秒")
                page_layout.addRow("模板名称 (*):", widgets['template_name']); page_layout.addRow("匹配阈值:", widgets['threshold']); page_layout.addRow("检查超时:", widgets['timeout'])
                # Jump labels
                widgets['on_success_jump_to_label'] = QLineEdit(); widgets['on_success_jump_to_label'].setPlaceholderText("(可选) 模板找到时跳转")
                widgets['on_failure_jump_to_label'] = QLineEdit(); widgets['on_failure_jump_to_label'].setPlaceholderText("(可选) 模板未找到时跳转")
                page_layout.addRow("成功跳转标签:", widgets['on_success_jump_to_label'])
                page_layout.addRow("失败跳转标签:", widgets['on_failure_jump_to_label'])
            else: # Catch-all for any other types (should ideally not happen if all enums covered)
                page_layout.addRow(QLabel(f"类型 '{st_type.value}' 参数配置。"))


            self.subtask_param_stack.addWidget(page_widget)
            self._stacked_widget_map[index] = st_type
            self._subtask_widgets[st_type] = widgets

        if self.subtask_type_combo.count() > 0:
             self.subtask_param_stack.setCurrentIndex(0)

    def _connect_subtask_param_signals_once(self):
        """【保持不变】只在初始化时调用一次，连接所有参数控件的信号。"""
        logger.debug("==> _connect_subtask_param_signals_once: 开始连接所有参数信号...")
        try: # 描述编辑框
            self.subtask_desc_edit.textChanged.connect(self._on_current_subtask_param_changed)
        except Exception as e:
             logger.error(f"    连接描述编辑框信号时出错: {e}", exc_info=False)

        for st_type_iter, widgets_iter in self._subtask_widgets.items():
            for widget_name, widget in widgets_iter.items():
                if widget_name.startswith('_'): continue
                try:
                    if isinstance(widget, QLineEdit):
                        widget.textChanged.connect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QCheckBox):
                        widget.stateChanged.connect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QSpinBox):
                        widget.valueChanged.connect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.valueChanged.connect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QComboBox):
                        widget.currentIndexChanged.connect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QRadioButton):
                        try: widget.toggled.disconnect(self._on_current_subtask_param_changed)
                        except TypeError: pass
                        widget.toggled.connect(self._on_current_subtask_param_changed)
                except Exception as e:
                    logger.error(f"      连接 {widget_name} 信号时出错: {e}", exc_info=False)
        logger.debug("<== _connect_subtask_param_signals_once: 完成连接所有参数信号。")

    def _toggle_swipe_widgets(self, is_direction_mode, direction_combo, start_widget, end_widget, page_layout):
        """【已修改】切换 Swipe 参数页面的控件显隐。传入的是包含布局的 Widget。"""
        direction_label = page_layout.labelForField(direction_combo)
        if direction_label: direction_label.setVisible(is_direction_mode)
        direction_combo.setVisible(is_direction_mode)

        start_label = page_layout.labelForField(start_widget)
        if start_label: start_label.setVisible(not is_direction_mode)
        start_widget.setVisible(not is_direction_mode)

        end_label = page_layout.labelForField(end_widget)
        if end_label: end_label.setVisible(not is_direction_mode)
        end_widget.setVisible(not is_direction_mode)

    def _connect_subtask_param_signals(self):
        """【修改】连接所有参数控件的信号到更新槽。增加安全检查。"""
        try: # 尝试断开旧连接，以防万一
            self.subtask_desc_edit.textChanged.disconnect(self._on_current_subtask_param_changed)
        except TypeError: pass
        # 重新连接描述编辑框
        self.subtask_desc_edit.textChanged.connect(self._on_current_subtask_param_changed)

        for st_type, widgets in self._subtask_widgets.items():
            for widget_name, widget in widgets.items():
                # 跳过内部引用
                if widget_name.startswith('_'): continue
                # 根据控件类型连接合适的信号
                try:
                    if isinstance(widget, QLineEdit):
                        widget.textChanged.disconnect(self._on_current_subtask_param_changed)
                except TypeError: pass
                if isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QCheckBox):
                    try: widget.stateChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.stateChanged.connect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QSpinBox):
                    try: widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.valueChanged.connect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QDoubleSpinBox):
                    try: widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.valueChanged.connect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QComboBox):
                    try: widget.currentIndexChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.currentIndexChanged.connect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QRadioButton):
                    # RadioButton 在组内切换时会触发 toggled
                    try: widget.toggled.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.toggled.connect(self._on_current_subtask_param_changed)
                # QGroupBox 不需要连接信号

    def _populate_existing_subtasks(self):
        """如果正在编辑任务，则填充现有的子任务列表。"""
        if self.task and not self.task.use_ai_driver and self.task.subtasks:
            for subtask_dict in self.task.subtasks:
                self._add_subtask_item(subtask_dict)
            # 选中第一个（如果存在）
            if self.subtask_list_widget.count() > 0:
                self.subtask_list_widget.setCurrentRow(0)
                # 确保选中后参数区域和按钮状态正确更新
                self.on_subtask_selection_changed(self.subtask_list_widget.currentItem(), None)
                self.on_subtask_multi_selection_changed()
            else:
                # 如果没有子任务，禁用参数编辑区和部分按钮
                self.subtask_param_group.setEnabled(False)
                self.remove_subtask_btn.setEnabled(False)
                self.move_up_btn.setEnabled(False)
                self.move_down_btn.setEnabled(False)
                self.copy_subtasks_btn.setEnabled(False)

    def _add_subtask_item(self, subtask_data: Dict[str, Any]):
        """向 QListWidget 添加一个子任务项，并存储数据。"""
        list_item = QListWidgetItem()
        self._update_list_item_display(list_item, subtask_data)  # 设置显示文本
        list_item.setData(Qt.UserRole, subtask_data)  # 存储完整数据
        self.subtask_list_widget.addItem(list_item)

    def _update_list_item_display(self, list_item: QListWidgetItem, subtask_data: Dict[str, Any]):
        """根据子任务数据更新 QListWidgetItem 的显示文本。
        【已修改】显示标签和跳转信息。
        """
        st_type_str = subtask_data.get('type', '未知类型')
        desc = subtask_data.get('description', '')
        label = subtask_data.get('label', '')
        details = ""
        prefix = "➡️ " # Default prefix

        try:
            st_type = SubtaskType(st_type_str)
            # --- 特殊标记和细节 ---
            if label: prefix = f"🏷️({label}) " # Label prefix

            if st_type == SubtaskType.COMMENT:
                details = f"注释: {subtask_data.get('text', '')[:30]}"
                prefix = "💬 "
            elif st_type == SubtaskType.LOOP_START:
                details = f"循环 {subtask_data.get('count', '?')} 次"; prefix += "↪️ "
            elif st_type == SubtaskType.LOOP_END:
                details = "结束循环"; prefix += "↩️ "
            elif st_type == SubtaskType.WAIT: details = f"{subtask_data.get('duration', '?')}s"
            elif st_type == SubtaskType.FIND_AND_CLICK_TEXT: details = f"点击文本'{subtask_data.get('target_text', '?')}'"
            elif st_type == SubtaskType.TEMPLATE_CLICK: details = f"点击模板'{subtask_data.get('template_name', '?')}'"
            elif st_type == SubtaskType.SWIPE:
                if subtask_data.get('direction'): details = f"滑动({subtask_data.get('direction')})"
                elif subtask_data.get('start') and subtask_data.get('end'): details = f"滑动({subtask_data.get('start')}->{subtask_data.get('end')})"
                else: details = "滑动(?)"
            elif st_type == SubtaskType.BACK: details = "返回键"
            elif st_type == SubtaskType.AI_STEP:
                goal = subtask_data.get('goal', '?'); details = f"AI目标: {goal[:20]}{'...' if len(goal)>20 else ''}"
            elif st_type == SubtaskType.ESP_COMMAND:
                cmd = subtask_data.get('command_string', '?'); details = f"ESP: {cmd[:20]}{'...' if len(cmd)>20 else ''}"
            elif st_type == SubtaskType.CHECK_TEXT_EXISTS or st_type == SubtaskType.CHECK_TEMPLATE_EXISTS:
                target = subtask_data.get('target_text') or subtask_data.get('template_name', '?')
                details = f"检查'{target[:15]}{'...' if len(target)>15 else ''}'"
                prefix = "❔ " + prefix.replace("➡️ ", "") # Question mark prefix for checks
                jump_info = []
                if subtask_data.get('on_success_jump_to_label'): jump_info.append(f"✅↪️{subtask_data['on_success_jump_to_label']}")
                if subtask_data.get('on_failure_jump_to_label'): jump_info.append(f"❌↪️{subtask_data['on_failure_jump_to_label']}")
                if jump_info: details += f" ({', '.join(jump_info)})"
        except ValueError:
            pass # Unknown type, will use default display

        display_text = f"{prefix}{st_type_str}: {details}"
        if desc: display_text += f" ({desc})"
        list_item.setText(display_text)
        tooltip_text = json.dumps(subtask_data, indent=2, ensure_ascii=False)
        list_item.setToolTip(tooltip_text)

    def on_subtask_type_changed(self, index):
        """当子任务类型下拉框变化时，切换 QStackedWidget 页面。"""
        if index in self._stacked_widget_map:
            st_type = self._stacked_widget_map[index]
            logger.debug(f"子任务类型更改为: {st_type.value} (索引 {index})")
            self.subtask_param_stack.setCurrentIndex(index)
            # 当类型改变时，主动触发一次参数更新，以使用新类型的默认值更新当前选中的列表项（如果有选中）
            logger.debug("类型更改后，调用 _on_current_subtask_param_changed 更新数据")
            self._on_current_subtask_param_changed()
        else:
            logger.warning(f"无法找到索引 {index} 对应的子任务参数页面。")

    def on_subtask_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """
        【最终关键修改】直接使用信号传递的 'current' 参数获取数据，而不是 selectedItems()。
        """
        current_text = current.text().split('\n')[0] if current else "None"
        previous_text = previous.text().split('\n')[0] if previous else "None"
        logger.debug(f"on_subtask_selection_changed: current='{current_text}', previous='{previous_text}'")

        # ===> 修改点：直接使用信号传递的 'current' 参数 <===
        # 不再使用 self.subtask_list_widget.selectedItems()
        target_item = current # 直接使用信号提供的当前项
        # ==================================================

        if target_item:
            item_text_for_log = target_item.text().split(chr(10))[0] # 使用 target_item
            logger.debug(f"  处理信号提供的当前项: '{item_text_for_log}'. 启用参数编辑并加载数据...")
            self.subtask_param_group.setEnabled(True)
            # ===> 从 'target_item' (即 'current') 获取数据 <===
            subtask_data = target_item.data(Qt.UserRole)
            # ==============================================
            if isinstance(subtask_data, dict):
                self._load_subtask_data_to_widgets(subtask_data) # 调用加载
            else:
                # 获取数据失败
                logger.warning(f"  当前项 '{item_text_for_log}' 没有有效的字典数据。清空参数区。")
                self._load_subtask_data_to_widgets({}) # 清空
                self.subtask_param_group.setEnabled(False)
        else:
            # current 参数为 None 的情况 (例如列表清空时)
            logger.debug("  信号提供的当前项为 None，禁用参数编辑区并清空。")
            self.subtask_param_group.setEnabled(False)
            self._load_subtask_data_to_widgets({})

        logger.debug(f"on_subtask_selection_changed: 处理完毕。")

    def _disconnect_all_param_signals(self):
        """【已修改】临时断开所有参数控件的信号连接。增加日志和健壮性。"""
        logger.debug("==> 开始断开所有参数控件信号...")
        try:  # 描述编辑框
            self.subtask_desc_edit.textChanged.disconnect(self._on_current_subtask_param_changed)
            logger.debug("    断开描述编辑框信号。")
        except TypeError:
            # logger.debug("    描述编辑框信号已断开或未连接。")
            pass # 忽略错误，说明已经断开或者从未连接

        # 遍历所有类型的控件字典
        for st_type_iter, widgets_iter in self._subtask_widgets.items():
            # logger.debug(f"    处理类型: {st_type_iter.value}")
            for widget_name, widget in widgets_iter.items():
                if widget_name.startswith('_'): continue  # 跳过内部引用 (如 _page_layout_ref)
                try:
                    # 根据控件类型尝试断开对应的主要信号
                    if isinstance(widget, QLineEdit):
                        widget.textChanged.disconnect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QCheckBox):
                        widget.stateChanged.disconnect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QSpinBox):
                        widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QComboBox):
                        widget.currentIndexChanged.disconnect(self._on_current_subtask_param_changed)
                    elif isinstance(widget, QRadioButton):
                        # RadioButton 的 toggled 信号比较特殊，确保只在必要时连接/断开
                        widget.toggled.disconnect(self._on_current_subtask_param_changed)
                    # QGroupBox 不需要断开
                    # logger.debug(f"      成功断开 {widget_name} ({type(widget).__name__}) 信号。")
                except TypeError:
                    # logger.debug(f"      信号 for {widget_name} ({type(widget).__name__}) 已断开或未连接。")
                    pass  # 忽略断开未连接的信号
                except Exception as e:
                    # 捕获其他潜在错误
                    logger.error(f"      断开 {widget_name} 信号时发生未知错误: {e}", exc_info=False)
        logger.debug("<== 完成断开所有参数控件信号。")

    def on_subtask_multi_selection_changed(self):
        """【新增】当列表选择发生变化时（单选或多选），更新按钮状态。"""
        selected_items = self.subtask_list_widget.selectedItems()
        count = len(selected_items)
        total_rows = self.subtask_list_widget.count()

        # 编辑按钮：只有单选时启用
        # self.subtask_param_group.setEnabled(count == 1) # 这个在 on_subtask_selection_changed 中处理

        # 移除、复制按钮：至少选中一项
        self.remove_subtask_btn.setEnabled(count > 0)
        self.copy_subtasks_btn.setEnabled(count > 0)

        # 上移按钮：选中项 > 0 且 没有选中第一行
        can_move_up = count > 0 and all(self.subtask_list_widget.row(item) > 0 for item in selected_items)
        self.move_up_btn.setEnabled(can_move_up)

        # 下移按钮：选中项 > 0 且 没有选中最后一行
        can_move_down = count > 0 and all(self.subtask_list_widget.row(item) < total_rows - 1 for item in selected_items)
        self.move_down_btn.setEnabled(can_move_down)

        # 粘贴按钮：检查剪贴板内容
        clipboard = QApplication.clipboard()
        self.paste_subtasks_btn.setEnabled(clipboard.mimeData().hasText())
        logger.debug(f"更新按钮状态: count={count}, up={can_move_up}, down={can_move_down}, paste={self.paste_subtasks_btn.isEnabled()}")

    def _load_subtask_data_to_widgets(self, subtask_data: Dict[str, Any]):
        """将子任务数据加载到参数控件中。
        【已修改】处理新的 label 和跳转字段。
        """
        if self._is_loading_data: return
        self._is_loading_data = True
        try:
            st_type_str = subtask_data.get('type')
            found_index = -1
            st_type_enum = None
            if st_type_str:
                try:
                    st_type_enum = SubtaskType(st_type_str)
                    for i in range(self.subtask_type_combo.count()):
                        if self.subtask_type_combo.itemData(i) == st_type_enum:
                            found_index = i; break
                except ValueError: pass # Invalid type string

            if found_index == -1 and self.subtask_type_combo.count() > 0:
                found_index = 0 # Default to first type if not found or invalid
                st_type_enum = self.subtask_type_combo.itemData(found_index)
            elif self.subtask_type_combo.count() == 0:
                 self._is_loading_data = False; return

            self.subtask_type_combo.setCurrentIndex(found_index)
            self.subtask_param_stack.setCurrentIndex(found_index)
            QApplication.processEvents() # Ensure stack widget updates

            self.subtask_desc_edit.setText(subtask_data.get('description', ''))

            current_widgets = self._subtask_widgets.get(st_type_enum, {})
            if not current_widgets: self._is_loading_data = False; return

            # --- 通用标签加载 ---
            if 'label' in current_widgets:
                current_widgets['label'].setText(subtask_data.get('label', ''))

            # --- 特定类型参数加载 ---
            if st_type_enum == SubtaskType.COMMENT:
                if 'text' in current_widgets: current_widgets['text'].setText(subtask_data.get('text', ''))
            # ... (existing cases for LOOP_START, WAIT etc.) ...
            elif st_type_enum == SubtaskType.LOOP_START:
                if 'count' in current_widgets: current_widgets['count'].setValue(int(subtask_data.get('count', 3)))
            elif st_type_enum == SubtaskType.WAIT:
                if 'duration' in current_widgets: current_widgets['duration'].setValue(float(subtask_data.get('duration', 1.0)))
            elif st_type_enum == SubtaskType.FIND_AND_CLICK_TEXT:
                if 'target_text' in current_widgets: current_widgets['target_text'].setText(subtask_data.get('target_text', ''))
                if 'partial_match' in current_widgets: current_widgets['partial_match'].setChecked(subtask_data.get('partial_match', True))
                if 'attempts' in current_widgets: current_widgets['attempts'].setValue(int(subtask_data.get('attempts', 3)))
                if 'timeout' in current_widgets: current_widgets['timeout'].setValue(float(subtask_data.get('timeout', 10.0)))
            elif st_type_enum == SubtaskType.TEMPLATE_CLICK:
                if 'template_name' in current_widgets: current_widgets['template_name'].setText(subtask_data.get('template_name', ''))
                if 'threshold' in current_widgets: current_widgets['threshold'].setValue(float(subtask_data.get('threshold', 0.75)))
                if 'attempts' in current_widgets: current_widgets['attempts'].setValue(int(subtask_data.get('attempts', 3)))
                if 'timeout' in current_widgets: current_widgets['timeout'].setValue(float(subtask_data.get('timeout', 10.0)))
            elif st_type_enum == SubtaskType.SWIPE: # (Swipe loading unchanged)
                is_direction = 'direction' in subtask_data and subtask_data['direction'] is not None;
                if 'direction_radio' in current_widgets: current_widgets['direction_radio'].setChecked(is_direction)
                if 'coords_radio' in current_widgets: current_widgets['coords_radio'].setChecked(not is_direction)
                if is_direction:
                    direction = subtask_data.get('direction', 'down'); index = current_widgets['direction_combo'].findText(direction); current_widgets['direction_combo'].setCurrentIndex(index if index >=0 else 0)
                    current_widgets['start_x'].clear(); current_widgets['start_y'].clear(); current_widgets['end_x'].clear(); current_widgets['end_y'].clear()
                else:
                    start = subtask_data.get('start', [0, 0]); end = subtask_data.get('end', [0, 0])
                    current_widgets['start_x'].setText(str(start[0] if len(start)>0 else 0)); current_widgets['start_y'].setText(str(start[1] if len(start)>1 else 0))
                    current_widgets['end_x'].setText(str(end[0] if len(end)>0 else 0)); current_widgets['end_y'].setText(str(end[1] if len(end)>1 else 0))
                    current_widgets['direction_combo'].setCurrentIndex(0)
                current_widgets['duration'].setValue(int(subtask_data.get('duration', 500)))
                page_layout = current_widgets.get('_page_layout_ref'); start_widget = current_widgets.get('_start_widget_ref'); end_widget = current_widgets.get('_end_widget_ref'); direction_combo = current_widgets.get('direction_combo')
                if page_layout and start_widget and end_widget and direction_combo: self._toggle_swipe_widgets(current_widgets['direction_radio'].isChecked(), direction_combo, start_widget, end_widget, page_layout)
            elif st_type_enum == SubtaskType.AI_STEP:
                if 'goal' in current_widgets: current_widgets['goal'].setText(subtask_data.get('goal', ''))
            elif st_type_enum == SubtaskType.ESP_COMMAND:
                if 'command_string' in current_widgets: current_widgets['command_string'].setText(subtask_data.get('command_string', ''))
            elif st_type_enum == SubtaskType.CHECK_TEXT_EXISTS:
                if 'target_text' in current_widgets: current_widgets['target_text'].setText(subtask_data.get('target_text', ''))
                if 'partial_match' in current_widgets: current_widgets['partial_match'].setChecked(subtask_data.get('partial_match', True))
                if 'timeout' in current_widgets: current_widgets['timeout'].setValue(float(subtask_data.get('timeout', 5.0)))
                if 'on_success_jump_to_label' in current_widgets: current_widgets['on_success_jump_to_label'].setText(subtask_data.get('on_success_jump_to_label', ''))
                if 'on_failure_jump_to_label' in current_widgets: current_widgets['on_failure_jump_to_label'].setText(subtask_data.get('on_failure_jump_to_label', ''))
            elif st_type_enum == SubtaskType.CHECK_TEMPLATE_EXISTS:
                if 'template_name' in current_widgets: current_widgets['template_name'].setText(subtask_data.get('template_name', ''))
                if 'threshold' in current_widgets: current_widgets['threshold'].setValue(float(subtask_data.get('threshold', 0.75)))
                if 'timeout' in current_widgets: current_widgets['timeout'].setValue(float(subtask_data.get('timeout', 5.0)))
                if 'on_success_jump_to_label' in current_widgets: current_widgets['on_success_jump_to_label'].setText(subtask_data.get('on_success_jump_to_label', ''))
                if 'on_failure_jump_to_label' in current_widgets: current_widgets['on_failure_jump_to_label'].setText(subtask_data.get('on_failure_jump_to_label', ''))

        except Exception as e:
            logger.error(f"Error loading subtask data to widgets: {e}", exc_info=True)
        finally:
            self._is_loading_data = False

    def _disconnect_param_signals(self, st_type: SubtaskType):
        """临时断开指定类型参数控件的信号连接。"""
        if st_type not in self._subtask_widgets: return
        widgets = self._subtask_widgets[st_type]
        try:
            self.subtask_desc_edit.textChanged.disconnect(self._on_current_subtask_param_changed)
        except TypeError:
            pass
        for widget in widgets.values():
            try:
                if isinstance(widget, QLineEdit):
                    widget.textChanged.disconnect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.disconnect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QSpinBox):
                    widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QDoubleSpinBox):
                    widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QComboBox):
                    widget.currentIndexChanged.disconnect(self._on_current_subtask_param_changed)
                elif isinstance(widget, QRadioButton):
                    widget.toggled.disconnect(self._on_current_subtask_param_changed)
            except TypeError:
                pass  # 忽略断开未连接的信号

    def _reconnect_param_signals(self, st_type: SubtaskType):
        """【已修改】重新连接指定类型参数控件的信号。增加健壮性和日志。"""
        if not isinstance(st_type, SubtaskType):
            logger.error(f"_reconnect_param_signals: 提供的类型无效: {st_type}")
            return

        logger.debug(f"==> 开始重新连接类型 '{st_type.value}' 的信号...")
        try: # 描述编辑框
            # 先尝试断开，避免重复连接（虽然理论上不应该发生）
            try: self.subtask_desc_edit.textChanged.disconnect(self._on_current_subtask_param_changed)
            except TypeError: pass
            self.subtask_desc_edit.textChanged.connect(self._on_current_subtask_param_changed)
            logger.debug("    重新连接描述编辑框信号。")
        except Exception as e:
             logger.error(f"    重新连接描述编辑框信号时出错: {e}", exc_info=False)


        if st_type not in self._subtask_widgets:
            logger.warning(f"    未找到类型 '{st_type.value}' 的控件字典，无法重新连接信号。")
            return

        widgets = self._subtask_widgets[st_type]
        for widget_name, widget in widgets.items():
            if widget_name.startswith('_'): continue # 跳过内部引用
            try:
                # 根据控件类型重新连接信号
                if isinstance(widget, QLineEdit):
                    # 先尝试断开
                    try: widget.textChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.textChanged.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QLineEdit) 的 textChanged 信号。")
                elif isinstance(widget, QCheckBox):
                    try: widget.stateChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.stateChanged.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QCheckBox) 的 stateChanged 信号。")
                elif isinstance(widget, QSpinBox):
                    try: widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.valueChanged.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QSpinBox) 的 valueChanged 信号。")
                elif isinstance(widget, QDoubleSpinBox):
                    try: widget.valueChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.valueChanged.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QDoubleSpinBox) 的 valueChanged 信号。")
                elif isinstance(widget, QComboBox):
                    try: widget.currentIndexChanged.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    widget.currentIndexChanged.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QComboBox) 的 currentIndexChanged 信号。")
                elif isinstance(widget, QRadioButton):
                    # RadioButton 的 toggled 信号需要特别小心，因为它在状态改变时都会触发
                    # 确保在组内切换时能正确更新数据
                    try: widget.toggled.disconnect(self._on_current_subtask_param_changed)
                    except TypeError: pass
                    # 只有当它被选中时，它的 toggled(True) 才应该触发更新，
                    # 但连接 toggled 通常是安全的，因为槽函数内部会读取当前状态
                    widget.toggled.connect(self._on_current_subtask_param_changed)
                    # logger.debug(f"    重新连接 {widget_name} (QRadioButton) 的 toggled 信号。")
            except Exception as e:
                 logger.error(f"    重新连接 {widget_name} 信号时发生错误: {e}", exc_info=False)

        logger.debug(f"<== 完成重新连接类型 '{st_type.value}' 的信号。")

    def _on_current_subtask_param_changed(self, *args):
        """当当前选中的子任务参数被修改时，更新 QListWidgetItem 中的数据。
        【已修改】保存新的 label 和跳转字段。
        """
        if self._is_loading_data: return

        selected_items = self.subtask_list_widget.selectedItems()
        if not selected_items: return
        current_item = selected_items[0]
        if not current_item: return

        current_type_index = self.subtask_type_combo.currentIndex()
        if current_type_index < 0: return
        st_type = self.subtask_type_combo.itemData(current_type_index)
        if not st_type or not isinstance(st_type, SubtaskType): return

        widgets = self._subtask_widgets.get(st_type)
        if not widgets: return

        new_subtask_data = {'type': st_type.value}
        try:
            new_subtask_data['description'] = self.subtask_desc_edit.text().strip()
            # --- 通用标签保存 ---
            if 'label' in widgets:
                new_subtask_data['label'] = widgets['label'].text().strip()
                if not new_subtask_data['label']: del new_subtask_data['label'] # 如果为空则不保存

            # --- 特定类型参数保存 ---
            if st_type == SubtaskType.COMMENT:
                if 'text' in widgets: new_subtask_data['text'] = widgets['text'].text().strip()
            # ... (existing cases for LOOP_START, WAIT etc.) ...
            elif st_type == SubtaskType.LOOP_START:
                if 'count' in widgets: new_subtask_data['count'] = widgets['count'].value()
            elif st_type == SubtaskType.WAIT:
                if 'duration' in widgets: new_subtask_data['duration'] = widgets['duration'].value()
            elif st_type == SubtaskType.FIND_AND_CLICK_TEXT:
                if 'target_text' in widgets: new_subtask_data['target_text'] = widgets['target_text'].text().strip()
                if 'partial_match' in widgets: new_subtask_data['partial_match'] = widgets['partial_match'].isChecked()
                if 'attempts' in widgets: new_subtask_data['attempts'] = widgets['attempts'].value()
                if 'timeout' in widgets: new_subtask_data['timeout'] = widgets['timeout'].value()
            elif st_type == SubtaskType.TEMPLATE_CLICK:
                if 'template_name' in widgets: new_subtask_data['template_name'] = widgets['template_name'].text().strip()
                if 'threshold' in widgets: new_subtask_data['threshold'] = widgets['threshold'].value()
                if 'attempts' in widgets: new_subtask_data['attempts'] = widgets['attempts'].value()
                if 'timeout' in widgets: new_subtask_data['timeout'] = widgets['timeout'].value()
            elif st_type == SubtaskType.SWIPE: # (Swipe saving unchanged)
                if 'duration' in widgets: new_subtask_data['duration'] = widgets['duration'].value()
                if widgets['direction_radio'].isChecked():
                    new_subtask_data['direction'] = widgets['direction_combo'].currentText()
                else:
                    sx = int(widgets['start_x'].text() or 0); sy = int(widgets['start_y'].text() or 0)
                    ex = int(widgets['end_x'].text() or 0); ey = int(widgets['end_y'].text() or 0)
                    new_subtask_data['start'] = [sx, sy]; new_subtask_data['end'] = [ex, ey]
            elif st_type == SubtaskType.AI_STEP:
                if 'goal' in widgets: new_subtask_data['goal'] = widgets['goal'].text().strip()
            elif st_type == SubtaskType.ESP_COMMAND:
                if 'command_string' in widgets: new_subtask_data['command_string'] = widgets['command_string'].text().strip()
            elif st_type == SubtaskType.CHECK_TEXT_EXISTS:
                if 'target_text' in widgets: new_subtask_data['target_text'] = widgets['target_text'].text().strip()
                if 'partial_match' in widgets: new_subtask_data['partial_match'] = widgets['partial_match'].isChecked()
                if 'timeout' in widgets: new_subtask_data['timeout'] = widgets['timeout'].value()
                if 'on_success_jump_to_label' in widgets:
                    val = widgets['on_success_jump_to_label'].text().strip()
                    if val: new_subtask_data['on_success_jump_to_label'] = val
                if 'on_failure_jump_to_label' in widgets:
                    val = widgets['on_failure_jump_to_label'].text().strip()
                    if val: new_subtask_data['on_failure_jump_to_label'] = val
            elif st_type == SubtaskType.CHECK_TEMPLATE_EXISTS:
                if 'template_name' in widgets: new_subtask_data['template_name'] = widgets['template_name'].text().strip()
                if 'threshold' in widgets: new_subtask_data['threshold'] = widgets['threshold'].value()
                if 'timeout' in widgets: new_subtask_data['timeout'] = widgets['timeout'].value()
                if 'on_success_jump_to_label' in widgets:
                    val = widgets['on_success_jump_to_label'].text().strip()
                    if val: new_subtask_data['on_success_jump_to_label'] = val
                if 'on_failure_jump_to_label' in widgets:
                    val = widgets['on_failure_jump_to_label'].text().strip()
                    if val: new_subtask_data['on_failure_jump_to_label'] = val

            current_item.setData(Qt.UserRole, new_subtask_data)
            self._update_list_item_display(current_item, new_subtask_data)
        except ValueError as e: logger.warning(f"读取参数控件值时发生转换错误: {e}")
        except KeyError as e: logger.error(f"读取参数控件值时发生Key错误: {e}", exc_info=True)
        except Exception as e: logger.error(f"读取参数控件值时发生未知错误: {e}", exc_info=True)

    def on_add_subtask(self):
        """添加一个新的子任务到列表末尾。"""
        # 使用当前选中的类型（或默认第一个类型）
        current_type_index = self.subtask_type_combo.currentIndex()
        if current_type_index < 0:
            if self.subtask_type_combo.count() > 0:
                current_type_index = 0
            else:
                QMessageBox.warning(self, "错误", "没有可用的子任务类型。"); return

        st_type = self.subtask_type_combo.itemData(current_type_index)
        if not st_type: return

        # 创建一个包含默认值的子任务字典
        default_subtask_data = {'type': st_type.value, 'description': ''}
        # 可以为特定类型添加更多默认值
        if st_type == SubtaskType.WAIT:
            default_subtask_data['duration'] = 1.0
        elif st_type == SubtaskType.FIND_AND_CLICK_TEXT:
            default_subtask_data.update({'target_text': '', 'partial_match': True, 'attempts': 3, 'timeout': 10.0})
        elif st_type == SubtaskType.TEMPLATE_CLICK:
            default_subtask_data.update({'template_name': '', 'threshold': 0.75, 'attempts': 3, 'timeout': 10.0})
        elif st_type == SubtaskType.SWIPE:
            default_subtask_data.update({'direction': 'down', 'duration': 500})  # 默认方向
        elif st_type == SubtaskType.AI_STEP:
            default_subtask_data['goal'] = ''
        elif st_type == SubtaskType.ESP_COMMAND:
            default_subtask_data['command_string'] = ''
        elif st_type == SubtaskType.CHECK_TEXT_EXISTS:
            default_subtask_data.update(
                {'target_text': '', 'partial_match': True, 'expected_result': True, 'timeout': 5.0})
        elif st_type == SubtaskType.CHECK_TEMPLATE_EXISTS:
            default_subtask_data.update(
                {'template_name': '', 'threshold': 0.75, 'expected_result': True, 'timeout': 5.0})

        # 添加到列表
        self._add_subtask_item(default_subtask_data)
        # 选中新添加的项
        self.subtask_list_widget.setCurrentRow(self.subtask_list_widget.count() - 1)

    def on_remove_subtask(self):
        """【已修改】移除列表中所有选中的子任务。"""
        selected_items = self.subtask_list_widget.selectedItems()
        if not selected_items: return
        reply = QMessageBox.question(self, "确认移除", f"确定要移除选中的 {len(selected_items)} 个子任务吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No: return

        rows_to_remove = sorted([self.subtask_list_widget.row(item) for item in selected_items], reverse=True)
        for row in rows_to_remove:
            self.subtask_list_widget.takeItem(row)
        # 更新选中状态和按钮
        self.on_subtask_selection_changed(self.subtask_list_widget.currentItem(), None)
        self.on_subtask_multi_selection_changed()

    def on_move_subtask_up(self):
        """【已修改】将所有选中的子任务向上移动一位。"""
        selected_items = self.subtask_list_widget.selectedItems()
        if not selected_items: return
        rows_to_move = sorted([self.subtask_list_widget.row(item) for item in selected_items])
        if rows_to_move[0] == 0: return # 不能移动第一行

        self.subtask_list_widget.setUpdatesEnabled(False) # 优化性能
        try:
            for row in rows_to_move:
                item = self.subtask_list_widget.takeItem(row)
                self.subtask_list_widget.insertItem(row - 1, item)
            # 重新选中移动后的项 (保持选中状态)
            for item in selected_items:
                item.setSelected(True)
            # 确保第一个移动的项是当前项
            if selected_items:
                 self.subtask_list_widget.setCurrentItem(selected_items[0])
        finally:
            self.subtask_list_widget.setUpdatesEnabled(True)
        self.on_subtask_multi_selection_changed() # 更新按钮状态

    def on_move_subtask_down(self):
        """【已修改】将所有选中的子任务向下移动一位。"""
        selected_items = self.subtask_list_widget.selectedItems()
        if not selected_items: return
        rows_to_move = sorted([self.subtask_list_widget.row(item) for item in selected_items], reverse=True)
        if rows_to_move[0] == self.subtask_list_widget.count() - 1: return # 不能移动最后一行

        self.subtask_list_widget.setUpdatesEnabled(False) # 优化性能
        try:
            for row in rows_to_move:
                item = self.subtask_list_widget.takeItem(row)
                self.subtask_list_widget.insertItem(row + 1, item)
            # 重新选中移动后的项
            for item in selected_items:
                item.setSelected(True)
            # 确保第一个移动的项是当前项
            if selected_items:
                 self.subtask_list_widget.setCurrentItem(selected_items[0])
        finally:
            self.subtask_list_widget.setUpdatesEnabled(True)
        self.on_subtask_multi_selection_changed() # 更新按钮状态

    def on_copy_subtasks(self):
        """【新增】复制选中的子任务数据到剪贴板 (JSON格式)。"""
        selected_items = self.subtask_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要复制的子任务。")
            return

        rows = sorted([self.subtask_list_widget.row(item) for item in selected_items])
        subtasks_to_copy = []
        for row in rows:
            item = self.subtask_list_widget.item(row)
            data = item.data(Qt.UserRole)
            if isinstance(data, dict):
                subtasks_to_copy.append(data)

        if not subtasks_to_copy:
            QMessageBox.warning(self, "错误", "选中的项目没有有效的子任务数据。")
            return

        try:
            json_data = json.dumps(subtasks_to_copy, indent=2, ensure_ascii=False)
            clipboard = QApplication.clipboard()
            clipboard.setText(json_data)
            logger.info(f"已复制 {len(subtasks_to_copy)} 个子任务到剪贴板。")
            # # 暂时不在日志区显示，避免干扰
            # self.task_progress_text.append(
            #     f"[{datetime.now().strftime('%H:%M:%S')}] {len(subtasks_to_copy)} 个子任务已复制。")
            self.on_subtask_multi_selection_changed()  # 更新粘贴按钮状态
        except Exception as e:
            logger.error(f"复制子任务到剪贴板时出错: {e}", exc_info=True)
            QMessageBox.critical(self, "复制错误", f"无法将子任务序列化为JSON: {e}")

    def on_paste_subtasks(self):
        """【新增】从剪贴板粘贴子任务数据。"""
        clipboard = QApplication.clipboard()
        json_data = clipboard.text()
        if not json_data:
            QMessageBox.information(self, "提示", "剪贴板为空或内容不是文本。")
            return

        try:
            pasted_subtasks = json.loads(json_data)
            if not isinstance(pasted_subtasks, list):
                raise ValueError("剪贴板内容不是一个有效的子任务列表 (JSON Array)。")

            valid_subtasks = []
            for i, data in enumerate(pasted_subtasks):
                if not isinstance(data, dict) or 'type' not in data:
                    raise ValueError(f"剪贴板中的第 {i + 1} 项不是有效的子任务字典或缺少 'type'。")
                try:
                    SubtaskType(data['type'])
                except ValueError:
                    raise ValueError(f"剪贴板中的第 {i + 1} 项包含无效的子任务类型: {data['type']}")
                valid_subtasks.append(data)

            if not valid_subtasks:
                QMessageBox.warning(self, "粘贴失败", "剪贴板内容解析成功，但未包含有效的子任务数据。")
                return

            current_row = self.subtask_list_widget.currentRow()
            insert_row = current_row + 1 if current_row >= 0 else self.subtask_list_widget.count()

            self.subtask_list_widget.setUpdatesEnabled(False)
            items_to_select = []
            for data in reversed(valid_subtasks):
                list_item = QListWidgetItem()
                self._update_list_item_display(list_item, data)
                list_item.setData(Qt.UserRole, data)
                self.subtask_list_widget.insertItem(insert_row, list_item)
                items_to_select.append(list_item)
            self.subtask_list_widget.setUpdatesEnabled(True)

            self.subtask_list_widget.clearSelection()
            for item in reversed(items_to_select):
                item.setSelected(True)
            if items_to_select:
                self.subtask_list_widget.setCurrentItem(items_to_select[-1])

            logger.info(f"已从剪贴板粘贴 {len(valid_subtasks)} 个子任务。")
            # # 不在日志区显示
            # self.task_progress_text.append(
            #     f"[{datetime.now().strftime('%H:%M:%S')}] {len(valid_subtasks)} 个子任务已粘贴。")
            self.on_subtask_multi_selection_changed()

        except json.JSONDecodeError:
            QMessageBox.warning(self, "粘贴失败", "剪贴板内容不是有效的 JSON 格式。")
        except ValueError as e:
            QMessageBox.warning(self, "粘贴失败", f"剪贴板内容格式错误: {e}")
        except Exception as e:
            logger.error(f"粘贴子任务时出错: {e}", exc_info=True); QMessageBox.critical(self, "粘贴错误",
                                                                                        f"粘贴子任务时发生未知错误: {e}")

    # --- 其他函数 (toggle_subtask_editor, get_task_info, _validate_subtask_data, accept) 保持不变 ---
    def toggle_subtask_editor(self):
        """根据选择的执行模式，显示或隐藏子任务编辑器。"""
        is_subtask_mode = self.subtask_driven_radio.isChecked()
        self.subtask_editor_widget.setVisible(is_subtask_mode)
        # self.adjustSize() # 调整对话框大小以适应内容变化, 暂时注释，可能导致窗口过小
        if not is_subtask_mode:
            self.subtask_param_group.setEnabled(False)
            self.remove_subtask_btn.setEnabled(False)
            self.move_up_btn.setEnabled(False)
            self.move_down_btn.setEnabled(False)
            self.copy_subtasks_btn.setEnabled(False)
            self.paste_subtasks_btn.setEnabled(False)  # 切换模式时也禁用粘贴
        else:
            self.on_subtask_selection_changed(self.subtask_list_widget.currentItem(), None)
            self.on_subtask_multi_selection_changed()  # 切换回来时更新按钮状态

    def get_task_info(self) -> Optional[Dict[str, Any]]:
        """获取任务信息，如果是子任务模式，则从 QListWidget 提取数据。"""
        name = self.name_edit.text().strip()
        if not name: QMessageBox.warning(self, "输入错误", "任务名称不能为空。"); return None

        task_type_str = self.type_combo.currentText()
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            QMessageBox.warning(self, "输入错误", f"无效的任务类型: {task_type_str}"); return None

        app_name = self.app_edit.text().strip()
        priority = self.priority_spin.value()
        max_retries = self.max_retries_spin.value()
        device_name = self.device_combo.currentData()
        use_ai_driver = self.ai_driven_radio.isChecked()
        subtasks = []

        if not use_ai_driver:
            if self.subtask_list_widget.count() == 0: QMessageBox.warning(self, "输入错误",
                                                                          "子任务模式下，子任务序列不能为空。"); return None
            for i in range(self.subtask_list_widget.count()):
                item = self.subtask_list_widget.item(i)
                subtask_data = item.data(Qt.UserRole)
                if isinstance(subtask_data, dict) and 'type' in subtask_data:
                    validation_error = self._validate_subtask_data(subtask_data)
                    if validation_error: QMessageBox.warning(self, "子任务错误",
                                                             f"第 {i + 1} 个子任务 ({subtask_data.get('type')}) 参数无效:\n{validation_error}\n请修正后再保存。"); self.subtask_list_widget.setCurrentRow(
                        i); return None
                    subtasks.append(subtask_data)
                else:
                    logger.error(f"列表项 {i} 缺少有效的子任务数据: {subtask_data}"); QMessageBox.critical(self,
                                                                                                           "内部错误",
                                                                                                           f"第 {i + 1} 个子任务的数据丢失或格式错误。"); return None

        task_id = self.task.task_id if self.task else None
        return {"task_id": task_id, "name": name, "type": task_type, "app_name": app_name, "priority": priority,
                "max_retries": max_retries, "device_name": device_name, "use_ai_driver": use_ai_driver,
                "subtasks": subtasks}

    def _validate_subtask_data(self, data: Dict[str, Any]) -> Optional[str]:
        """在保存前验证单个子任务字典的必要参数。
        【已修改】检查标签格式 (可选)。
        """
        st_type_str = data.get('type')
        if not st_type_str: return "缺少 'type' 字段。"
        try: st_type = SubtaskType(st_type_str)
        except ValueError: return f"未知的子任务类型: '{st_type_str}'。"

        label = data.get('label')
        if label is not None and not isinstance(label, str): return "'label' 必须是字符串。"
        if label and not re.match(r'^[a-zA-Z0-9_.-]+$', label): # 简单的标签格式校验
            return f"标签 '{label}' 包含无效字符。只允许字母、数字、下划线、点、短横线。"

        on_success_jump = data.get('on_success_jump_to_label')
        if on_success_jump is not None and not isinstance(on_success_jump, str): return "'on_success_jump_to_label' 必须是字符串。"
        if on_success_jump and not re.match(r'^[a-zA-Z0-9_.-]+$', on_success_jump):
            return f"成功跳转标签 '{on_success_jump}' 包含无效字符。"

        on_failure_jump = data.get('on_failure_jump_to_label')
        if on_failure_jump is not None and not isinstance(on_failure_jump, str): return "'on_failure_jump_to_label' 必须是字符串。"
        if on_failure_jump and not re.match(r'^[a-zA-Z0-9_.-]+$', on_failure_jump):
            return f"失败跳转标签 '{on_failure_jump}' 包含无效字符。"

        if st_type == SubtaskType.COMMENT:
            if data.get('text', '').strip() == '' and data.get('description', '').strip() == '':
                 return "注释类型的 'text' 或 'description' 至少需要一个非空。"
        # ... (existing validation for LOOP_START, WAIT, etc.) ...
        elif st_type == SubtaskType.LOOP_START: # (validation logic unchanged)
            count = data.get('count');
            if count is None: return "'count' 不能为空。"
            try:
                count_int = int(count); max_loops = CONFIG.get("MAX_LOOP_ITERATIONS", 1000);
                if not (0 < count_int <= max_loops): return f"'count' 必须是 1 到 {max_loops} 之间的整数。"
            except ValueError: return "'count' 必须是有效的整数。"
        elif st_type == SubtaskType.WAIT: # (validation logic unchanged)
            dur = data.get('duration');
            if dur is None: return "'duration' 不能为空。";
            try:
                float_dur = float(dur);
                if float_dur <= 0:
                    return "'duration' 必须是正数。"
            except ValueError: return "'duration' 必须是有效的数字。"
        elif st_type == SubtaskType.FIND_AND_CLICK_TEXT: # (validation logic unchanged)
            if not data.get('target_text'): return "'target_text' 不能为空。"
        elif st_type == SubtaskType.TEMPLATE_CLICK: # (validation logic unchanged)
            if not data.get('template_name'): return "'template_name' 不能为空。"
        elif st_type == SubtaskType.SWIPE: # (validation logic unchanged)
            has_direction = 'direction' in data and data.get('direction')
            has_coords = ('start' in data and isinstance(data.get('start'), list) and len(data['start']) == 2 and
                          'end' in data and isinstance(data.get('end'), list) and len(data['end']) == 2)
            if not has_direction and not has_coords: return "必须提供 'direction' 或有效的 'start' 和 'end' 坐标。"
            if has_direction and has_coords: return "不能同时指定 'direction' 和 'start'/'end' 坐标。"
            if has_coords:
                try: int(data['start'][0]); int(data['start'][1]); int(data['end'][0]); int(data['end'][1])
                except (ValueError, TypeError, IndexError):
                    return "'start' 和 'end' 坐标必须是有效的整数列表。"
        elif st_type == SubtaskType.AI_STEP: # (validation logic unchanged)
            if data.get('goal', '').strip() == '': return "'goal' 不能为空。"
        elif st_type == SubtaskType.ESP_COMMAND: # (validation logic unchanged)
            if data.get('command_string', '').strip() == '': return "'command_string' 不能为空。"
        elif st_type == SubtaskType.CHECK_TEXT_EXISTS:
            if not data.get('target_text'): return "'target_text' 不能为空。"
        elif st_type == SubtaskType.CHECK_TEMPLATE_EXISTS:
            if not data.get('template_name'): return "'template_name' 不能为空。"
        return None

    def accept(self):
        """重写 accept 逻辑，在保存前进行最终的循环结构校验。"""
        if self.subtask_driven_radio.isChecked():
            subtasks = [];
            for i in range(self.subtask_list_widget.count()):
                item = self.subtask_list_widget.item(i);
                data = item.data(Qt.UserRole)
                if isinstance(data, dict):
                    subtasks.append(data)
                else:
                    QMessageBox.critical(self, "内部错误", f"第 {i + 1} 个子任务数据无效。"); return

            loop_stack_check = []
            for i, task_data in enumerate(subtasks):
                task_type = task_data.get('type')
                if task_type == SubtaskType.LOOP_START.value:
                    loop_stack_check.append(i)
                elif task_type == SubtaskType.LOOP_END.value:
                    if not loop_stack_check: QMessageBox.warning(self, "循环结构错误",
                                                                 f"第 {i + 1} 个子任务是 LOOP_END，但没有找到匹配的 LOOP_START。"); self.subtask_list_widget.setCurrentRow(
                        i); return
                    loop_stack_check.pop()

            if loop_stack_check: start_index = loop_stack_check[0]; QMessageBox.warning(self, "循环结构错误",
                                                                                        f"第 {start_index + 1} 个子任务是 LOOP_START，但没有找到匹配的 LOOP_END。"); self.subtask_list_widget.setCurrentRow(
                start_index); return

        # --- 获取任务信息前进行一次最终数据校验 ---
        task_info = self.get_task_info()
        if task_info is None:
            return  # get_task_info 内部会弹出错误消息

        # 如果所有校验通过，则调用父类的 accept
        super().accept()

# --- UI Classes ---

class HumanInterventionDialog(QDialog):
    """用于人工干预 AI 决策的对话框。"""

    def __init__(self, action_str: str, justification: str, ai_prompt: str, ai_response: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("人工干预确认")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)

        # 显示信息
        info_group = QGroupBox("AI 决策详情")
        info_layout = QFormLayout()

        self.action_label = QLabel(f"`{action_str}`")
        self.action_label.setWordWrap(True)
        info_layout.addRow("建议操作:", self.action_label)

        self.justification_label = QLabel(justification if justification else "无")
        self.justification_label.setWordWrap(True)
        info_layout.addRow("操作理由:", self.justification_label)

        # 使用 QTextEdit 显示完整的上下文，更易读
        self.context_tabs = QTabWidget()

        self.prompt_text = QTextEdit()
        self.prompt_text.setPlainText(ai_prompt)
        self.prompt_text.setReadOnly(True)
        self.context_tabs.addTab(self.prompt_text, "AI 输入 (Prompt)")

        self.response_text = QTextEdit()
        self.response_text.setPlainText(ai_response)
        self.response_text.setReadOnly(True)
        self.context_tabs.addTab(self.response_text, "AI 输出 (Response)")

        info_layout.addRow("完整上下文:", self.context_tabs)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.approve_btn = QPushButton("✅ 同意执行")
        self.approve_btn.setStyleSheet("background-color: #c8e6c9;")  # 绿色背景
        self.approve_btn.clicked.connect(self.accept)  # 同意则关闭对话框并返回 Accepted

        self.reject_btn = QPushButton("❌ 拒绝执行")
        self.reject_btn.setStyleSheet("background-color: #ffcdd2;")  # 红色背景
        self.reject_btn.clicked.connect(self.reject)  # 拒绝则关闭对话框并返回 Rejected

        button_layout.addStretch()
        button_layout.addWidget(self.approve_btn)
        button_layout.addWidget(self.reject_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)


class MainUI(QMainWindow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.screenshot_manager = ScreenshotManager(config)
        self.esp_controller = ESPController(config)
        self.ai_analyzer = AIAnalyzer(config)
        self.task_executor = TaskExecutor(
            config, self.screenshot_manager, self.ai_analyzer,
            self.esp_controller, None, self
        )
        self.task_scheduler = TaskScheduler(config, self.task_executor)
        self.task_executor.task_scheduler = self.task_scheduler
        logger.info("核心组件初始化完成。")

        self.current_pixmap_item = None
        self.last_raw_screenshot_for_template: Optional[np.ndarray] = None
        self.defining_template_mode: bool = False
        self.template_rect_start_pos: Optional[QPointF] = None
        self.template_selection_rect_item: Optional[QGraphicsItem] = None

        # --- 新增：坐标调试相关状态 ---
        self.debugging_device_name: Optional[str] = None
        self.debugging_template_name: Optional[str] = None
        self.current_debug_offset_x: float = 0.0
        self.current_debug_offset_y: float = 0.0
        self._original_device_coord_map_backup: Optional[Dict[str, Any]] = None # 用于恢复设备原始配置
        # --- 新增结束 ---

        self.init_ui()

        try:
            self.task_scheduler.device_update_required.connect(self.update_device_table)
            self.task_scheduler.task_update_required.connect(self.update_task_table)
            logger.info("TaskScheduler 信号已连接到 UI 更新槽。")
        except Exception as connect_err:
            logger.error(f"连接 TaskScheduler 信号时出错: {connect_err}", exc_info=True)
            QMessageBox.critical(self, "信号连接错误", f"无法连接调度器信号:\n{connect_err}")

        self.human_intervention_mode_checkbox.setChecked(self.config.get("ENABLE_HUMAN_INTERVENTION", False))

        self.task_executor.screenshot_updated.connect(self.display_screenshot)
        self.task_executor.task_progress_updated.connect(self.update_task_progress_display)
        self.task_executor.request_human_intervention.connect(self._handle_intervention_request_ui)

        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(1000)

        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self.check_alerts)
        self.alert_timer.start(10000)

        self.setup_ui_logging()
        QTimer.singleShot(50, self._deferred_initialization)
        logger.info("系统 UI 初始化完成 (串行调度模式)。")

    def _deferred_initialization(self):
        """在事件循环开始后执行的初始化步骤。"""
        logger.info("执行延迟初始化 (加载设备和任务)...")
        # 加载设备和任务定义
        self._load_devices_from_config() # 从 config 加载设备到 scheduler
        self.task_scheduler._load_all_predefined_tasks() # 从 config 加载任务到 scheduler
        logger.info("延迟初始化完成。触发初始 UI 更新...")
        # 显式触发初始UI更新
        self.update_device_table()
        self.update_task_table()
        # 尝试自动连接 ESP
        if self.config.get("ESP_IP") and self.config.get("ESP_PORT"):
             logger.info("调度 ESP 自动连接...")
             QTimer.singleShot(100, self.on_esp_connect)


    def init_ui(self):
        self.setWindowTitle("智能手机自动化控制系统 v2.6 (坐标调试)") # 版本更新
        self.setGeometry(50, 50, 1400, 900) # 原始尺寸

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_esp_control_tab(), "ESP 控制")
        tab_widget.addTab(self.create_ai_control_tab(), "AI/调度 控制")
        tab_widget.addTab(self.create_task_scheduler_tab(), "设备与任务")
        tab_widget.addTab(self.create_coordinate_map_debug_tab(), "坐标调试") # 新增Tab
        tab_widget.addTab(self.create_settings_tab(), "系统设置")
        left_layout.addWidget(tab_widget)

        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group, 1)
        left_panel.setMinimumWidth(450)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        display_splitter = QSplitter(Qt.Vertical)
        screenshot_group = QGroupBox("实时截图 (滚轮缩放, 鼠标拖动, 右键定义模板)")
        screenshot_layout = QVBoxLayout()
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphics_view.setBackgroundBrush(QColor(Qt.darkGray))
        self.graphics_view.mousePressEvent = self.graphics_view_mouse_press_event
        self.graphics_view.mouseMoveEvent = self.graphics_view_mouse_move_event
        self.graphics_view.mouseReleaseEvent = self.graphics_view_mouse_release_event
        self.graphics_view.wheelEvent = self.graphics_view_wheel_event
        screenshot_layout.addWidget(self.graphics_view)
        btn_layout = QHBoxLayout()
        self.manual_screenshot_btn = QPushButton("手动截图 (主摄像头)")
        self.manual_screenshot_btn.clicked.connect(self.on_manual_screenshot)
        self.reset_zoom_btn = QPushButton("重置缩放/视图")
        self.reset_zoom_btn.clicked.connect(self.reset_screenshot_zoom)
        self.define_template_btn = QPushButton("框选定义模板")
        self.define_template_btn.setCheckable(True)
        self.define_template_btn.toggled.connect(self.on_toggle_define_template_mode)
        self.save_current_screenshot_btn = QPushButton("保存当前截图")
        self.save_current_screenshot_btn.clicked.connect(self.on_save_current_screenshot)
        self.save_current_screenshot_btn.setEnabled(False)
        btn_layout.addWidget(self.manual_screenshot_btn)
        btn_layout.addWidget(self.reset_zoom_btn)
        btn_layout.addWidget(self.define_template_btn)
        btn_layout.addWidget(self.save_current_screenshot_btn)
        btn_layout.addStretch()
        screenshot_layout.addLayout(btn_layout)
        screenshot_group.setLayout(screenshot_layout)
        display_splitter.addWidget(screenshot_group)
        ocr_group = QGroupBox("OCR 识别结果")
        ocr_layout = QVBoxLayout()
        self.ocr_text = QTextEdit()
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setFont(QFont("SimSun", 10))
        ocr_layout.addWidget(self.ocr_text)
        ocr_group.setLayout(ocr_layout)
        display_splitter.addWidget(ocr_group)
        display_splitter.setStretchFactor(0, 3)
        display_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(display_splitter, 2)
        progress_splitter = QSplitter(Qt.Vertical)
        ai_decision_group = QGroupBox("AI决策记录")
        ai_decision_layout = QVBoxLayout()
        self.ai_decision_text = QTextEdit()
        self.ai_decision_text.setReadOnly(True)
        self.ai_decision_text.setFont(QFont("Courier New", 9))
        ai_decision_layout.addWidget(self.ai_decision_text)
        ai_decision_group.setLayout(ai_decision_layout)
        progress_splitter.addWidget(ai_decision_group)
        task_progress_group = QGroupBox("任务详细日志")
        task_progress_layout = QVBoxLayout()
        self.task_progress_text = QTextEdit()
        self.task_progress_text.setReadOnly(True)
        self.task_progress_text.setFont(QFont("Courier New", 9))
        task_progress_layout.addWidget(self.task_progress_text)
        task_progress_group.setLayout(task_progress_layout)
        progress_splitter.addWidget(task_progress_group)
        progress_splitter.setSizes([100, 200])
        right_layout.addWidget(progress_splitter, 1)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        self.setMinimumSize(1200, 800)
        logger.info("UI 布局初始化完成。")

    def create_coordinate_map_debug_tab(self):
        """创建坐标映射调试功能的Tab页面。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        tab.setLayout(layout)

        # 1. 设置区域
        setup_group = QGroupBox("调试设置")
        setup_layout = QFormLayout()

        self.debug_device_combo = QComboBox()
        self.debug_device_combo.addItem("选择设备...", None)
        # 设备列表会在 _load_devices_from_config 后或 update_device_table 时更新
        setup_layout.addRow("选择调试设备:", self.debug_device_combo)

        self.debug_template_combo = QComboBox()
        self.debug_template_combo.addItem("选择模板...", None)
        # 模板列表会在 _populate_coord_debug_tab_templates 中填充
        refresh_templates_btn = QPushButton("刷新模板列表")
        refresh_templates_btn.clicked.connect(self._populate_coord_debug_tab_templates)
        template_layout = QHBoxLayout()
        template_layout.addWidget(self.debug_template_combo, 1)
        template_layout.addWidget(refresh_templates_btn)
        setup_layout.addRow("目标点击模板:", template_layout)

        offset_group = QGroupBox("当前调试偏移 (像素)")
        offset_layout = QFormLayout(offset_group)
        self.debug_offset_x_edit = QLineEdit("0.0")
        self.debug_offset_y_edit = QLineEdit("0.0")
        self.debug_offset_x_edit.setPlaceholderText("例如: -5.0")
        self.debug_offset_y_edit.setPlaceholderText("例如: 2.5")
        offset_layout.addRow("X 偏移 (offset_x):", self.debug_offset_x_edit)
        offset_layout.addRow("Y 偏移 (offset_y):", self.debug_offset_y_edit)
        setup_layout.addRow(offset_group)

        start_session_btn = QPushButton("加载设备配置 / 开始新会话")
        start_session_btn.clicked.connect(self.on_coord_debug_start_session)
        setup_layout.addRow(start_session_btn)

        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group)

        # 2. 操作区域
        action_group = QGroupBox("调试操作")
        action_layout = QVBoxLayout(action_group)

        self.debug_click_target_btn = QPushButton("🕹️ 点击目标模板 (应用当前偏移)")
        self.debug_click_target_btn.clicked.connect(self.on_coord_debug_click_target)
        self.debug_click_target_btn.setEnabled(False)  # 初始禁用
        action_layout.addWidget(self.debug_click_target_btn)

        action_layout.addWidget(QLabel("--- 手动像素调整 ---"))
        manual_adj_layout = QFormLayout()
        self.debug_dx_pixel_edit = QLineEdit("0.0")
        self.debug_dy_pixel_edit = QLineEdit("0.0")
        self.debug_dx_pixel_edit.setPlaceholderText("X方向修正值 (像素)")
        self.debug_dy_pixel_edit.setPlaceholderText("Y方向修正值 (像素)")
        manual_adj_layout.addRow("像素 dx:", self.debug_dx_pixel_edit)
        manual_adj_layout.addRow("像素 dy:", self.debug_dy_pixel_edit)
        apply_pixel_adj_btn = QPushButton("应用像素调整并重试点击")
        apply_pixel_adj_btn.clicked.connect(self.on_coord_debug_apply_pixel_adj)
        manual_adj_layout.addRow(apply_pixel_adj_btn)
        action_layout.addLayout(manual_adj_layout)

        action_layout.addWidget(QLabel("--- 基于物理偏差的半自动调整 ---"))
        phys_adj_layout = QFormLayout()
        self.debug_dx_phys_mm_edit = QLineEdit("0.0")
        self.debug_dy_phys_mm_edit = QLineEdit("0.0")
        self.debug_dx_phys_mm_edit.setPlaceholderText("X方向物理偏差(mm), 正表示点偏右")
        self.debug_dy_phys_mm_edit.setPlaceholderText("Y方向物理偏差(mm), 正表示点偏下")  # 注意这里的物理坐标系与屏幕可能相反
        phys_adj_layout.addRow("物理偏差 dx (mm):", self.debug_dx_phys_mm_edit)
        phys_adj_layout.addRow("物理偏差 dy (mm):", self.debug_dy_phys_mm_edit)
        apply_phys_adj_btn = QPushButton("计算像素偏移并应用调整 (然后重试点击)")
        apply_phys_adj_btn.clicked.connect(self.on_coord_debug_apply_phys_adj_and_retry)
        phys_adj_layout.addRow(apply_phys_adj_btn)
        action_layout.addLayout(phys_adj_layout)

        layout.addWidget(action_group)

        # 3. 保存区域
        save_group = QGroupBox("保存与应用")
        save_layout = QHBoxLayout(save_group)
        save_current_device_btn = QPushButton("💾 保存当前偏移到此设备")
        save_current_device_btn.clicked.connect(self.on_coord_debug_save_to_device)
        apply_to_all_btn = QPushButton("⚠️ 将此设备偏移应用到所有设备")
        apply_to_all_btn.clicked.connect(self.on_coord_debug_apply_to_all)
        save_layout.addWidget(save_current_device_btn)
        save_layout.addWidget(apply_to_all_btn)
        layout.addWidget(save_group)

        # 4. 日志区域
        log_group = QGroupBox("调试日志")
        log_layout_v = QVBoxLayout(log_group)  # Renamed to avoid conflict
        self.coord_debug_log_text = QTextEdit()
        self.coord_debug_log_text.setReadOnly(True)
        self.coord_debug_log_text.setFont(QFont("Courier New", 9))
        log_layout_v.addWidget(self.coord_debug_log_text)
        layout.addWidget(log_group, 1)  # 给日志更多空间

        layout.addStretch()
        return tab

    def _log_coord_debug(self, message: str):
        """辅助函数，用于在坐标调试日志区域记录信息。"""
        if hasattr(self, 'coord_debug_log_text'):
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.coord_debug_log_text.append(f"[{timestamp}] {message}")
            self.coord_debug_log_text.verticalScrollBar().setValue(
                self.coord_debug_log_text.verticalScrollBar().maximum()
            )

    def on_coord_debug_apply_to_all(self):
        """将当前调试的偏移量应用到所有其他设备 (内存和全局 CONFIG)。"""
        if not self.debugging_device_name:
            QMessageBox.warning(self, "无会话", "请先调试一个设备并获取基准偏移量。")
            return

        num_devices = len(self.task_scheduler.devices)
        if num_devices <= 1:
            QMessageBox.information(self, "无需操作", "只有一个设备，无需应用到其他设备。")
            return

        reply = QMessageBox.warning(self, "高风险操作确认",
                                    f"确定要将设备 '{self.debugging_device_name}' 当前的调试偏移 "
                                    f"X={self.current_debug_offset_x:.2f}, Y={self.current_debug_offset_y:.2f} "
                                    f"应用到其他所有 ({num_devices - 1} 个) 设备吗？\n"
                                    "这可能会覆盖它们各自的精确校准！此操作不可逆 (除非手动改回)。",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        if "DEVICE_CONFIGS" not in self.config: self.config["DEVICE_CONFIGS"] = {}
        applied_count = 0
        for dev_name, device_obj in self.task_scheduler.devices.items():
            if dev_name == self.debugging_device_name: continue  # 跳过当前调试的设备

            dev_conf = self.config["DEVICE_CONFIGS"].get(dev_name, {})
            coord_map = dev_conf.get("COORDINATE_MAP", {}).copy()
            if not isinstance(coord_map, dict): coord_map = {}

            coord_map["offset_x"] = round(self.current_debug_offset_x, 3)
            coord_map["offset_y"] = round(self.current_debug_offset_y, 3)
            if "scale_x" not in coord_map: coord_map["scale_x"] = 1.0
            if "scale_y" not in coord_map: coord_map["scale_y"] = 1.0

            dev_conf["COORDINATE_MAP"] = coord_map
            self.config["DEVICE_CONFIGS"][dev_name] = dev_conf
            device_obj._config["COORDINATE_MAP"] = coord_map  # 更新设备对象内部配置
            applied_count += 1
            self._log_coord_debug(f"已将偏移应用到设备 '{dev_name}': {coord_map}")

        self._log_coord_debug(f"已将当前调试偏移应用到 {applied_count} 个其他设备。")
        QMessageBox.information(self, "应用完成",
                                f"调试偏移已应用到 {applied_count} 个其他设备的内存配置中。\n"
                                "请记得在“系统设置”Tab中保存以持久化。")

    def on_coord_debug_save_to_device(self):
        """将当前调试的偏移量保存到选定设备的配置中 (内存和全局 CONFIG)。"""
        if not self.debugging_device_name:
            QMessageBox.warning(self, "无会话", "没有正在调试的设备。")
            return

        device = self.task_scheduler.devices.get(self.debugging_device_name)
        if not device:
            QMessageBox.critical(self, "错误", f"找不到设备对象: {self.debugging_device_name}")
            return

        reply = QMessageBox.question(self, "确认保存",
                                     f"确定要将偏移 X={self.current_debug_offset_x:.2f}, Y={self.current_debug_offset_y:.2f} "
                                     f"保存到设备 '{self.debugging_device_name}' 的配置吗？\n"
                                     f"(这将更新内存中的配置，需手动点击“保存所有设置”以持久化到文件)",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return

        # 获取设备配置，如果不存在则创建
        if "DEVICE_CONFIGS" not in self.config: self.config["DEVICE_CONFIGS"] = {}
        dev_conf = self.config["DEVICE_CONFIGS"].get(self.debugging_device_name, {})

        # 获取或创建 COORDINATE_MAP
        coord_map = dev_conf.get("COORDINATE_MAP", {}).copy()
        if not isinstance(coord_map, dict): coord_map = {}

        coord_map["offset_x"] = round(self.current_debug_offset_x, 3)  # 保存时保留几位小数
        coord_map["offset_y"] = round(self.current_debug_offset_y, 3)
        if "scale_x" not in coord_map: coord_map["scale_x"] = 1.0  # 确保scale存在
        if "scale_y" not in coord_map: coord_map["scale_y"] = 1.0

        dev_conf["COORDINATE_MAP"] = coord_map
        self.config["DEVICE_CONFIGS"][self.debugging_device_name] = dev_conf

        # 同时更新设备对象内部的配置
        device._config["COORDINATE_MAP"] = coord_map

        self._log_coord_debug(f"已将调试偏移保存到设备 '{self.debugging_device_name}' 的内存配置: {coord_map}")
        QMessageBox.information(self, "保存成功",
                                f"偏移已更新到设备 '{self.debugging_device_name}' 的内存配置中。\n"
                                "请记得在“系统设置”Tab中点击“保存所有设置到 config.json”以永久保存。")
        # 清理备份，因为已经“正式”应用了
        self._original_device_coord_map_backup = json.loads(json.dumps(coord_map))

    def on_coord_debug_apply_phys_adj_and_retry(self):
        """根据观察到的物理偏差 (mm) 计算像素偏移，应用并重试。"""
        if not self.debugging_device_name:
            QMessageBox.warning(self, "无会话", "请先开始一个调试会话。")
            return
        try:
            phys_error_dx_mm = float(self.debug_dx_phys_mm_edit.text())  # 用户观察到的X方向物理误差
            phys_error_dy_mm = float(self.debug_dy_phys_mm_edit.text())  # 用户观察到的Y方向物理误差
        except ValueError:
            QMessageBox.warning(self, "输入错误", "物理偏差必须是有效的数字 (mm)。")
            return

        # 屏幕像素与物理尺寸的转换关系 (从 _pixel_to_commanded_machine_coords 简化)
        # 这些是硬编码的校准结果，表示每像素对应的物理移动量（mm/pixel）
        # 注意：这些值是负数，因为屏幕像素增加通常对应物理坐标的减少
        # crop_w, crop_h = self.config.get("CROPPED_RESOLUTION") # 应获取当前调试设备的分辨率
        device = self.task_scheduler.devices.get(self.debugging_device_name)
        if not device: self._log_coord_debug("无法获取设备以计算物理偏差，操作中止。"); return

        # 从设备配置或全局配置获取裁剪分辨率
        cropped_res = device.get_config("CROPPED_RESOLUTION", CONFIG["CROPPED_RESOLUTION"])
        screen_width_pixels = cropped_res[0]
        screen_height_pixels = cropped_res[1]

        # 假设的物理行程，这些值来自于 _pixel_to_commanded_machine_coords 中的硬编码
        # (M(115, 274) for Px(0,0) and M(-15, 84) for Px(1080,1440))
        # X方向物理行程: 115 - (-15) = 130 mm
        # Y方向物理行程: 274 - 84   = 190 mm
        physical_travel_x_mm = 130.0
        physical_travel_y_mm = 190.0

        # 每像素对应的物理毫米数 (注意符号，保持与原始转换一致)
        mm_per_pixel_x = -physical_travel_x_mm / screen_width_pixels
        mm_per_pixel_y = -physical_travel_y_mm / screen_height_pixels

        if abs(mm_per_pixel_x) < 1e-6 or abs(mm_per_pixel_y) < 1e-6:
            self._log_coord_debug("错误: 计算出的 mm/pixel 比例过小或为零，无法进行调整。")
            QMessageBox.critical(self, "计算错误", "无法计算有效的毫米/像素比例。")
            return

        # 计算需要的像素调整量
        # 如果实际点击偏右 (phys_error_dx_mm > 0)，说明发送的物理X值太大了，
        # 而物理X值与 (pixel_x + offset_x) 成反比（因为 mm_per_pixel_x 是负数）。
        # 所以，要减小物理X值，需要增加 (pixel_x + offset_x)，即增加 offset_x。
        # delta_offset_x_pixels = phys_error_dx_mm / mm_per_pixel_x
        #
        # 修正逻辑：
        # 物理偏差 dx_error_mm > 0 表示实际点击在目标右侧（物理 X 值更大）。
        # 我们希望下一次点击的物理 X 值减小 dx_error_mm。
        # commanded_mx = (mm_per_pixel_x * (px + offset_x)) + C_x
        # 我们希望 commanded_mx_new = commanded_mx_old - dx_error_mm
        # (mm_per_pixel_x * (px + offset_x_new)) + C_x = (mm_per_pixel_x * (px + offset_x_old)) + C_x - dx_error_mm
        # mm_per_pixel_x * offset_x_new = mm_per_pixel_x * offset_x_old - dx_error_mm
        # offset_x_new = offset_x_old - (dx_error_mm / mm_per_pixel_x)
        # 所以，像素调整量 pixel_adjust_dx = - (phys_error_dx_mm / mm_per_pixel_x)
        pixel_adjust_dx = - (phys_error_dx_mm / mm_per_pixel_x)
        pixel_adjust_dy = - (phys_error_dy_mm / mm_per_pixel_y)  # 同理 Y

        self._log_coord_debug(f"物理偏差(mm): dx={phys_error_dx_mm}, dy={phys_error_dy_mm}")
        self._log_coord_debug(f"转换因子(mm/px): mm_per_px_X={mm_per_pixel_x:.4f}, mm_per_px_Y={mm_per_pixel_y:.4f}")
        self._log_coord_debug(f"计算得到的像素调整量: dx_px={pixel_adjust_dx:.2f}, dy_px={pixel_adjust_dy:.2f}")

        self.current_debug_offset_x += pixel_adjust_dx
        self.current_debug_offset_y += pixel_adjust_dy

        self.debug_offset_x_edit.setText(f"{self.current_debug_offset_x:.2f}")
        self.debug_offset_y_edit.setText(f"{self.current_debug_offset_y:.2f}")
        self._log_coord_debug(
            f"应用物理偏差调整。新调试偏移: X={self.current_debug_offset_x:.2f}, Y={self.current_debug_offset_y:.2f}")

        target_device = self.task_scheduler.devices.get(self.debugging_device_name)
        if target_device:
            self._apply_debug_offsets_to_device_config(target_device)
            if self.debug_template_name:
                self.on_coord_debug_click_target()
            else:
                self._log_coord_debug("调整已应用，但未选择目标模板，请手动点击。")
        else:
            self._log_coord_debug("错误: 找不到当前调试设备对象。")

    def _populate_coord_debug_tab_templates(self):
        """填充坐标调试Tab中的模板下拉列表。"""
        if not hasattr(self, 'debug_template_combo'): return

        current_selection = self.debug_template_combo.currentText()
        if current_selection == "选择模板...": current_selection = None  # Handle placeholder

        self.debug_template_combo.clear()
        self.debug_template_combo.addItem("选择模板...", None)

        self.ai_analyzer.load_templates()  # 确保加载最新的模板
        templates = sorted(self.ai_analyzer.templates.keys())
        for name in templates:
            self.debug_template_combo.addItem(name, name)

        if current_selection and current_selection in templates:
            self.debug_template_combo.setCurrentText(current_selection)
        elif templates:  # 如果之前没选或选的没了，默认选第一个模板
            self.debug_template_combo.setCurrentIndex(1)

    def on_coord_debug_click_target(self):
        """执行点击目标模板的操作。"""
        if not self.debugging_device_name or not self.debug_template_name:
            QMessageBox.warning(self, "设置不完整", "请先选择调试设备和目标模板，并开始调试会话。")
            return

        device = self.task_scheduler.devices.get(self.debugging_device_name)
        if not device:
            QMessageBox.critical(self, "错误", f"找不到设备对象: {self.debugging_device_name}")
            return

        # 确保最新的调试偏移已应用
        self._apply_debug_offsets_to_device_config(device)

        self._log_coord_debug(f"尝试点击模板 '{self.debug_template_name}' 在设备 '{device.name}' 上...")
        self.debug_click_target_btn.setEnabled(False)  # 防止重复点击
        QApplication.processEvents()

        # 1. 获取截图并找到模板中心像素
        screenshot = self.screenshot_manager.take_screenshot(device)
        if screenshot is None:
            self._log_coord_debug("错误: 获取截图失败。")
            QMessageBox.critical(self, "错误", "获取设备截图失败，无法继续。")
            self.debug_click_target_btn.setEnabled(True)
            return
        self.display_screenshot(screenshot.copy(), f"调试截图@{device.name}")  # 显示原始图

        match_result = self.ai_analyzer.template_matching(screenshot, template_name=self.debug_template_name)
        if not match_result.get("match"):
            self._log_coord_debug(
                f"错误: 在当前屏幕上未找到模板 '{self.debug_template_name}'。置信度: {match_result.get('confidence', 'N/A'):.3f}")
            QMessageBox.warning(self, "未找到模板",
                                f"在设备 '{device.name}' 的当前屏幕上未找到模板 '{self.debug_template_name}'。")
            self.debug_click_target_btn.setEnabled(True)
            return

        center_pixel_x, center_pixel_y = match_result["center"]
        self._log_coord_debug(f"模板 '{self.debug_template_name}' 中心像素: ({center_pixel_x}, {center_pixel_y})")

        # 在截图上标记目标点
        annotated_img = screenshot.copy()
        cv2.circle(annotated_img, (center_pixel_x, center_pixel_y), 15, (0, 255, 0), 2)  # 绿色目标
        cv2.drawMarker(annotated_img, (center_pixel_x, center_pixel_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        self.display_screenshot(annotated_img, f"调试目标@{device.name}")
        QApplication.processEvents()

        # 2. 执行点击 (使用 TaskExecutor 的方法，它会应用当前的 COORDINATE_MAP)
        # 注意：_execute_click 是一个原子操作，包含了移动、点击、返回原点和延时
        # 它会使用 device._config["COORDINATE_MAP"]，我们已在调试会话中临时修改了它
        self._log_coord_debug(f"调用 TaskExecutor._execute_click 点击像素 ({center_pixel_x}, {center_pixel_y}) ...")
        click_result = self.task_executor._execute_click(device, center_pixel_x, center_pixel_y)

        if click_result.get("success"):
            self._log_coord_debug("点击命令成功发送并完成（包括返回原点和延时）。请观察实际点击位置。")
            # 再次截图显示点击后的效果
            time.sleep(0.5)  # 给界面一点反应时间
            final_screenshot = self.screenshot_manager.take_screenshot(device)
            if final_screenshot is not None:
                self.display_screenshot(final_screenshot.copy(), f"点击后@{device.name}")
        else:
            err_msg = click_result.get("error", "未知点击错误")
            self._log_coord_debug(f"点击命令失败: {err_msg}")
            QMessageBox.critical(self, "点击失败", f"执行点击操作失败: {err_msg}")

        self.debug_click_target_btn.setEnabled(True)

    def on_coord_debug_apply_pixel_adj(self):
        """应用手动输入的像素偏移调整量，并重试点击。"""
        if not self.debugging_device_name:
            QMessageBox.warning(self, "无会话", "请先开始一个调试会话。")
            return
        try:
            dx = float(self.debug_dx_pixel_edit.text())
            dy = float(self.debug_dy_pixel_edit.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "像素调整量必须是有效的数字。")
            return

        self.current_debug_offset_x += dx
        self.current_debug_offset_y += dy

        self.debug_offset_x_edit.setText(f"{self.current_debug_offset_x:.2f}")
        self.debug_offset_y_edit.setText(f"{self.current_debug_offset_y:.2f}")
        self._log_coord_debug(
            f"应用像素调整: dx={dx}, dy={dy}。新调试偏移: X={self.current_debug_offset_x:.2f}, Y={self.current_debug_offset_y:.2f}")

        # 更新设备内存配置并重试点击
        device = self.task_scheduler.devices.get(self.debugging_device_name)
        if device:
            self._apply_debug_offsets_to_device_config(device)
            if self.debug_template_name:  # 只有选择了模板才自动重试
                self.on_coord_debug_click_target()
            else:
                self._log_coord_debug("调整已应用，但未选择目标模板，请手动点击。")
        else:
            self._log_coord_debug("错误: 找不到当前调试设备对象。")

    def on_coord_debug_start_session(self):
        """开始或更新一个坐标调试会话。"""
        device_name = self.debug_device_combo.currentData()
        if not device_name:
            QMessageBox.warning(self, "选择设备", "请先选择一个要调试的设备。")
            self.debug_click_target_btn.setEnabled(False)
            return

        device = self.task_scheduler.devices.get(device_name)
        if not device:
            QMessageBox.critical(self, "错误", f"找不到设备对象: {device_name}")
            self.debug_click_target_btn.setEnabled(False)
            return

        # 如果之前正在调试其他设备，恢复其原始配置
        if self.debugging_device_name and self.debugging_device_name != device_name and self._original_device_coord_map_backup:
            old_device_to_restore = self.task_scheduler.devices.get(self.debugging_device_name)
            if old_device_to_restore:
                self._log_coord_debug(
                    f"恢复设备 '{self.debugging_device_name}' 的原始坐标映射: {self._original_device_coord_map_backup}")
                old_device_to_restore._config["COORDINATE_MAP"] = self._original_device_coord_map_backup
            self._original_device_coord_map_backup = None

        self.debugging_device_name = device_name
        self._log_coord_debug(f"开始/更新设备 '{device_name}' 的调试会话。")

        # 备份当前设备即将被修改的坐标映射
        # 使用深拷贝，防止后续修改影响备份
        current_map_from_config = device.get_config("COORDINATE_MAP", {})
        self._original_device_coord_map_backup = json.loads(json.dumps(current_map_from_config))

        try:
            # 从UI输入框获取初始/当前调试偏移量
            self.current_debug_offset_x = float(self.debug_offset_x_edit.text())
            self.current_debug_offset_y = float(self.debug_offset_y_edit.text())
        except ValueError:
            # 如果UI输入无效，则从设备配置加载
            coord_map = device.get_config("COORDINATE_MAP", {})  # 获取设备当前配置
            self.current_debug_offset_x = float(coord_map.get("offset_x", 0.0))
            self.current_debug_offset_y = float(coord_map.get("offset_y", 0.0))
            self.debug_offset_x_edit.setText(str(self.current_debug_offset_x))
            self.debug_offset_y_edit.setText(str(self.current_debug_offset_y))
            self._log_coord_debug(
                f"UI偏移值无效，从设备配置加载偏移: X={self.current_debug_offset_x}, Y={self.current_debug_offset_y}")

        self._log_coord_debug(
            f"当前会话调试偏移设定为: X={self.current_debug_offset_x}, Y={self.current_debug_offset_y}")

        # 临时应用调试偏移到设备配置（内存中）
        self._apply_debug_offsets_to_device_config(device)

        self.debug_template_name = self.debug_template_combo.currentData()
        if self.debug_template_name:
            self.debug_click_target_btn.setEnabled(True)
            self._log_coord_debug(f"目标模板: '{self.debug_template_name}'。可以开始点击。")
        else:
            self.debug_click_target_btn.setEnabled(False)
            self._log_coord_debug("未选择目标模板，请选择模板后才能点击。")

    def _apply_debug_offsets_to_device_config(self, device: Optional[Device] = None):
        """将 self.current_debug_offset_x/y 应用到指定设备 (或当前调试设备) 的内存配置中。"""
        target_device = device
        if not target_device and self.debugging_device_name:
            target_device = self.task_scheduler.devices.get(self.debugging_device_name)

        if not target_device:
            self._log_coord_debug("错误: _apply_debug_offsets_to_device_config 未找到目标设备。")
            return

        # 获取设备当前的COORDINATE_MAP，如果不存在则创建一个新的
        device_coord_map = target_device.get_config("COORDINATE_MAP", {}).copy()  # 使用.copy()避免修改原始引用
        if not isinstance(device_coord_map, dict):  # 防御性编程，确保是字典
            device_coord_map = {}

        device_coord_map["offset_x"] = self.current_debug_offset_x
        device_coord_map["offset_y"] = self.current_debug_offset_y
        # 保留可能存在的 scale 值
        if "scale_x" not in device_coord_map: device_coord_map["scale_x"] = 1.0
        if "scale_y" not in device_coord_map: device_coord_map["scale_y"] = 1.0

        # 更新设备对象内部的配置
        target_device._config["COORDINATE_MAP"] = device_coord_map
        self._log_coord_debug(f"已临时应用调试偏移到设备 '{target_device.name}' 内存: {device_coord_map}")

    def _populate_coord_debug_tab_devices(self):
        """填充坐标调试Tab中的设备下拉列表。"""
        if not hasattr(self, 'debug_device_combo'): return
        current_selection = self.debug_device_combo.currentData()
        self.debug_device_combo.clear()
        self.debug_device_combo.addItem("选择设备...", None)
        devices = sorted(self.task_scheduler.devices.keys())
        for name in devices:
            self.debug_device_combo.addItem(name, name)
        if current_selection and current_selection in devices:
            self.debug_device_combo.setCurrentText(current_selection)
        elif devices:  # 默认选中第一个（如果存在）
            self.debug_device_combo.setCurrentIndex(1)

    def _load_devices_from_config(self):
        # ... (代码不变) ...
        devices_config = self.config.get("DEVICE_CONFIGS", {})
        loaded_count = 0
        if not isinstance(devices_config, dict):
             logger.error("配置文件中的 DEVICE_CONFIGS 不是有效的字典格式！")
             return

        for name, dev_config in devices_config.items():
            if not isinstance(dev_config, dict):
                logger.warning(f"跳过无效的设备配置条目 (非字典): {name}")
                continue
            try:
                # 确保使用大写键进行查找和传递
                apps = dev_config.get("APPS", dev_config.get("apps", [])) # 兼容小写
                position = dev_config.get("POSITION", dev_config.get("position", ""))

                # 创建Device实例时，其内部_config会通过get_config方法规范化键
                # 这里我们确保传递给Device构造函数的apps和position是存在的
                device = Device(name=name, apps=apps, position=position)

                # 直接将从config加载的dev_config (可能包含小写键) 赋予Device._config
                # Device的get_config方法会处理大小写转换
                device._config = dev_config.copy() # 使用副本以防意外修改原始配置

                self.task_scheduler.add_device(device)
                loaded_count += 1
            except Exception as e:
                logger.error(f"加载设备配置 '{name}' 时出错: {e}", exc_info=True)
        if loaded_count > 0:
            logger.info(f"从配置加载了 {loaded_count} 个设备。")
        else:
            logger.info("未从配置加载任何设备。")
        self._populate_coord_debug_tab_devices() # 加载后更新调试Tab设备列表
        
    # --- 其他 UI 创建函数 (`create_esp_control_tab`, `create_ai_control_tab`, `create_task_scheduler_tab`, `create_settings_tab`) 保持不变 ---
    # ... (代码省略) ...
    def create_esp_control_tab(self): # (代码不变)
        tab = QWidget()
        layout = QVBoxLayout()
        conn_group = QGroupBox("ESP-12F 连接控制")
        conn_layout = QFormLayout()
        self.esp_ip_edit = QLineEdit(self.config["ESP_IP"])
        conn_layout.addRow("ESP IP地址:", self.esp_ip_edit)
        self.esp_port_edit = QLineEdit(str(self.config["ESP_PORT"]))
        conn_layout.addRow("ESP端口:", self.esp_port_edit)
        button_layout = QHBoxLayout()
        self.esp_connect_btn = QPushButton("连接")
        self.esp_connect_btn.clicked.connect(self.on_esp_connect)
        self.esp_disconnect_btn = QPushButton("断开")
        self.esp_disconnect_btn.clicked.connect(self.on_esp_disconnect)
        self.esp_disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.esp_connect_btn)
        button_layout.addWidget(self.esp_disconnect_btn)
        conn_layout.addRow(button_layout)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        cmd_group = QGroupBox("手动命令")
        cmd_layout = QGridLayout()
        self.home_btn = QPushButton("回原点 (HOME)")
        self.home_btn.clicked.connect(self.on_esp_home)
        cmd_layout.addWidget(self.home_btn, 0, 0)
        self.set_origin_btn = QPushButton("设置原点 (O)")
        self.set_origin_btn.clicked.connect(self.on_esp_set_origin)
        cmd_layout.addWidget(self.set_origin_btn, 0, 1)
        self.status_btn = QPushButton("获取状态 (STATUS)")
        self.status_btn.clicked.connect(self.on_esp_status)
        cmd_layout.addWidget(self.status_btn, 0, 2)

        move_form = QFormLayout()
        self.x_distance_edit = QLineEdit("10")
        x_move_btn = QPushButton("移动 X轴 (mm)")
        x_move_btn.clicked.connect(self.on_move_x)
        self.x_move_btn = x_move_btn
        move_form.addRow(x_move_btn, self.x_distance_edit)
        self.y_distance_edit = QLineEdit("10")
        y_move_btn = QPushButton("移动 Y轴 (mm)")
        y_move_btn.clicked.connect(self.on_move_y)
        self.y_move_btn = y_move_btn
        move_form.addRow(y_move_btn, self.y_distance_edit)
        cmd_layout.addLayout(move_form, 1, 0, 1, 3)

        # --- 修改：区分像素点击和物理点击 ---
        # 像素坐标点击 (假设 ESP 支持小写 x, y)
        pixel_click_layout = QHBoxLayout()
        self.pixel_click_x_edit = QLineEdit("540")
        self.pixel_click_y_edit = QLineEdit("720")
        pixel_click_btn = QPushButton("像素坐标点击 (x,y)")
        pixel_click_btn.clicked.connect(self.on_esp_pixel_click) # 连接到新的处理函数
        self.pixel_click_btn = pixel_click_btn
        pixel_click_layout.addWidget(pixel_click_btn)
        pixel_click_layout.addWidget(QLabel("x:"))
        pixel_click_layout.addWidget(self.pixel_click_x_edit)
        pixel_click_layout.addWidget(QLabel("y:"))
        pixel_click_layout.addWidget(self.pixel_click_y_edit)
        cmd_layout.addLayout(pixel_click_layout, 2, 0, 1, 3)

        # 物理坐标点击 (mm)
        phys_click_layout = QHBoxLayout()
        self.phys_click_x_edit = QLineEdit("100")
        self.phys_click_y_edit = QLineEdit("100")
        phys_click_btn = QPushButton("物理坐标点击 (X,Y mm)")
        phys_click_btn.clicked.connect(self.on_esp_phys_click) # 连接到新的处理函数
        self.phys_click_btn = phys_click_btn
        phys_click_layout.addWidget(phys_click_btn)
        phys_click_layout.addWidget(QLabel("X:"))
        phys_click_layout.addWidget(self.phys_click_x_edit)
        phys_click_layout.addWidget(QLabel("Y:"))
        phys_click_layout.addWidget(self.phys_click_y_edit)
        cmd_layout.addLayout(phys_click_layout, 3, 0, 1, 3)
        # --- 修改结束 ---

        custom_layout = QHBoxLayout()
        self.custom_cmd_edit = QLineEdit()
        self.custom_cmd_edit.setPlaceholderText("例如: G X100 Y200 M1") # 示例改为物理坐标
        custom_btn = QPushButton("发送自定义命令")
        custom_btn.clicked.connect(self.on_send_custom_cmd)
        self.custom_cmd_btn = custom_btn
        custom_layout.addWidget(custom_btn)
        custom_layout.addWidget(self.custom_cmd_edit)
        cmd_layout.addLayout(custom_layout, 4, 0, 1, 3) # 行号调整为4

        cmd_group.setLayout(cmd_layout)
        layout.addWidget(cmd_group)

        resp_group = QGroupBox("ESP 响应")
        resp_layout = QVBoxLayout()
        self.esp_response_text = QTextEdit()
        self.esp_response_text.setReadOnly(True)
        self.esp_response_text.setMaximumHeight(100)
        resp_layout.addWidget(self.esp_response_text)
        resp_group.setLayout(resp_layout)
        layout.addWidget(resp_group)

        layout.addStretch(1)
        tab.setLayout(layout)
        self.update_esp_button_states(False) # 初始禁用按钮
        return tab

    def update_esp_button_states(self, connected: bool):
        """根据 ESP 连接状态启用/禁用 ESP 命令按钮。"""
        buttons_to_toggle = [
            self.home_btn, self.set_origin_btn, self.status_btn,
            self.x_move_btn, self.y_move_btn,
            self.pixel_click_btn, # 使用像素点击按钮变量名
            self.phys_click_btn,  # 使用物理点击按钮变量名
            self.custom_cmd_btn
        ]
        # 确保按钮存在再操作
        for btn in buttons_to_toggle:
            if hasattr(self, btn.objectName()): # 更安全的检查方式
                 btn.setEnabled(connected)
            # 或者简单的检查：
            # if btn: btn.setEnabled(connected)

        # 单独处理连接/断开按钮
        if hasattr(self, 'esp_connect_btn'):
            self.esp_connect_btn.setEnabled(not connected)
        if hasattr(self, 'esp_disconnect_btn'):
            self.esp_disconnect_btn.setEnabled(connected)

    def create_ai_control_tab(self): # (代码不变)
        tab = QWidget()
        layout = QVBoxLayout()
        control_group = QGroupBox("任务调度器控制")
        control_layout = QHBoxLayout()
        self.start_scheduler_btn = QPushButton("▶️ 启动调度器")
        self.start_scheduler_btn.clicked.connect(self.on_start_scheduler)
        self.stop_scheduler_btn = QPushButton("⏹️ 停止调度器")
        self.stop_scheduler_btn.clicked.connect(self.on_stop_scheduler)
        self.stop_scheduler_btn.setEnabled(False)
        control_layout.addWidget(self.start_scheduler_btn)
        control_layout.addWidget(self.stop_scheduler_btn)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        conn_status_group = QGroupBox("连接状态")
        conn_layout = QFormLayout()
        self.check_esp_label = QLabel("未知")
        self.check_camera_label = QLabel("未知")
        conn_layout.addRow("ESP 控制器:", self.check_esp_label)
        conn_layout.addRow("主摄像头 (ADB):", self.check_camera_label)
        check_btn = QPushButton("检查所有连接")
        check_btn.clicked.connect(self.on_check_connections)
        conn_layout.addRow(check_btn)
        conn_status_group.setLayout(conn_layout)
        layout.addWidget(conn_status_group)

        settings_group = QGroupBox("全局自动化设置")
        settings_layout = QFormLayout()
        self.ai_intervention_timeout_edit = QLineEdit(str(self.config.get("AI_INTERVENTION_TIMEOUT", 300))) # 使用 get 获取默认值
        settings_layout.addRow("AI干预超时 (秒):", self.ai_intervention_timeout_edit)
        self.human_intervention_mode_checkbox = QCheckBox("启用人工干预模式 (AI决策前需确认)")
        self.human_intervention_mode_checkbox.setChecked(self.config.get("ENABLE_HUMAN_INTERVENTION", False))
        settings_layout.addRow(self.human_intervention_mode_checkbox)
        self.human_intervention_alert_checkbox = QCheckBox("启用设备错误声音提示") # 标签更清晰
        self.human_intervention_alert_checkbox.setChecked(self.config.get("HUMAN_INTERVENTION_ALERT", True))
        settings_layout.addRow(self.human_intervention_alert_checkbox)

        save_global_settings_btn = QPushButton("应用全局设置 (不保存到文件)") # 按钮说明更清晰
        save_global_settings_btn.clicked.connect(self.on_apply_global_automation_settings)
        settings_layout.addRow(save_global_settings_btn)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def create_task_scheduler_tab(self): # (代码不变, 但注意 _create_task_table 的调用)
        tab = QWidget()
        layout = QHBoxLayout()
        device_panel = QWidget()
        device_layout = QVBoxLayout(device_panel)
        device_group = QGroupBox("设备管理")
        dg_layout = QVBoxLayout()
        self.device_table = QTableWidget()
        self.device_table.setColumnCount(6)
        self.device_table.setHorizontalHeaderLabels(["设备名称", "应用", "机位", "状态", "当前任务", "运行时长"])
        self.device_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.device_table.setSelectionMode(QTableWidget.SingleSelection)
        self.device_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive) # 改为 Interactive
        self.device_table.horizontalHeader().setStretchLastSection(True) # 拉伸最后一列
        self.device_table.verticalHeader().setVisible(False)
        self.device_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.device_table.itemSelectionChanged.connect(self.on_device_selection_changed)
        dg_layout.addWidget(self.device_table)
        dev_btn_layout = QHBoxLayout()
        add_dev_btn = QPushButton("➕ 添加")
        add_dev_btn.clicked.connect(self.on_add_device)
        self.edit_dev_btn = QPushButton("✏️ 编辑")
        self.edit_dev_btn.clicked.connect(self.on_edit_device)
        self.edit_dev_btn.setEnabled(False)
        self.remove_dev_btn = QPushButton("➖ 移除")
        self.remove_dev_btn.clicked.connect(self.on_remove_device)
        self.remove_dev_btn.setEnabled(False)
        dev_btn_layout.addWidget(add_dev_btn)
        dev_btn_layout.addWidget(self.edit_dev_btn)
        dev_btn_layout.addWidget(self.remove_dev_btn)
        dev_btn_layout.addStretch()
        self.stop_dev_task_btn = QPushButton("⏹️ 停止当前任务")
        self.stop_dev_task_btn.clicked.connect(self.on_stop_device_task)
        self.stop_dev_task_btn.setEnabled(False)
        dev_btn_layout.addWidget(self.stop_dev_task_btn)
        dg_layout.addLayout(dev_btn_layout)
        device_group.setLayout(dg_layout)
        device_layout.addWidget(device_group)

        task_panel = QWidget()
        task_layout = QVBoxLayout(task_panel)
        task_group = QGroupBox("任务管理")
        tg_layout = QVBoxLayout()
        self.task_tab_widget = QTabWidget()
        # 注意这里的列名和数量需要与 update_task_table 对应
        self.pending_task_table = self._create_task_table(["名称", "类型", "应用", "设备", "优先级", "重试"])
        self.running_task_table = self._create_task_table(["名称", "设备", "类型", "进度", "运行时长"])
        self.completed_task_table = self._create_task_table(["名称", "设备", "状态", "结束时间", "总时长", "错误信息"])
        self.task_tab_widget.addTab(self.pending_task_table, "⏳ 等待中 (0)")
        self.task_tab_widget.addTab(self.running_task_table, "▶️ 运行中 (0)")
        self.task_tab_widget.addTab(self.completed_task_table, "✅ 已完成/失败 (0)")
        tg_layout.addWidget(self.task_tab_widget)
        task_btn_layout = QHBoxLayout()
        add_task_btn = QPushButton("➕ 添加任务")
        add_task_btn.clicked.connect(self.on_add_task)
        # --- 新增编辑按钮 ---
        self.edit_task_btn = QPushButton("✏️ 编辑选中任务")
        self.edit_task_btn.clicked.connect(self.on_edit_task)
        self.edit_task_btn.setEnabled(False) # 默认禁用
        # --- 结束 ---
        self.cancel_task_btn = QPushButton("❌ 取消选中任务")
        self.cancel_task_btn.clicked.connect(self.on_cancel_selected_task)
        self.cancel_task_btn.setEnabled(False)
        task_btn_layout.addWidget(add_task_btn)
        task_btn_layout.addWidget(self.edit_task_btn) # 添加编辑按钮
        task_btn_layout.addWidget(self.cancel_task_btn)
        task_btn_layout.addStretch()
        tg_layout.addLayout(task_btn_layout)
        task_group.setLayout(tg_layout)
        task_layout.addWidget(task_group)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(device_panel)
        splitter.addWidget(task_panel)
        splitter.setSizes([600, 800]) # 调整比例
        layout.addWidget(splitter)
        tab.setLayout(layout)
        return tab

    def _create_task_table(self, headers: List[str]) -> QTableWidget:
        """辅助函数，用于创建标准的任务表格。"""
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        # 设置列宽调整模式为交互式，允许用户调整，但最后一列可以拉伸
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        # table.horizontalHeader().setStretchLastSection(True) # 在 update_task_table 中根据需要拉伸特定列
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # 不允许直接编辑
        table.setAlternatingRowColors(True)  # 隔行变色
        table.itemSelectionChanged.connect(self.on_task_selection_changed)  # 连接选择变化信号
        # 启用排序
        table.setSortingEnabled(True)
        return table

    def setup_ui_logging(self):
        """配置日志处理器以将日志显示在 UI 文本框中。"""
        # --- 系统日志处理器 ---
        log_handler = QTextEditLogger(self.log_text)
        # 设置日志格式
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        log_handler.setFormatter(log_formatter)
        # 将处理器添加到根 logger
        logging.getLogger().addHandler(log_handler)
        # 确保根 logger 的级别足够低以捕获所需信息 (INFO 及以上)
        logging.getLogger().setLevel(logging.INFO)
        logger.info("系统日志 UI 处理器已设置。")

        # --- AI 决策日志处理器 ---
        ai_decision_handler = QTextEditLogger(self.ai_decision_text)
        # 设置 AI 日志格式
        ai_formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')
        ai_decision_handler.setFormatter(ai_formatter)
        # 将处理器添加到特定的 AI logger
        ai_decision_logger.addHandler(ai_decision_handler)
        # 确保 AI logger 的级别也设置正确
        ai_decision_logger.setLevel(logging.INFO)
        ai_decision_logger.info("AI 决策日志 UI 处理器已设置。")  # 使用 AI logger 记录

    @pyqtSlot(str, str)
    def update_task_progress_display(self, device_name: str, progress_text: str):
        """槽函数：更新任务详细日志文本区域。"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] [{device_name}] {progress_text}"
        self.task_progress_text.append(log_entry)
        # 自动滚动到底部，以便始终看到最新的日志
        scrollbar = self.task_progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def create_settings_tab(self): # (代码不变)
        tab = QWidget()
        layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        adb_group = QGroupBox("ADB & 截图设置")
        adb_layout = QFormLayout()
        adb_path_layout = QHBoxLayout()
        self.adb_path_edit = QLineEdit(self.config["ADB_PATH"])
        browse_adb_btn = QPushButton("浏览...")
        browse_adb_btn.clicked.connect(self.on_browse_adb)
        adb_path_layout.addWidget(self.adb_path_edit, 1)
        adb_path_layout.addWidget(browse_adb_btn)
        adb_layout.addRow("ADB路径:", adb_path_layout)
        cam_dev_layout = QHBoxLayout()
        self.camera_device_edit = QLineEdit(self.config.get("CAMERA_DEVICE_ID", "")) # 使用 get
        refresh_devices_btn = QPushButton("刷新设备列表")
        refresh_devices_btn.clicked.connect(self.on_refresh_devices)
        cam_dev_layout.addWidget(self.camera_device_edit, 1)
        cam_dev_layout.addWidget(refresh_devices_btn)
        adb_layout.addRow("主摄像头设备ID:", cam_dev_layout)
        self.screenshot_res_edit = QLineEdit(f"{self.config['SCREENSHOT_RESOLUTION'][0]}x{self.config['SCREENSHOT_RESOLUTION'][1]}")
        self.cropped_res_edit = QLineEdit(f"{self.config['CROPPED_RESOLUTION'][0]}x{self.config['CROPPED_RESOLUTION'][1]}")
        adb_layout.addRow("默认截图分辨率 (WxH):", self.screenshot_res_edit)
        adb_layout.addRow("默认裁剪分辨率 (WxH):", self.cropped_res_edit)
        adb_group.setLayout(adb_layout)
        settings_layout.addWidget(adb_group)

        ocr_group = QGroupBox("OCR 设置")
        ocr_layout = QFormLayout()
        self.ocr_url_edit = QLineEdit(self.config["OCR_API_URL"])
        ocr_layout.addRow("Umi-OCR API 地址:", self.ocr_url_edit)
        self.ocr_confidence_edit = QLineEdit(str(self.config["OCR_CONFIDENCE_THRESHOLD"]))
        ocr_layout.addRow("OCR 可信度阈值 (0-1):", self.ocr_confidence_edit)
        ocr_test_btn = QPushButton("测试 OCR 连接")
        ocr_test_btn.clicked.connect(self.on_test_ocr)
        ocr_layout.addRow(ocr_test_btn)
        ocr_group.setLayout(ocr_layout)
        settings_layout.addWidget(ocr_group)

        ai_group = QGroupBox("AI API 设置")
        ai_layout = QFormLayout()
        self.ai_url_edit = QLineEdit(self.config["AI_API_URL"])
        ai_layout.addRow("API 地址:", self.ai_url_edit)
        self.ai_key_edit = QLineEdit(self.config["AI_API_KEY"])
        self.ai_key_edit.setEchoMode(QLineEdit.Password)
        ai_layout.addRow("API 密钥:", self.ai_key_edit)
        self.ai_model_edit = QLineEdit(self.config["AI_MODEL"])
        ai_layout.addRow("AI 模型名称:", self.ai_model_edit)
        ai_test_btn = QPushButton("测试 AI API")
        ai_test_btn.clicked.connect(self.on_test_ai_api)
        ai_layout.addRow(ai_test_btn)
        ai_group.setLayout(ai_layout)
        settings_layout.addWidget(ai_group)

        template_group = QGroupBox("模板匹配设置")
        template_layout = QFormLayout()
        self.template_threshold_edit = QLineEdit(str(self.config["TEMPLATE_MATCHING_THRESHOLD"]))
        template_layout.addRow("默认匹配阈值 (0-1):", self.template_threshold_edit)
        open_template_folder_btn = QPushButton("打开模板文件夹")
        open_template_folder_btn.clicked.connect(self.on_open_template_folder)
        template_layout.addRow(open_template_folder_btn)
        template_group.setLayout(template_layout)
        settings_layout.addWidget(template_group)

        settings_layout.addStretch(1)
        scroll_area.setWidget(settings_widget)
        layout.addWidget(scroll_area)

        save_btn = QPushButton("💾 保存所有设置到 config.json")
        save_btn.setStyleSheet("padding: 5px;")
        save_btn.clicked.connect(self.on_save_settings)
        layout.addWidget(save_btn)

        tab.setLayout(layout)
        return tab


    # --- 截图和模板定义相关方法 ---
    @pyqtSlot()
    def reset_screenshot_zoom(self):
        """重置截图视图的缩放和平移。"""
        if self.current_pixmap_item:
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            self.graphics_view.setTransform(QTransform())
        if self.defining_template_mode:
            self.define_template_btn.setChecked(False)

    @pyqtSlot(bool)
    def on_toggle_define_template_mode(self, checked):
        """切换是否进入框选定义模板模式。"""
        self.defining_template_mode = checked
        if checked:
             # 检查是否有截图可供定义
             if self.last_raw_screenshot_for_template is None:
                 QMessageBox.warning(self, "无法定义模板", "请先获取截图，然后才能框选定义模板。")
                 self.define_template_btn.setChecked(False) # 自动取消勾选
                 return

             logger.info("进入模板定义模式，请在截图上框选区域。")
             self.graphics_view.setDragMode(QGraphicsView.NoDrag)
             QApplication.setOverrideCursor(Qt.CrossCursor)
             if self.template_selection_rect_item and self.template_selection_rect_item in self.graphics_scene.items():
                 self.graphics_scene.removeItem(self.template_selection_rect_item)
             self.template_selection_rect_item = None
             self.template_rect_start_pos = None
        else:
             logger.info("退出模板定义模式。")
             self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
             QApplication.restoreOverrideCursor()
             if self.template_selection_rect_item and self.template_selection_rect_item in self.graphics_scene.items():
                 self.graphics_scene.removeItem(self.template_selection_rect_item)
             self.template_selection_rect_item = None
             self.template_rect_start_pos = None

    def graphics_view_mouse_press_event(self, event):
        """处理截图视图上的鼠标按下事件，用于开始框选模板或触发像素坐标点击。"""
        scene_pos = self.graphics_view.mapToScene(event.pos())

        if self.defining_template_mode and event.button() == Qt.LeftButton:
            self.template_rect_start_pos = scene_pos
            if self.template_selection_rect_item and self.template_selection_rect_item in self.graphics_scene.items():
                self.graphics_scene.removeItem(self.template_selection_rect_item)
            pen = QPen(QColor("red"), 1, Qt.DashLine)
            self.template_selection_rect_item = self.graphics_scene.addRect(QRectF(scene_pos, scene_pos), pen)
            self.template_selection_rect_item.setZValue(10)
            event.accept()
        elif event.button() == Qt.LeftButton and not self.defining_template_mode and self.current_pixmap_item:
            # --- 左键点击非框选模式：触发像素坐标点击 ---
            item_pos = self.current_pixmap_item.mapFromScene(scene_pos)
            pixmap = self.current_pixmap_item.pixmap()
            if 0 <= item_pos.x() < pixmap.width() and 0 <= item_pos.y() < pixmap.height():
                 pixel_x = int(item_pos.x())
                 pixel_y = int(item_pos.y())
                 logger.info(f"检测到截图区域左键点击，像素坐标: ({pixel_x}, {pixel_y})")
                 # 可以选择直接发送点击命令，或者填充到手动命令区域
                 self.pixel_click_x_edit.setText(str(pixel_x))
                 self.pixel_click_y_edit.setText(str(pixel_y))
                 QToolTip.showText(event.globalPos(), f"像素坐标: ({pixel_x}, {pixel_y})", self.graphics_view)
                 # 可以在这里直接调用点击函数（如果 ESP 已连接）
                 # if self.esp_controller.connected:
                 #     self.on_esp_pixel_click()
                 event.accept()
            else:
                 super(QGraphicsView, self.graphics_view).mousePressEvent(event) # 点击在图像外，执行默认拖动
        else:
            super(QGraphicsView, self.graphics_view).mousePressEvent(event) # 其他情况（右键等）执行默认拖动

    def graphics_view_mouse_move_event(self, event): # (代码不变)
        """处理截图视图上的鼠标移动事件，用于更新框选矩形。"""
        if self.defining_template_mode and self.template_rect_start_pos and self.template_selection_rect_item:
            current_pos = self.graphics_view.mapToScene(event.pos())
            rect = QRectF(self.template_rect_start_pos, current_pos).normalized()
            self.template_selection_rect_item.setRect(rect)
            event.accept()
        else:
            super(QGraphicsView, self.graphics_view).mouseMoveEvent(event)

    def graphics_view_mouse_release_event(self, event): # (代码不变)
        """处理截图视图上的鼠标释放事件，用于完成框选并定义模板。"""
        if self.defining_template_mode and event.button() == Qt.LeftButton and self.template_rect_start_pos and self.template_selection_rect_item:
            current_pos = self.graphics_view.mapToScene(event.pos())
            selection_rect_scene = QRectF(self.template_rect_start_pos, current_pos).normalized()
            self.template_rect_start_pos = None # 重置起始点

            if selection_rect_scene.width() < 5 or selection_rect_scene.height() < 5:
                logger.warning("选择的模板区域太小，已取消。")
                if self.template_selection_rect_item in self.graphics_scene.items(): self.graphics_scene.removeItem(self.template_selection_rect_item)
                self.template_selection_rect_item = None; self.define_template_btn.setChecked(False)
                event.accept(); return

            if self.last_raw_screenshot_for_template is None:
                QMessageBox.warning(self, "错误", "没有可用于定义模板的原始截图。"); self.define_template_btn.setChecked(False)
                if self.template_selection_rect_item in self.graphics_scene.items(): self.graphics_scene.removeItem(self.template_selection_rect_item)
                self.template_selection_rect_item = None; event.accept(); return

            if not self.current_pixmap_item: logger.error("无法进行坐标转换，当前没有Pixmap项。"); event.accept(); return

            # --- 坐标转换：场景 -> Item -> 原始图像 ---
            try:
                transform = self.current_pixmap_item.sceneTransform()
                inverse_transform, invertible = transform.inverted()
                if not invertible: raise ValueError("Pixmap transform is not invertible")

                # 场景坐标转 Item 坐标
                item_top_left = inverse_transform.map(selection_rect_scene.topLeft())
                item_bottom_right = inverse_transform.map(selection_rect_scene.bottomRight())
                selection_rect_item = QRectF(item_top_left, item_bottom_right).normalized()

                # Item 坐标转原始图像像素坐标 (假设 Item 未被缩放或扭曲，其坐标系与 Pixmap 一致)
                pixmap = self.current_pixmap_item.pixmap()
                pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
                img_h, img_w = self.last_raw_screenshot_for_template.shape[:2]

                # 确保比例一致，否则需要缩放坐标
                if pixmap_w != img_w or pixmap_h != img_h:
                     logger.warning("Pixmap 尺寸与原始图像不符，模板可能不准确。")
                     # 可以选择报错或按比例缩放坐标，这里简化为直接使用，可能不准
                     # scale_x = img_w / pixmap_w
                     # scale_y = img_h / pixmap_h
                     # item_x = int(selection_rect_item.x() * scale_x) ... etc.

                # 提取整数坐标并限制在图像范围内
                item_x = max(0, int(selection_rect_item.x()))
                item_y = max(0, int(selection_rect_item.y()))
                item_w = min(img_w - item_x, int(selection_rect_item.width()))
                item_h = min(img_h - item_y, int(selection_rect_item.height()))

                if item_w <= 0 or item_h <= 0: logger.warning("计算出的模板像素尺寸无效。"); event.accept(); return

                # 裁剪原始图像
                cropped_image = self.last_raw_screenshot_for_template[item_y: item_y + item_h, item_x: item_x + item_w]
                logger.info(f"从原始图像裁剪区域: x={item_x}, y={item_y}, w={item_w}, h={item_h}")

                # 弹出对话框获取名称
                template_name, ok = QInputDialog.getText(self, "定义模板", "请输入模板名称:", QLineEdit.Normal, "")
                if ok and template_name and template_name.strip():
                    save_success = self.ai_analyzer.add_template(template_name.strip(), cropped_image)
                    if save_success:
                        QMessageBox.information(self, "模板已保存", f"模板 '{template_name.strip()}' 已成功保存。")
                    else:
                        QMessageBox.critical(self, "保存失败", f"无法保存模板 '{template_name.strip()}'。\n请查看日志。")
                else: logger.info("用户取消了模板定义或名称无效。")

            except Exception as e:
                logger.error(f"定义模板时出错: {e}", exc_info=True)
                QMessageBox.critical(self, "模板定义错误", f"处理模板时发生错误:\n{e}")

            # 清理并退出模式
            if self.template_selection_rect_item in self.graphics_scene.items(): self.graphics_scene.removeItem(self.template_selection_rect_item)
            self.template_selection_rect_item = None; self.define_template_btn.setChecked(False); event.accept()
        else:
            super(QGraphicsView, self.graphics_view).mouseReleaseEvent(event)

    def graphics_view_wheel_event(self, event): # (代码不变)
        """处理 QGraphicsView 上的鼠标滚轮事件以进行缩放。"""
        if not self.current_pixmap_item:
            QGraphicsView.wheelEvent(self.graphics_view, event); return
        zoom_in_factor = 1.15; zoom_out_factor = 1 / zoom_in_factor
        old_transformation_anchor = self.graphics_view.transformationAnchor()
        old_resize_anchor = self.graphics_view.resizeAnchor()
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0: self.graphics_view.scale(zoom_in_factor, zoom_in_factor)
        else: self.graphics_view.scale(zoom_out_factor, zoom_out_factor)
        self.graphics_view.setTransformationAnchor(old_transformation_anchor)
        self.graphics_view.setResizeAnchor(old_resize_anchor)

    @pyqtSlot()
    def on_save_current_screenshot(self): # (代码不变)
        """将当前显示的截图保存到用户指定位置"""
        if self.last_raw_screenshot_for_template is None:
            QMessageBox.warning(self, "无法保存", "当前没有可供保存的截图。"); return
        default_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存截图", default_filename, "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;所有文件 (*)")
        if save_path:
            try:
                success = cv2.imwrite(save_path, self.last_raw_screenshot_for_template)
                if success:
                    logger.info(f"当前截图已保存到: {save_path}"); QMessageBox.information(self, "保存成功", f"截图已保存到:\n{save_path}")
                else:
                    logger.error(f"使用 cv2.imwrite 保存截图到 {save_path} 失败。"); QMessageBox.warning(self, "保存失败", f"无法将截图保存到指定位置。")
            except Exception as e:
                logger.error(f"保存截图到 {save_path} 时发生错误: {e}", exc_info=True); QMessageBox.critical(self, "保存错误", f"保存截图时发生错误:\n{e}")

    @pyqtSlot(object, str)
    def display_screenshot(self, cv_image: Optional[np.ndarray], source_info: str = "未知来源"): # (代码不变)
        """在 QGraphicsView 中显示 OpenCV 图像（带标注）。"""
        can_save_and_define = False
        if cv_image is not None:
            try:
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3: self.last_raw_screenshot_for_template = cv_image.copy()
                elif len(cv_image.shape) == 2: self.last_raw_screenshot_for_template = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                else: self.last_raw_screenshot_for_template = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
                can_save_and_define = True
            except Exception as conv_err:
                logger.warning(f"无法将截图转换为BGR格式进行备份: {conv_err}"); self.last_raw_screenshot_for_template = None
        else: self.last_raw_screenshot_for_template = None

        self.save_current_screenshot_btn.setEnabled(can_save_and_define)
        # 只有在非定义模式下，才根据是否有图来启用按钮
        if not self.defining_template_mode:
            self.define_template_btn.setEnabled(can_save_and_define)

        if cv_image is None:
            self.graphics_scene.clear(); self.current_pixmap_item = None
            if self.template_selection_rect_item: self.template_selection_rect_item = None
            self.template_rect_start_pos = None
            text_item = self.graphics_scene.addText(f"无可用截图\n({source_info})"); text_item.setDefaultTextColor(Qt.white)
            # self.graphics_view.centerOn(text_item) # 可能导致视图混乱，移除
            self.graphics_view.resetTransform() # 重置视图变换
            return
        try:
            if len(cv_image.shape) == 3: h, w, ch = cv_image.shape; fmt = QImage.Format_RGB888; img_data = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).data
            elif len(cv_image.shape) == 2: h, w = cv_image.shape; ch = 1; fmt = QImage.Format_Grayscale8; img_data = cv_image.data
            else: raise ValueError("不支持的图像维度")
            bytes_per_line = ch * w
            qimg = QImage(img_data, w, h, bytes_per_line, fmt)
            if qimg.isNull(): logger.error("从截图数据创建 QImage 失败。"); self.graphics_scene.clear(); self.current_pixmap_item = None; return

            pixmap = QPixmap.fromImage(qimg)
            current_center = self.graphics_view.mapToScene(self.graphics_view.viewport().rect().center())
            current_transform = self.graphics_view.transform()
            is_first_image = self.current_pixmap_item is None

            self.graphics_scene.clear()
            self.current_pixmap_item = self.graphics_scene.addPixmap(pixmap)
            self.graphics_scene.setSceneRect(self.current_pixmap_item.boundingRect())

            if is_first_image: self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
            else:
                try: # 恢复之前的视图变换和中心点
                    if current_transform.isAffine(): self.graphics_view.setTransform(current_transform); self.graphics_view.centerOn(current_center)
                    else: self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio) # 变换无效则重置
                except Exception as view_err: logger.warning(f"恢复视图时出错: {view_err}"); self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)

            # 如果在定义模式，重新绘制矩形框
            if self.defining_template_mode and self.template_rect_start_pos:
                pen = QPen(QColor("red"), 1, Qt.DashLine); self.template_selection_rect_item = self.graphics_scene.addRect(QRectF(self.template_rect_start_pos, self.template_rect_start_pos), pen); self.template_selection_rect_item.setZValue(10)
        except Exception as e: logger.error(f"显示截图时出错: {e}", exc_info=True); self.graphics_scene.clear(); self.current_pixmap_item = None


    # --- UI 更新逻辑 (`update_ui_elements`, `update_device_table`, `update_task_table`, `_update_ocr_display`) ---
    # (大部分保持不变，但 update_task_table 需要适配新列)
    def update_ui_elements(self):
        """【已修改】周期性 UI 更新函数，调用定时器。简化截图和 OCR 更新逻辑。"""
        # 更新连接状态标签
        self.check_esp_label.setText("已连接" if self.esp_controller.connected else "未连接")
        cam_connected = False
        if self.screenshot_manager.primary_device_id:
            cam_connected = self.screenshot_manager.connected_devices.get(self.screenshot_manager.primary_device_id, False)
        self.check_camera_label.setText("已连接" if cam_connected else "未连接")

        # 更新按钮状态
        self.update_esp_button_states(self.esp_controller.connected)
        scheduler_running = (self.task_scheduler.scheduler_thread is not None and
                             self.task_scheduler.scheduler_thread.is_alive())
        self.start_scheduler_btn.setEnabled(not scheduler_running)
        self.stop_scheduler_btn.setEnabled(scheduler_running)
        self.manual_screenshot_btn.setEnabled(cam_connected)

        # --- 简化截图和 OCR 显示逻辑 ---
        active_device_screenshot = None
        active_device_ocr = None
        active_device_name = "无活动设备"

        # 优先显示有活动任务且刚更新截图的设备
        running_devices_with_data = [
            dev for dev_name, task in self.task_scheduler.running_tasks.items()
            if (dev := self.task_scheduler.devices.get(dev_name)) and dev.last_screenshot is not None
        ]

        if running_devices_with_data:
            # 按最后更新时间排序，最新的在前面
            running_devices_with_data.sort(key=lambda d: d.last_update_time, reverse=True)
            active_device = running_devices_with_data[0]
            active_device_screenshot = active_device.last_screenshot
            active_device_ocr = active_device.last_ocr_result
            active_device_name = active_device.name
        else:
            # 如果没有运行中的设备有截图，则显示最后一次手动或捕获的截图
            active_device_screenshot = self.screenshot_manager.last_captured_image
            if active_device_screenshot is not None:
                active_device_name = "手动/上次截图"
                # 尝试查找最近的 OCR 结果（不一定匹配当前截图）
                latest_ocr_time = None
                latest_ocr_result = None
                for dev in self.task_scheduler.devices.values():
                     if dev.last_ocr_result and (latest_ocr_time is None or dev.last_update_time > latest_ocr_time):
                         latest_ocr_time = dev.last_update_time
                         latest_ocr_result = dev.last_ocr_result
                active_device_ocr = latest_ocr_result


        # 【修改】总是调用 display_screenshot 和 _update_ocr_display
        # 让这两个函数内部处理 None 的情况
        self.display_screenshot(active_device_screenshot, active_device_name)
        self._update_ocr_display(active_device_ocr, active_device_name)
        # --- 修改结束 ---

    def update_device_table(self): # (代码不变)
        selected_device_name = None
        if self.device_table.selectedItems():
            selected_row = self.device_table.selectedItems()[0].row()
            if selected_row < self.device_table.rowCount():
                name_item = self.device_table.item(selected_row, 0)
                if name_item: selected_device_name = name_item.text()

        self.device_table.setSortingEnabled(False)
        self.device_table.setRowCount(0)
        devices = sorted(list(self.task_scheduler.devices.values()), key=lambda d: d.name)
        status_colors = {DeviceStatus.IDLE: QColor("#c8e6c9"), DeviceStatus.BUSY: QColor("#fff9c4"), DeviceStatus.INITIALIZING: QColor("#bbdefb"), DeviceStatus.WAITING: QColor("#e1bee7"), DeviceStatus.ERROR: QColor("#ffcdd2"), DeviceStatus.DISCONNECTED: QColor("#cfd8dc")}

        new_selected_row = -1
        for i, device in enumerate(devices):
            self.device_table.insertRow(i)
            self.device_table.setItem(i, 0, QTableWidgetItem(device.name))
            self.device_table.setItem(i, 1, QTableWidgetItem(", ".join(device.apps)))
            self.device_table.setItem(i, 2, QTableWidgetItem(device.position))
            status_item = QTableWidgetItem(device.status.value); status_item.setBackground(status_colors.get(device.status, QColor("white"))); status_item.setTextAlignment(Qt.AlignCenter)
            self.device_table.setItem(i, 3, status_item)
            task_name = device.current_task.name if device.current_task else "-"
            self.device_table.setItem(i, 4, QTableWidgetItem(task_name))
            runtime = device.get_runtime() if device.status not in [DeviceStatus.IDLE, DeviceStatus.DISCONNECTED] else "-"
            runtime_item = QTableWidgetItem(runtime); runtime_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.device_table.setItem(i, 5, runtime_item)
            if device.name == selected_device_name: new_selected_row = i

        self.device_table.setSortingEnabled(True)
        if new_selected_row != -1: self.device_table.selectRow(new_selected_row)
        else: self.device_table.clearSelection()
        self.on_device_selection_changed()

        self._populate_coord_debug_tab_devices()

    def update_task_table(self): # (适配新列，逻辑不变)
        """更新任务表（等待中、运行中、已完成），包含 task_id 和重试信息。"""
        try:
            task_lists = self.task_scheduler.get_task_lists()

            # --- 填充等待中表格 (增加重试列) ---
            pending_table = self.pending_task_table
            pending_table.setSortingEnabled(False)
            # 保存当前选中行
            selected_pending_id = None
            if pending_table.selectedItems():
                row = pending_table.selectedItems()[0].row()
                id_item = pending_table.item(row, 0)
                if id_item: selected_pending_id = id_item.data(Qt.UserRole)
            pending_table.setRowCount(0)
            pending_tasks = task_lists.get("pending", [])
            self.task_tab_widget.setTabText(0, f"⏳ 等待中 ({len(pending_tasks)})")
            new_pending_selection = -1
            for i, task in enumerate(pending_tasks):
                pending_table.insertRow(i)
                name_item = QTableWidgetItem(task.name); name_item.setData(Qt.UserRole, task.task_id) # 存储 ID
                pending_table.setItem(i, 0, name_item)
                pending_table.setItem(i, 1, QTableWidgetItem(task.type.value))
                pending_table.setItem(i, 2, QTableWidgetItem(task.app_name or "-"))
                pending_table.setItem(i, 3, QTableWidgetItem(task.assigned_device_name or "自动"))
                prio_item = QTableWidgetItem(str(task.priority)); prio_item.setTextAlignment(Qt.AlignCenter)
                pending_table.setItem(i, 4, prio_item)
                retry_str = f"{task.retry_count}/{task.max_retries}" if task.max_retries > 0 else "-"
                retry_item = QTableWidgetItem(retry_str); retry_item.setTextAlignment(Qt.AlignCenter)
                if task.retry_count > 0: retry_item.setBackground(QColor("#ffe0b2")) # 橙色背景提示重试
                pending_table.setItem(i, 5, retry_item)
                if task.task_id == selected_pending_id: new_pending_selection = i # 记录新行号
            pending_table.resizeColumnsToContents() # 调整列宽
            pending_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch) # 名称列拉伸
            pending_table.setSortingEnabled(True)
            if new_pending_selection != -1: pending_table.selectRow(new_pending_selection) # 恢复选中

            # --- 填充运行中表格 (使用 get_progress_display) ---
            running_table = self.running_task_table
            running_table.setSortingEnabled(False)
            selected_running_id = None
            if running_table.selectedItems():
                 row = running_table.selectedItems()[0].row()
                 id_item = running_table.item(row, 0)
                 if id_item: selected_running_id = id_item.data(Qt.UserRole)
            running_table.setRowCount(0)
            running_tasks = task_lists.get("running", [])
            self.task_tab_widget.setTabText(1, f"▶️ 运行中 ({len(running_tasks)})")
            new_running_selection = -1
            for i, task in enumerate(running_tasks):
                running_table.insertRow(i)
                name_item = QTableWidgetItem(task.name); name_item.setData(Qt.UserRole, task.task_id) # 存储 ID
                running_table.setItem(i, 0, name_item)
                running_table.setItem(i, 1, QTableWidgetItem(task.assigned_device_name or "未知"))
                running_table.setItem(i, 2, QTableWidgetItem(task.type.value))
                progress_item = QTableWidgetItem(task.get_progress_display()) # 使用包含重试信息的方法
                running_table.setItem(i, 3, progress_item)
                runtime_item = QTableWidgetItem(task.get_runtime()); runtime_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                running_table.setItem(i, 4, runtime_item)
                if task.task_id == selected_running_id: new_running_selection = i
            running_table.resizeColumnsToContents()
            running_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch) # 进度列拉伸
            running_table.setSortingEnabled(True)
            if new_running_selection != -1: running_table.selectRow(new_running_selection)

            # --- 填充已完成表格 (代码不变) ---
            completed_table = self.completed_task_table
            completed_table.setSortingEnabled(False)
            # 保存滚动条位置
            # scroll_pos = completed_table.verticalScrollBar().value()
            # 保存选中
            selected_completed_id = None
            if completed_table.selectedItems():
                 row = completed_table.selectedItems()[0].row()
                 id_item = completed_table.item(row, 0)
                 if id_item: selected_completed_id = id_item.data(Qt.UserRole)
            completed_table.setRowCount(0)
            completed_tasks = task_lists.get("completed", [])
            self.task_tab_widget.setTabText(2, f"✅ 已完成/失败 ({len(completed_tasks)})")
            new_completed_selection = -1
            for i, task in enumerate(reversed(completed_tasks)): # 保持倒序插入
                completed_table.insertRow(i)
                name_item = QTableWidgetItem(task.name); name_item.setData(Qt.UserRole, task.task_id) # 存储 ID
                completed_table.setItem(i, 0, name_item)
                completed_table.setItem(i, 1, QTableWidgetItem(task.assigned_device_name or "-"))
                status_item = QTableWidgetItem(task.status.value)
                if task.status == TaskStatus.FAILED: status_item.setBackground(QColor("#ffcdd2"))
                elif task.status == TaskStatus.CANCELED: status_item.setBackground(QColor("#eeeeee"))
                else: status_item.setBackground(QColor("#c8e6c9"))
                completed_table.setItem(i, 2, status_item)
                end_time_str = task.end_time.strftime("%Y-%m-%d %H:%M:%S") if task.end_time else "-"
                completed_table.setItem(i, 3, QTableWidgetItem(end_time_str))
                runtime_item = QTableWidgetItem(task.get_runtime()); runtime_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                completed_table.setItem(i, 4, runtime_item)
                error_item = QTableWidgetItem(task.error or ""); error_item.setToolTip(task.error or "") # 添加 Tooltip 显示完整错误
                completed_table.setItem(i, 5, error_item)
                if task.task_id == selected_completed_id: new_completed_selection = i
            completed_table.resizeColumnsToContents()
            completed_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch) # 错误信息列拉伸
            completed_table.setSortingEnabled(True)
            # 恢复滚动条位置 (如果需要)
            # completed_table.verticalScrollBar().setValue(scroll_pos)
            if new_completed_selection != -1: completed_table.selectRow(new_completed_selection) # 恢复选中

            self.on_task_selection_changed() # 更新按钮状态

        except Exception as e:
            logger.error(f"更新任务表时出错: {e}", exc_info=True)

    @pyqtSlot(dict)
    def _handle_intervention_request_ui(self, intervention_data: dict):
        """槽函数：在主线程中弹出人工干预对话框。"""
        # 确保在主线程执行 (虽然信号连接机制通常会保证)
        if threading.current_thread() != threading.main_thread():
            logger.warning("尝试在非主线程处理人工干预 UI 请求，跳过。")
            # 理论上 QueuedConnection 会处理，这里是额外检查
            return

        dialog = HumanInterventionDialog(
            action_str=intervention_data.get("action_str", "N/A"),
            justification=intervention_data.get("justification", "N/A"),
            ai_prompt=intervention_data.get("ai_prompt", "N/A"),
            ai_response=intervention_data.get("ai_response_raw", "N/A"),
            parent=self # 设置父窗口
        )
        result = dialog.exec_() # 显示模态对话框

        # 将结果写回 TaskExecutor 的共享变量
        with QMutexLocker(self.task_executor.intervention_mutex):
            self.task_executor.intervention_result = (result == QDialog.Accepted)
        logger.debug(f"人工干预对话框结果: {'Accepted' if result == QDialog.Accepted else 'Rejected'}")

    def _update_ocr_display(self, ocr_result: Optional[Dict[str, Any]], source_info: str = "未知来源"): # (代码不变)
        """Updates the OCR text display area."""
        if ocr_result and "data" in ocr_result and isinstance(ocr_result["data"], list):
            text_content = f"来源: {source_info} | 时间: {datetime.now().strftime('%H:%M:%S')}\n"
            text_content += f"识别文本数: {len(ocr_result['data'])}\n---\n"
            lines = []
            for item in ocr_result["data"]:
                if isinstance(item, dict) and "text" in item:
                    conf = item.get('confidence')
                    conf_str = f" [{conf:.2f}]" if conf is not None else "" # 修正：检查 conf 是否为 None
                    lines.append(f"{item['text']}{conf_str}")
            text_content += "\n".join(lines)
            self.ocr_text.setText(text_content)
        elif ocr_result and "error" in ocr_result:
            self.ocr_text.setText(f"来源: {source_info}\nOCR 错误: {ocr_result['error']}")
        elif ocr_result and ocr_result.get("code") == 101: # Code 101 means success but no text found
             self.ocr_text.setText(f"来源: {source_info} | 时间: {datetime.now().strftime('%H:%M:%S')}\nOCR 成功，未检测到文本。")
        else: # No result or other error
            self.ocr_text.setText(f"来源: {source_info}\n(无有效OCR结果)")

    # --- 事件处理器 (Slots) ---
    def on_device_selection_changed(self): # (代码不变)
        """Updates button states when device selection changes."""
        selected = bool(self.device_table.selectedItems())
        self.edit_dev_btn.setEnabled(selected)
        self.remove_dev_btn.setEnabled(selected)
        stop_enabled = False
        if selected:
            row = self.device_table.selectedItems()[0].row()
            if row < self.device_table.rowCount():
                device_name_item = self.device_table.item(row, 0)
                if device_name_item:
                    device_name = device_name_item.text()
                    device = self.task_scheduler.get_device(device_name)
                    # 允许停止 BUSY, INITIALIZING, WAITING, ERROR 状态的任务
                    if device and device.status in [DeviceStatus.BUSY, DeviceStatus.INITIALIZING, DeviceStatus.WAITING, DeviceStatus.ERROR]:
                        stop_enabled = True
        self.stop_dev_task_btn.setEnabled(stop_enabled)

    def on_task_selection_changed(self):
        """Updates button states when task selection changes in any table."""
        cancel_enabled = False
        edit_enabled = False
        current_table = self.task_tab_widget.currentWidget()
        if isinstance(current_table, QTableWidget) and current_table.selectedItems():
            # 允许取消 PENDING 或 RUNNING 任务
            if current_table == self.pending_task_table or current_table == self.running_task_table:
                cancel_enabled = True
            # 只允许编辑 PENDING 任务
            if current_table == self.pending_task_table:
                edit_enabled = True
        self.cancel_task_btn.setEnabled(cancel_enabled)
        self.edit_task_btn.setEnabled(edit_enabled) # 更新编辑按钮状态

    # --- ESP 控制相关函数 ---
    def on_esp_connect(self): # (代码不变)
        self.config["ESP_IP"] = self.esp_ip_edit.text()
        try: self.config["ESP_PORT"] = int(self.esp_port_edit.text())
        except ValueError: QMessageBox.warning(self, "错误", "请输入有效的 ESP 端口号。"); return
        self.esp_controller.esp_ip = self.config["ESP_IP"]
        self.esp_controller.esp_port = self.config["ESP_PORT"]
        self.esp_response_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 连接 ESP...")
        QApplication.processEvents()
        success = self.esp_controller.connect()
        self.esp_response_text.append(f"-> {'连接成功' if success else '连接失败'}")
        self.update_esp_button_states(success)

    def on_esp_disconnect(self): # (代码不变)
        self.esp_response_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 断开 ESP...")
        QApplication.processEvents()
        self.esp_controller.disconnect()
        self.esp_response_text.append("-> 已断开")
        self.update_esp_button_states(False)

    def _send_esp_and_log(self, command_func: Callable, *args): # (代码不变)
        """Helper to send ESP command and log response."""
        if not self.esp_controller.connected: QMessageBox.warning(self, "错误", "ESP未连接"); return
        cmd_name = args[0] if command_func == self.esp_controller.send_command else command_func.__name__
        cmd_args_str = f"{args[1:]}" if command_func == self.esp_controller.send_command else f"{args}"
        self.esp_response_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 发送: {cmd_name}{cmd_args_str}")
        QApplication.processEvents()
        try:
            result = command_func(*args)
            response_str = json.dumps(result, ensure_ascii=False, indent=2) # 美化输出
            self.esp_response_text.append(f"-> 响应: {response_str}")
            if not result.get("success"): QMessageBox.warning(self, "ESP命令失败", f"命令执行失败:\n{result.get('error', '未知错误')}")
        except Exception as e: logger.error(f"Error sending ESP command via UI: {e}", exc_info=True); self.esp_response_text.append(f"-> 错误: {e}"); QMessageBox.critical(self, "ESP命令错误", f"执行命令时出错: {e}")

    def on_esp_home(self): self._send_esp_and_log(self.esp_controller.home) # (代码不变)
    def on_esp_set_origin(self): self._send_esp_and_log(self.esp_controller.set_origin) # (代码不变)
    def on_esp_status(self): self._send_esp_and_log(self.esp_controller.get_status) # (代码不变)
    def on_move_x(self): # (代码不变)
        try: distance = float(self.x_distance_edit.text())
        except ValueError: QMessageBox.warning(self, "输入错误", "请输入有效的 X 轴移动距离。"); return
        self._send_esp_and_log(self.esp_controller.move_x, distance)
    def on_move_y(self): # (代码不变)
        try: distance = float(self.y_distance_edit.text())
        except ValueError: QMessageBox.warning(self, "输入错误", "请输入有效的 Y 轴移动距离。"); return
        self._send_esp_and_log(self.esp_controller.move_y, distance)

    def on_esp_pixel_click(self):
        """处理手动像素坐标点击按钮。"""
        try:
            x = int(self.pixel_click_x_edit.text())
            y = int(self.pixel_click_y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的整数像素坐标。"); return
        self._send_esp_and_log(self.esp_controller.pixel_click, x, y)

    def on_esp_phys_click(self):
        """处理手动物理坐标点击按钮。"""
        try:
            x = float(self.phys_click_x_edit.text()) # 物理坐标允许浮点数
            y = float(self.phys_click_y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的物理坐标 (mm)。"); return
        # 直接调用 ESPController 的 click 方法，它接收物理坐标
        self._send_esp_and_log(self.esp_controller.click, x, y)

    def on_send_custom_cmd(self): # (代码不变)
        cmd = self.custom_cmd_edit.text().strip()
        if not cmd: QMessageBox.warning(self, "输入错误", "自定义命令不能为空。"); return
        self._send_esp_and_log(self.esp_controller.send_command, cmd)

    # --- 调度器和设置相关函数 ---
    def on_check_connections(self): # (代码不变)
        logger.info("Checking connections...")
        self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 检查连接状态...")
        QApplication.processEvents()
        esp_ok = self.esp_controller.connect() if not self.esp_controller.connected else True
        cam_ok = self.screenshot_manager.connect_device()
        self.task_progress_text.append(f"-> ESP: {'OK' if esp_ok else '失败'} | 主摄像头ADB: {'OK' if cam_ok else '失败'}")
        self.check_esp_label.setText("已连接" if esp_ok else "未连接")
        self.check_camera_label.setText("已连接" if cam_ok else "未连接")
        self.update_esp_button_states(esp_ok) # 更新按钮状态

    def on_start_scheduler(self): # (代码不变)
        logger.info("User requested to start scheduler.")
        self.task_scheduler.start_scheduler()
        self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 任务调度器已启动")
        self.update_ui_elements()

    def on_stop_scheduler(self): # (代码不变)
        logger.info("User requested to stop scheduler.")
        self.task_scheduler.stop_scheduler()
        self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 任务调度器已停止")
        self.update_ui_elements()

    def on_apply_global_automation_settings(self): # (代码不变)
        """Applies settings from the AI/Task Control tab."""
        try:
            timeout_val = self.ai_intervention_timeout_edit.text()
            self.config["AI_INTERVENTION_TIMEOUT"] = int(timeout_val) if timeout_val else 300 # 提供默认值
            self.config["HUMAN_INTERVENTION_ALERT"] = self.human_intervention_alert_checkbox.isChecked()
            self.config["ENABLE_HUMAN_INTERVENTION"] = self.human_intervention_mode_checkbox.isChecked()
            logger.info("Applied global automation settings (in memory).")
            QMessageBox.information(self, "成功", "全局自动化设置已应用 (需手动保存到文件)。")
        except ValueError: QMessageBox.warning(self, "输入错误", "请输入有效的AI干预超时时间 (秒)。")

    # --- 设备管理相关函数 ---
    def on_add_device(self): # (代码不变, 依赖 DeviceEditDialog)
        dialog = DeviceEditDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_device_info()
            if not info: return
            if info["name"] in self.task_scheduler.devices: QMessageBox.warning(self, "错误", f"设备名称 '{info['name']}' 已存在。"); return
            new_device = Device(name=info["name"], apps=info["apps"], position=info["position"])
            if "DEVICE_CONFIGS" not in self.config: self.config["DEVICE_CONFIGS"] = {}
            device_config_entry = {k: v for k, v in info.items() if k != 'name'} # 保存除 name 外的所有信息
            device_config_entry = {k: v for k, v in device_config_entry.items() if v is not None or k == "home_screen_template_threshold"} # 清理 None
            self.config["DEVICE_CONFIGS"][info["name"]] = device_config_entry
            new_device._config = device_config_entry
            self.task_scheduler.add_device(new_device)
            logger.info(f"设备 '{info['name']}' 已添加。请记得保存设置以持久化。")
            self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 设备 '{info['name']}' 已添加 (需手动保存设置)")

    def on_remove_device(self): # (代码不变)
        selected = self.device_table.selectedItems()
        if not selected: return
        row = selected[0].row(); device_name = self.device_table.item(row, 0).text()
        reply = QMessageBox.question(self, "确认移除", f"确定要移除设备 '{device_name}' 吗？\n相关配置和任务分配可能会受影响。", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes: logger.info(f"User requested removal of device '{device_name}'"); self.task_scheduler.remove_device(device_name)

    def on_edit_device(self): # (代码不变, 依赖 DeviceEditDialog)
        selected = self.device_table.selectedItems()
        if not selected: return
        row = selected[0].row(); device_name = self.device_table.item(row, 0).text()
        device = self.task_scheduler.get_device(device_name)
        if not device: return
        dialog = DeviceEditDialog(self, device)
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_device_info()
            if not info: return
            old_name = device.name; new_name = info["name"]
            if old_name != new_name and new_name in self.task_scheduler.devices: QMessageBox.warning(self, "错误", f"设备名称 '{new_name}' 已存在。"); return
            if device.status != DeviceStatus.IDLE and old_name != new_name: logger.warning(f"正在编辑非空闲设备 '{old_name}' 的名称为 '{new_name}'。")

            device.name = new_name; device.apps = info["apps"]; device.position = info["position"]
            if "DEVICE_CONFIGS" not in self.config: self.config["DEVICE_CONFIGS"] = {}
            device_config_entry = {k: v for k, v in info.items() if k != 'name'}
            device_config_entry = {k: v for k, v in device_config_entry.items() if v is not None or k == "home_screen_template_threshold"}
            if old_name != new_name and old_name in self.config.get("DEVICE_CONFIGS", {}): del self.config["DEVICE_CONFIGS"][old_name]
            self.config["DEVICE_CONFIGS"][new_name] = device_config_entry
            device._config = device_config_entry

            # --- 更新内部名称引用 (如果名称改变) ---
            if old_name != new_name:
                logger.warning(f"设备名称已更改 ('{old_name}' -> '{new_name}')。正在更新内部引用...")
                with self.task_scheduler.lock:
                    if old_name in self.task_scheduler.devices: self.task_scheduler.devices[new_name] = self.task_scheduler.devices.pop(old_name)
                    if old_name in self.task_scheduler.running_tasks: task = self.task_scheduler.running_tasks.pop(old_name); task.assigned_device_name = new_name; self.task_scheduler.running_tasks[new_name] = task
                # TaskExecutor 相关状态的更新 (如果需要保留这些状态)
                # with QMutexLocker(self.task_executor.intervention_mutex): # 假设用这个锁保护 executor 状态
                #     if old_name in self.task_executor.last_ai_click_target: self.task_executor.last_ai_click_target[new_name] = self.task_executor.last_ai_click_target.pop(old_name)
                #     if old_name in self.task_executor.last_ai_context: self.task_executor.last_ai_context[new_name] = self.task_executor.last_ai_context.pop(old_name)
                logger.info(f"设备名称从 '{old_name}' 更改为 '{new_name}'。相关引用已更新。")

            logger.info(f"设备 '{new_name}' 已更新。请记得保存设置以持久化。")
            self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 设备 '{new_name}' 已更新 (需手动保存设置)")
            # 立即触发设备表更新
            self.update_device_table()

    def on_stop_device_task(self):  # (代码不变)
        """停止（取消）选定设备上当前正在运行的任务。"""
        selected = self.device_table.selectedItems()
        if not selected: return
        row = selected[0].row()
        # --- 增加行号有效性检查 ---
        if row >= self.device_table.rowCount():
            logger.warning("选中的行已失效。")
            return
        device_name_item = self.device_table.item(row, 0)
        if not device_name_item:
            logger.error("无法获取选中设备的名称。")
            return
        device_name = device_name_item.text()
        # --- 结束检查 ---

        task_to_cancel = None
        task_name = "未知任务"
        # 从调度器获取当前设备运行的任务
        with self.task_scheduler.lock:
            task_to_cancel = self.task_scheduler.running_tasks.get(device_name)
            # --- 修正：只有 task_to_cancel 存在时才获取名称 ---
            if task_to_cancel:
                task_name = task_to_cancel.name
            # --- 修正结束 ---

        if task_to_cancel:
            reply = QMessageBox.question(self, "确认停止",
                                         f"确定要停止/取消设备 '{device_name}' 上的任务 '{task_name}' 吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            # --- 从这里继续 ---
            if reply == QMessageBox.Yes:
                logger.info(f"用户请求取消设备 '{device_name}' 上的任务 '{task_name}' (ID: {task_to_cancel.task_id})")
                # 调用调度器的取消任务方法
                canceled = self.task_scheduler.cancel_task(task_to_cancel.task_id)
                if canceled:
                    self.task_progress_text.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 已请求取消设备 '{device_name}' 的任务 '{task_name}'。")
                else:
                    # 取消失败的原因可能比较复杂，可能是任务刚好结束，或者状态不对
                    self.task_progress_text.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 取消设备 '{device_name}' 任务 '{task_name}' 的请求失败 (可能已结束或状态异常)。")
                    QMessageBox.warning(self, "取消失败", f"无法取消任务 '{task_name}'。\n请检查任务状态或稍后重试。")
        else:
            QMessageBox.information(self, "无任务", f"设备 '{device_name}' 当前没有正在执行的任务可停止。")

    # --- 任务管理相关函数 ---
    def on_add_task(self):  # (代码不变, 依赖 TaskEditDialog)
        """添加新任务。"""
        # 传入当前设备列表，以便在对话框中选择指定设备
        dialog = TaskEditDialog(self, devices=self.task_scheduler.devices)
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_task_info()
            if not info: return

            # 创建任务对象，构造函数会自动生成 ID
            new_task = Task(
                name=info["name"], task_type=info["type"], app_name=info["app_name"],
                priority=info["priority"], use_ai_driver=info["use_ai_driver"],
                subtasks=info["subtasks"], max_retries=info["max_retries"]  # 包含重试次数
            )
            # 分配设备名（如果指定）
            if info["device_name"]: new_task.assigned_device_name = info["device_name"]

            # 添加任务到调度器（内部会处理 config 更新）
            self.task_scheduler.add_task(new_task)
            self.task_progress_text.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] 添加任务: {new_task.name} (ID: {new_task.task_id}, 设备: {new_task.assigned_device_name or '自动'}) (需手动保存设置)")
            logger.info(f"任务 '{new_task.name}' (ID: {new_task.task_id}) 已添加。请记得保存设置以持久化。")
            # self.update_task_table() # add_task 内部会触发更新

    def on_edit_task(self):  # (代码不变, 依赖 TaskEditDialog)
        """编辑选中的等待中任务。"""
        current_table = self.task_tab_widget.currentWidget()
        if current_table != self.pending_task_table:
            QMessageBox.warning(self, "操作无效", "只能编辑等待中的任务。")
            return

        selected = current_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "选择任务", "请先在等待中列表选择一个任务进行编辑。")
            return

        row = selected[0].row()
        # --- 增加行号有效性检查 ---
        if row >= current_table.rowCount():
            logger.warning("选中的任务行已失效。")
            return
        id_item = current_table.item(row, 0)  # 第一列是名称，存有 ID
        if not id_item:
            logger.error("无法获取选中任务的 Item。")
            return
        task_id = id_item.data(Qt.UserRole)  # 从 UserRole 获取 ID
        if not task_id:
            logger.error(f"无法获取选中任务 '{id_item.text()}' 的 ID。")
            QMessageBox.critical(self, "错误", "无法获取任务 ID。")
            return

        # --- 从队列中查找并临时移除任务 ---
        task_to_edit = None
        original_item = None  # 保存原始的队列元组 (priority, timestamp, task_id, task_obj)
        items_to_requeue = []
        with self.task_scheduler.lock:
            while not self.task_scheduler.task_queue.empty():
                try:
                    item = self.task_scheduler.task_queue.get_nowait()
                    prio, ts, tid, task_obj = item
                    if tid == task_id:
                        task_to_edit = task_obj
                        original_item = item  # 保存原始元组
                        break  # 找到即停止
                    else:
                        items_to_requeue.append(item)
                except queue.Empty:
                    break  # 队列空了
            # 将未处理的任务放回
            for item in items_to_requeue:
                self.task_scheduler.task_queue.put(item)

        if not task_to_edit:
            logger.error(f"无法在队列中找到要编辑的任务 ID: {task_id}")
            QMessageBox.critical(self, "错误", "在任务队列中未找到所选任务。")
            # 如果 original_item 存在，说明找到了但 task_to_edit 没赋上值？理论上不应发生
            if original_item:
                with self.task_scheduler.lock: self.task_scheduler.task_queue.put(original_item)
            return

        # --- 弹出编辑对话框 ---
        dialog = TaskEditDialog(self, task=task_to_edit, devices=self.task_scheduler.devices)
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_task_info()
            if not info:
                # 用户取消或输入错误，将原任务放回队列
                if original_item:
                    with self.task_scheduler.lock: self.task_scheduler.task_queue.put(original_item)
                # 需要手动触发一次UI更新，因为任务状态未变但可能已从队列取出又放回
                self.task_scheduler._safe_emit_ui_callback(self.task_scheduler.on_task_status_changed)
                return

            # --- 更新任务对象属性 ---
            task_to_edit.name = info["name"]
            task_to_edit.type = info["type"]
            task_to_edit.app_name = info["app_name"]
            task_to_edit.priority = info["priority"]
            task_to_edit.max_retries = info["max_retries"]
            task_to_edit.assigned_device_name = info["device_name"]
            task_to_edit.use_ai_driver = info["use_ai_driver"]
            task_to_edit.subtasks = info["subtasks"]
            # 确保状态是 PENDING，重置重试计数
            task_to_edit.status = TaskStatus.PENDING
            task_to_edit.task_stage = "PENDING"
            task_to_edit.retry_count = 0
            task_to_edit.error = None  # 清除旧错误

            # --- 将更新后的任务添加回调度器 (add_task 会处理 config 更新和 UI 触发) ---
            self.task_scheduler.add_task(task_to_edit, trigger_update=True)  # 明确触发更新

            logger.info(f"任务 '{task_to_edit.name}' (ID: {task_id}) 已更新。")
            self.task_progress_text.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] 任务 '{task_to_edit.name}' 已更新 (需手动保存设置)")

        else:
            # 用户取消编辑，将原任务放回队列
            logger.info(f"用户取消编辑任务 ID: {task_id}")
            if original_item:
                with self.task_scheduler.lock:
                    self.task_scheduler.task_queue.put(original_item)
                # 需要触发一次 UI 更新以显示回队列的任务
                self.task_scheduler._safe_emit_ui_callback(self.task_scheduler.on_task_status_changed)

    def on_cancel_selected_task(self):  # (代码不变)
        """Cancels the selected task in the currently viewed task table."""
        current_table = self.task_tab_widget.currentWidget()
        if not isinstance(current_table, QTableWidget): return
        selected = current_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "选择任务", "请先在当前列表中选择一个任务。")
            return

        row = selected[0].row()
        # --- 增加行号有效性检查 ---
        if row >= current_table.rowCount():
            logger.warning("选中的任务行已失效。")
            return
        id_item = current_table.item(row, 0)  # 第一列是名称，存有 ID
        if not id_item:
            logger.error("无法获取选中任务的 Item。")
            return
        task_id = id_item.data(Qt.UserRole)  # 从 UserRole 获取 ID
        task_name = id_item.text()  # 获取任务名称用于提示

        if task_id is None:
            logger.error(f"无法获取选中任务 '{task_name}' 的 ID。")
            QMessageBox.critical(self, "错误", "无法获取所选任务的ID。")
            return

        reply = QMessageBox.question(self, "确认取消", f"确定要取消任务 '{task_name}' (ID: {task_id}) 吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            logger.info(f"User requested cancellation for task ID {task_id} ('{task_name}')")
            canceled = self.task_scheduler.cancel_task(task_id)  # 调用调度器的取消方法
            if canceled:
                self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 已请求取消任务: {task_name}")
            else:
                self.task_progress_text.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 取消任务 '{task_name}' 失败 (可能已完成或未找到)。")
                QMessageBox.warning(self, "取消失败", f"无法取消任务 '{task_name}'。请检查任务状态。")

    # --- 设置相关函数 ---
    def on_browse_adb(self):  # (代码不变)
        filter_str = "ADB Executable (adb.exe)" if sys.platform == "win32" else "All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "选择 ADB 可执行文件", "", filter_str)
        if path:
            self.adb_path_edit.setText(path)

    def on_refresh_devices(self):  # (代码不变)
        adb_path = self.adb_path_edit.text().strip()
        if not adb_path:
            QMessageBox.warning(self, "错误", "请先设置ADB路径。")
            return
        # 临时创建一个 ScreenshotManager 来获取设备列表，避免影响当前的实例
        temp_manager = ScreenshotManager({"ADB_PATH": adb_path})
        devices = temp_manager._get_adb_device_list()
        if devices:
            # 如果当前主摄像头 ID 为空，则自动填充第一个找到的设备
            if not self.camera_device_edit.text().strip():
                self.camera_device_edit.setText(devices[0])
            QMessageBox.information(self, "可用设备", f"找到 {len(devices)} 个ADB设备:\n" + "\n".join(devices))
        else:
            QMessageBox.warning(self, "无设备", "未找到连接的ADB设备。")

    def on_test_ocr(self):  # (代码不变, 依赖 ai_analyzer 和 screenshot_manager)
        """Handles the "Test OCR Connection" button click."""
        # 更新 analyzer 的配置
        self.config["OCR_API_URL"] = self.ocr_url_edit.text().strip()
        self.ai_analyzer.ocr_api_url = self.config["OCR_API_URL"]
        if not self.ai_analyzer.ocr_api_url:
            QMessageBox.warning(self, "错误", "请输入 OCR API 地址。")
            return

        logger.info("测试 OCR 连接...")
        self.task_progress_text.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 测试 OCR 连接 ({self.ai_analyzer.ocr_api_url})...")
        QApplication.processEvents()

        logger.info("正在获取测试截图...")
        test_image = self.screenshot_manager.take_screenshot()  # 使用主设备截图

        if test_image is None:
            logger.error("OCR 测试失败：无法获取测试截图。")
            self.task_progress_text.append("-> OCR 测试失败：无法获取测试截图。请检查 ADB 和设备连接。")
            QMessageBox.warning(self, "OCR 测试失败", "无法获取测试截图。\n请确保主摄像头设备已连接，并配置正确。")
            return

        logger.info("正在调用 OCR API...")
        ocr_result = self.ai_analyzer.perform_ocr(test_image)

        if ocr_result and ocr_result.get("code") in [100, 101]:
            num_items = len(ocr_result.get("data", [])) if isinstance(ocr_result.get("data"), list) else 0
            status_text = "成功" if ocr_result["code"] == 100 else "成功（无文本）"
            message = f"OCR 测试{status_text}。\n识别到 {num_items} 个文本项。"
            logger.info(message)
            self.task_progress_text.append(f"-> {message}")
            QMessageBox.information(self, f"OCR 测试{status_text}", message)
            # 更新 OCR 显示区域和截图区域
            self._update_ocr_display(ocr_result, f"测试截图@{self.screenshot_manager.primary_device_id or '主设备'}")
            self.display_screenshot(test_image, "OCR 测试截图")  # 显示原始测试截图
        else:
            error_code = ocr_result.get("code", "未知") if ocr_result else "无响应"
            error_message = ocr_result.get("data", "连接失败或无响应") if ocr_result else "连接失败或无响应"
            logger.error(f"OCR 测试失败。代码: {error_code}, 消息: {error_message}")
            self.task_progress_text.append(f"-> OCR 测试失败。代码: {error_code}, 消息: {error_message}")
            QMessageBox.warning(self, "OCR 测试失败",
                                f"无法连接到 OCR 服务或识别失败。\n错误代码: {error_code}\n错误信息: {error_message}\n\n请检查服务状态、API地址、网络。")
            self._update_ocr_display(None, "OCR 测试失败")
            self.display_screenshot(None, "OCR 测试失败")  # 清空截图显示

    def on_test_ai_api(self):  # (代码不变, 依赖 ai_analyzer)
        # 更新 analyzer 的配置
        self.config["AI_API_URL"] = self.ai_url_edit.text().strip()
        self.config["AI_API_KEY"] = self.ai_key_edit.text().strip()  # API Key 通常不需要 strip
        self.config["AI_MODEL"] = self.ai_model_edit.text().strip()
        self.ai_analyzer.ai_api_url = self.config["AI_API_URL"]
        self.ai_analyzer.ai_api_key = self.config["AI_API_KEY"]
        self.ai_analyzer.ai_model = self.config["AI_MODEL"]

        if not self.ai_analyzer.ai_api_key or not self.ai_analyzer.ai_api_url or not self.ai_analyzer.ai_model:
            QMessageBox.warning(self, "错误", "请输入完整的 AI API 地址、密钥和模型名称。")
            return

        logger.info("Testing AI API connection...")
        self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 测试 AI API 连接...")
        QApplication.processEvents()

        try:
            test_prompt = "你好，请确认你能正常工作。简单回复即可。"
            # 调用 AI 分析器的 get_ai_decision 方法
            ai_result = self.ai_analyzer.get_ai_decision(test_prompt, [], "API Test")

            # 检查返回结果
            if ai_result and ai_result.get("decision") is not None:  # 检查 decision 是否存在且不为 None
                self.task_progress_text.append("-> AI API 测试成功。")
                QMessageBox.information(self, "AI API测试成功",
                                        f"成功连接到 AI API ({self.ai_analyzer.ai_model})。\n\n响应:\n{ai_result['decision'][:200]}...")
            else:
                # 获取更详细的错误信息
                err_msg = ai_result.get("error", "未收到有效响应") if ai_result else "未收到有效响应"
                raw_resp = ai_result.get("raw_response", "") if ai_result else ""
                full_err_msg = f"错误: {err_msg}\n原始响应 (部分): {str(raw_resp)[:200]}"
                self.task_progress_text.append(f"-> AI API 测试失败: {err_msg}。")
                QMessageBox.warning(self, "AI API测试失败",
                                    f"无法从 AI API 获取有效响应。\n{full_err_msg}\n请检查密钥、地址、模型名称和网络连接。")

        except Exception as e:
            self.task_progress_text.append(f"-> AI API 测试错误: {e}")
            logger.error(f"AI API test error: {e}", exc_info=True)
            QMessageBox.critical(self, "AI API测试错误", f"测试 AI API 时发生错误: {e}")

    def on_save_settings(self):  # (代码不变)
        """保存所有可配置设置到 config.json。"""
        logger.info("正在保存所有设置到 config.json...")
        try:
            # --- 更新配置字典中各个部分的值 ---
            self.config["ADB_PATH"] = self.adb_path_edit.text().strip()
            self.config["CAMERA_DEVICE_ID"] = self.camera_device_edit.text().strip()
            # 解析分辨率，增加错误处理
            res_match = re.match(r'(\d+)\s*[xX]\s*(\d+)', self.screenshot_res_edit.text())
            if res_match:
                self.config["SCREENSHOT_RESOLUTION"] = [int(res_match.group(1)), int(res_match.group(2))]
            else:
                raise ValueError(f"无效的默认截图分辨率格式: '{self.screenshot_res_edit.text()}' (应为 WxH)")
            crop_match = re.match(r'(\d+)\s*[xX]\s*(\d+)', self.cropped_res_edit.text())
            if crop_match:
                self.config["CROPPED_RESOLUTION"] = [int(crop_match.group(1)), int(crop_match.group(2))]
            else:
                raise ValueError(f"无效的默认裁剪分辨率格式: '{self.cropped_res_edit.text()}' (应为 WxH)")

            self.config["OCR_API_URL"] = self.ocr_url_edit.text().strip()
            self.config["OCR_CONFIDENCE_THRESHOLD"] = float(self.ocr_confidence_edit.text())
            self.config["AI_API_URL"] = self.ai_url_edit.text().strip()
            self.config["AI_API_KEY"] = self.ai_key_edit.text()  # API Key 通常保留原样
            self.config["AI_MODEL"] = self.ai_model_edit.text().strip()
            self.config["TEMPLATE_MATCHING_THRESHOLD"] = float(self.template_threshold_edit.text())
            self.config["ESP_IP"] = self.esp_ip_edit.text().strip()
            self.config["ESP_PORT"] = int(self.esp_port_edit.text())
            self.config["AI_INTERVENTION_TIMEOUT"] = int(self.ai_intervention_timeout_edit.text())
            self.config["HUMAN_INTERVENTION_ALERT"] = self.human_intervention_alert_checkbox.isChecked()
            self.config["ENABLE_HUMAN_INTERVENTION"] = self.human_intervention_mode_checkbox.isChecked()

            # --- 设备和任务配置已在编辑/添加时更新到 self.config["DEVICE_CONFIGS"] 和 self.config["USER_TASKS"] ---
            # 确保这两个 key 存在且是正确的类型
            if "DEVICE_CONFIGS" not in self.config: self.config["DEVICE_CONFIGS"] = {}
            if "USER_TASKS" not in self.config: self.config["USER_TASKS"] = []

            # --- 写入文件 ---
            backup_file = "config.json.bak"
            if os.path.exists("config.json"):
                try:
                    shutil.copy2("config.json", backup_file); logger.info(f"配置文件已备份到 {backup_file}")
                except Exception as bk_err:
                    logger.warning(f"创建配置文件备份失败: {bk_err}")

            # 写入新配置
            with open("config.json", "w", encoding='utf-8') as f:
                # 使用 default=str 处理无法直接序列化的类型 (如 Enum)
                json.dump(self.config, f, indent=4, ensure_ascii=False, sort_keys=True, default=str)

            logger.info("设置已成功保存到 config.json。");
            QMessageBox.information(self, "设置已保存", "所有设置已成功保存到 config.json。")

            # --- 重新应用部分设置到活动组件 ---
            # (保持不变)
            self.screenshot_manager.adb_path = self.config["ADB_PATH"]
            self.screenshot_manager.primary_device_id = self.config["CAMERA_DEVICE_ID"]
            self.ai_analyzer.ocr_api_url = self.config["OCR_API_URL"];
            self.ai_analyzer.ai_api_url = self.config["AI_API_URL"];
            self.ai_analyzer.ai_api_key = self.config["AI_API_KEY"];
            self.ai_analyzer.ai_model = self.config["AI_MODEL"]
            self.esp_controller.esp_ip = self.config["ESP_IP"];
            self.esp_controller.esp_port = self.config["ESP_PORT"]

        except ValueError as ve:
            QMessageBox.critical(self, "保存设置错误", f"输入值无效，请检查设置: {ve}"); logger.error(
                f"解析设置值时出错: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "保存设置错误", f"保存设置到 config.json 时发生错误: {e}"); logger.error(
                f"保存设置时出错: {e}", exc_info=True)

    def is_human_intervention_enabled(self) -> bool:  # (代码不变)
        """检查人工干预模式是否启用"""
        return self.human_intervention_mode_checkbox.isChecked()

    def on_open_template_folder(self):  # (代码不变)
        """Opens the template directory in the file explorer."""
        path = os.path.abspath(self.ai_analyzer.template_dir)
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            logger.info(f"Opened template folder: {path}")
        except Exception as e:
            logger.error(f"Could not open template folder '{path}': {e}"); QMessageBox.warning(self, "无法打开",
                                                                                               f"无法自动打开文件夹:\n{path}\n\n错误: {e}")

    def on_manual_screenshot(self):  # (代码不变)
        """获取主摄像头设备的截图并显示（带标注）。"""
        logger.info("请求手动截图...");
        self.task_progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 手动获取主摄像头截图...");
        QApplication.processEvents()
        screenshot = self.screenshot_manager.take_screenshot(None)  # 传入 None 使用主设备
        if screenshot is not None:
            self.task_progress_text.append("-> 手动截图成功。");
            annotated_image = screenshot.copy()
            device_id_str = self.screenshot_manager.primary_device_id or '未知主设备'
            # 运行 OCR 并绘制框
            ocr_result = self.ai_analyzer.perform_ocr(annotated_image)
            self._update_ocr_display(ocr_result, f"手动截图@{device_id_str}")
            if ocr_result and ocr_result.get("code") == 100 and "data" in ocr_result:
                for item in ocr_result["data"]:
                    box = item.get("box");
                    if box and len(box) == 4:
                        try:
                            pts = np.array(box, np.int32).reshape((-1, 1, 2)); cv2.polylines(annotated_image, [pts],
                                                                                             isClosed=True,
                                                                                             color=(0, 255, 0),
                                                                                             thickness=1)
                        except:
                            pass  # 忽略绘制错误
            self.display_screenshot(annotated_image, f"手动@{device_id_str}")  # 显示带标注的图
        else:
            self.task_progress_text.append("-> 手动截图失败。"); QMessageBox.warning(self, "截图失败",
                                                                                    "无法获取主摄像头截图。"); self.display_screenshot(
                None, "截图失败")

    def check_alerts(self):  # (代码不变)
        """Checks for conditions requiring user alerts."""
        if not self.config.get("HUMAN_INTERVENTION_ALERT", True): return
        alert_needed = False;
        alert_message = ""
        with self.task_scheduler.lock:
            for device in self.task_scheduler.devices.values():
                if device.status == DeviceStatus.ERROR: alert_needed = True; alert_message = f"设备 '{device.name}' 处于错误状态！"; break
        if alert_needed: logger.warning(f"ALERT: {alert_message}"); self.task_progress_text.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ {alert_message} 请检查！"); QApplication.beep()

    def closeEvent(self, event):
        """处理窗口关闭事件，确保调试会话状态被妥善处理。"""
        # 恢复可能被临时修改的设备配置
        if self.debugging_device_name and self._original_device_coord_map_backup:
            device_to_restore = self.task_scheduler.devices.get(self.debugging_device_name)
            if device_to_restore:
                self._log_coord_debug(f"窗口关闭前，恢复设备 '{self.debugging_device_name}' 的原始坐标映射: {self._original_device_coord_map_backup}")
                device_to_restore._config["COORDINATE_MAP"] = self._original_device_coord_map_backup
            self._original_device_coord_map_backup = None
            self.debugging_device_name = None

        # ... (原有 closeEvent 逻辑不变) ...
        logger.info("收到关闭请求...")
        reply = QMessageBox.question(self, '确认退出', "是否要在退出前保存当前配置?",
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
        if reply == QMessageBox.Cancel:
            logger.info("关闭操作已取消。"); event.ignore(); return
        elif reply == QMessageBox.Yes:
            logger.info("正在退出前保存配置..."); self.on_save_settings()

        logger.info("正在停止任务调度器...");
        self.task_scheduler.stop_scheduler()
        logger.info("正在断开 ESP 连接...");
        self.esp_controller.disconnect()
        logger.info("关闭完成。")
        event.accept()
# --- Logger Helper ---
class QTextEditLogger(logging.Handler, QObject):
    log_received = pyqtSignal(str)
    def __init__(self, text_widget: QTextEdit):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.widget = text_widget
        self.widget.setReadOnly(True)
        self.widget.document().setMaximumBlockCount(5000) # 增加行数限制
        self.log_received.connect(self.widget.append, Qt.QueuedConnection) # 确保在主线程追加

    def emit(self, record):
        try:
            msg = self.format(record)
            # 检查消息是否过长，过长可能导致UI卡顿，可以选择截断
            # if len(msg) > 1000: msg = msg[:1000] + "..."
            self.log_received.emit(msg)
        except Exception as e:
            print(f"ERROR in QTextEditLogger.emit: {e}")
            self.handleError(record)

# --- Main Execution (保持不变) ---
def main():
    # 确保目录存在
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    try:
        # 设置 Qt 属性
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

        # 创建主窗口，直接使用在顶层加载好的 CONFIG
        logger.info("Creating MainUI instance...")
        # 这里的 CONFIG 应该是脚本顶层加载和合并后的结果
        window = MainUI(CONFIG)
        logger.info("MainUI instance created. Showing window...")
        window.show()
        logger.info("MainUI window shown.")

        logger.info("Starting application event loop (app.exec_())...")
        exit_code = app.exec_()
        logger.info(f"Application event loop finished with exit code {exit_code}.")
        sys.exit(exit_code)

    except ImportError as e:
        # 处理可能的库缺失问题
        print(f"发生致命错误：缺少必要的库 - {e}")
        logger.critical(f"发生致命错误：缺少必要的库 - {e}", exc_info=True)
        try:
             # 尝试用 PyQt 显示错误，如果 QApplication 启动失败则会跳过
             QMessageBox.critical(None, "依赖错误", f"无法启动应用，缺少必要的库:\n{e}\n\n请确保已安装所有依赖项 (例如: pip install PyQt5 requests numpy opencv-python Pillow)")
        except: pass
        sys.exit(1)
    except Exception as e:
        # 捕获其他启动阶段的致命错误
        print(f"发生致命错误导致应用无法启动: {e}")
        logger.critical(f"发生致命错误导致应用无法启动: {e}", exc_info=True)
        try:
            import traceback
            tb_info = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            # 尝试用 PyQt 显示错误
            QMessageBox.critical(None, "启动错误", f"应用程序启动失败:\n\n{e}\n\n详细信息请查看日志文件。\n\nTraceback:\n{tb_info[:1000]}...")
        except: pass # 如果连 QMessageBox 都失败也没办法了
        sys.exit(1)

# --- 脚本入口 (保持不变) ---
if __name__ == "__main__":
    # 配置全局异常处理器
    def excepthook(exc_type, exc_value, exc_tb):
        import traceback
        tb_info = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        # 记录到日志文件
        logger.critical(f"Unhandled exception caught by excepthook:\n{tb_info}")
        # 尝试在 UI 日志中显示（如果 UI 存在且 QTextEditLogger 正常工作）
        try: logger.critical(f"UI LOG (Unhandled exception): {exc_value}")
        except Exception as ui_log_err: print(f"Error logging exception to UI: {ui_log_err}")
        # 仍然调用默认的处理器，以便在控制台打印
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = excepthook # 应用钩子
    main()