# OCR驱动的桌面自动化工具

一个基于Python的应用程序，它使用屏幕捕获、先进的图像处理、OCR和AI来自动化控制通过三轴机械臂连接的物理设备上的复杂任务。

## 简介 (Overview)

本项目是一个为解决物理设备（如智能手机）缺乏传统API接口而设计的自动化工具。它的工作流程如下：
1.  **屏幕捕获**: 通过ADB捕获连接设备的屏幕。
2.  **图像处理**: 对图像进行处理，显著增强其清晰度，以提高OCR识别准确率。
3.  **元素识别**: 使用OCR服务和模板匹配识别屏幕上的文本和UI元素。
4.  **执行操作**: 通过一个定制的三轴硬件控制器（由ESP微控制器驱动）执行点击、滑动等物理操作。

其主要目标是在无法进行直接软件集成的设备上，自动执行重复性任务、进行自动化测试或控制应用程序。

## 功能亮点 (Features)

-   **🤖 强大的任务自动化引擎**:
    -   **AI驱动模式**: 利用大语言模型（LLM）分析屏幕内容，并决定实现目标的最佳下一步操作。
    -   **子任务模式**: 通过预定义的、精确的步骤序列（如 `find_and_click_text`, `wait`, `loop`, `check_template_exists`）来执行复杂的工作流。
-   **🖼️ 先进的图像增强管道**:
    -   包含CLAHE（对比度限制自适应直方图均衡）等技术，可显著提升屏幕截图质量，从而获得更高的OCR准确率，尤其是在低质量显示屏上。
-   **🎯 精准的硬件控制**:
    -   与基于ESP的三轴机械臂通信，以高精度执行物理交互，如点击、长按和滑动。
    -   支持为不同设备和物理设置配置坐标映射。
-   **✅ 稳健的任务调度器**:
    -   一个多线程调度器管理着一个任务优先队列。
    -   处理任务分配、错误处理和失败重试。
-   **🖥️ 功能完善的图形用户界面**:
    -   基于PyQt5构建，用于管理设备、任务和系统设置。
    -   通过实时屏幕视图、OCR结果和详细日志提供即时反馈。
-   **🧩 模板匹配**:
    -   使用OpenCV来视觉定位和交互那些无法单靠文本识别的图标和UI元素。

## 可视化展示 (Demo)

### 图像增强效果对比

这张图展示了图像处理管道如何改善低质量的屏幕截图，使其能被OCR引擎清晰地识别。

**[待办]**: 请您亲自创建这张对比图并替换此处的占位符。例如:
`![图像增强效果](assets/enhancement_comparison.png)`

### 系统工作流程图

此图表展示了系统内部的数据流和操作逻辑。

**[待办]**: 请您亲自绘制此流程图并替换此处的占位符。例如:
`![系统工作流程](assets/workflow.png)`
*流程: `[截取屏幕] -> [图像预处理] -> [OCR与模板匹配] -> [AI或子任务逻辑] -> [执行硬件控制指令]`*

### GIF动态操作演示

一个动态演示，展示工具自动化完成任务的全过程。

**[待办]**: 请您亲自录制此GIF动图并替换此处的占位符。例如:
`![GIF操作演示](assets/demo.gif)`

## 技术栈 (Tech Stack)

-   **语言**: Python 3
-   **GUI**: PyQt5
-   **核心库**:
    -   OpenCV (`opencv-python`)
    -   Pillow
    -   NumPy
    -   Requests
-   **硬件**: ESP-12F 或类似微控制器，用于三轴控制。
-   **外部服务**:
    -   一个通过ADB连接的设备用于屏幕捕获。
    -   一个OCR服务 (如 Umi-OCR) 的API端点。
    -   一个AI/LLM服务 (如 灵一万物) 的API端点。

## 安装与运行 (Getting Started)

请遵循以下步骤来启动和运行项目。

### 1. 环境要求

-   Python 3.8+
-   Android调试桥 (ADB)
-   一个基于ESP并刷入相应固件的三轴硬件控制器。
-   一个开启了USB调试模式的安卓设备。

### 2. 克隆项目

```bash
git clone https://github.com/Feibafly2/autoControlBasedOnThreeAxisMechine.git
cd autoControlBasedOnThreeAxisMechine
```

### 3. 安装依赖

推荐使用虚拟环境。
```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境 (Windows下为 `venv\Scripts\activate`)
source venv/bin/activate
# 安装依赖
pip install -r requirements.txt
```

### 4. 配置应用

在项目根目录中，通过复制以下模板创建一个 `config.json` 文件。

**`config.json` 模板:**
```json
{
    "ESP_IP": "192.168.101.11",
    "ESP_PORT": 8080,
    "OCR_API_URL": "http://127.0.0.1:1224/api/ocr",
    "AI_API_URL": "https://api.lingyiwanwu.com/v1/chat/completions",
    "AI_API_KEY": "YOUR_AI_API_KEY_HERE",
    "AI_MODEL": "yi-large-rag",
    "ADB_PATH": "adb",
    "CAMERA_DEVICE_ID": "YOUR_ADB_DEVICE_ID",
    "DEVICE_CONFIGS": {
        "My_Phone_1": {
            "SCREENSHOT_RESOLUTION": [1080, 1920],
            "CROPPED_RESOLUTION": [1080, 1440],
            "HOME_SCREEN_ANCHOR_TEXTS": ["电话", "短信", "相机"],
            "HOME_SCREEN_MIN_ANCHORS": 2,
            "machine_origin_x": 0,
            "machine_origin_y": 0
        }
    },
    "USER_TASKS": []
}
```

**关键配置项说明:**
-   `ESP_IP`, `ESP_PORT`: 你的硬件控制器的IP地址和端口。
-   `OCR_API_URL`: 你的OCR服务的端点地址。
-   `AI_API_KEY`: 你的AI服务的API密钥。
-   `CAMERA_DEVICE_ID`: 你的安卓设备的ID，可通过运行 `adb devices` 命令查看。
-   `DEVICE_CONFIGS`: 在这里定义每个物理设备。键名 (`My_Phone_1`) 是你为设备指定的唯一名称。

### 5. 运行项目

```bash
python main.py
```