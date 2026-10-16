# Lingyan Pro Vision

基于 USB 0BDA:3035 相机与 OpenCV 的惯性导航视觉校正方案（静态验证）。项目实现安全区轮廓识别、角点测距、地图角点身份判断，以及独立视觉绝对位姿解算。默认相机为 `usb0bda3035_1080p`，原图1920×1080。

当前惯导先验由地图UI模拟输入 `(x, y, 航向角)`；尚未接入真实IMU或惯导通信，也未实现运动中的时间同步、状态融合和控制闭环。视觉解算输出独立的绝对位置及航向，用于后续惯导校正方案的静态验证。

## 相机型号与采集参数

| 项目 | 当前配置 |
|---|---|
| 系统设备名称 | USB Camera |
| 硬件标识 | USB `VID_0BDA & PID_3035` |
| 项目相机ID | `usb0bda3035_1080p` |
| 厂商商业型号、传感器、镜头型号 | 现有资料未确认；硬件ID不能替代这些型号信息 |
| 采集尺寸 | 1920×1080原图 |
| 请求采集帧率 | 30 FPS；识别处理FPS以预览实测为准 |
| Windows取流后端 | Media Foundation / OpenCV `CAP_MSMF` |
| 压缩源模式 | 驱动记录为1080p MJPG 30 FPS；MSMF输出解码图像，配置不强设FOURCC |
| 图像方向 | 不旋转、不镜像；预览缩放不改变标定原图坐标 |
| 镜头模型 | OpenCV针孔模型 `opencv_pinhole`，5项畸变系数 |
| 默认摄像头序号 | 0，可用 `--index` 覆盖；序号不代表硬件身份 |
| 焦点、自动对焦、变焦 | 驱动未提供可用控制；保持镜头机械位置不变 |

相机驱动控制初值如下，数值均按当前驱动单位解释：

| 控制项 | 值/模式 | 控制项 | 值/模式 |
|---|---|---|---|
| 曝光 | 手动 −5 | 白平衡 | 自动，初始控制值6500，非锁定色温 |
| 亮度 | 0 | 对比度 | 40 |
| 色调 | 0 | 饱和度 | 66 |
| 锐度 | 50 | 伽马 | 350 |
| 背光补偿 | 0 | 增益 | 64 |

程序通过Windows驱动接口核对设备并应用、回读这些控制设置。完整配置见[相机注册表](camera_profiles.json)及[USB相机说明](doc/USB_0BDA_3035_1080p标定配置.md)。已有 `lingyan`（640×480）、`seekfree`（采集768×1024，旋转后1024×768）和 `sp3220l_1080p`（1920×1080，请求25 FPS）配置分别管理，不能混用内外参；SP3220L的硬件ID是 `1871:0501`，与默认相机不同。

## 库、依赖与验证环境

当前验证环境为 **Windows 11 x64、Python 3.12.14**。下面的“已验证版本”来自2026-10-16本机已安装环境，并不表示其他版本已经验证：

| 库/组件 | 安装约束 | 已验证版本 | 用途 |
|---|---|---|---|
| NumPy | `>=2.0,<3` | 2.5.3 | 数组、线性代数、直线拟合与位姿计算 |
| OpenCV Python | `opencv-python>=4.10,<6` | 5.0.0.93 | 相机采集、HSV分割、轮廓、标定和图像界面 |
| ReportLab | `>=4.4,<5` | 4.5.1 | 标定棋盘格PDF生成 |
| Pillow | 由ReportLab依赖解析 | 12.3.0 | 图像及PDF工具依赖 |
| charset-normalizer | 由ReportLab依赖解析 | 3.5.1 | ReportLab文本编码依赖 |
| Tkinter / Tcl-Tk | Python安装组件 | 随当前Python环境 | 地图选点与参数UI |
| Windows PowerShell / .NET | Windows系统组件 | 系统自带 | DirectShow设备身份和相机控制 |

`argparse`、`json`、`pathlib`、`dataclasses`、`subprocess`、`ctypes`、`unittest`等均为Python标准库，不需要单独pip安装。当前算法不依赖PyTorch、CUDA、深度学习模型权重或ROS。

[requirements.txt](requirements.txt)提供允许版本范围，[requirements-lock.txt](requirements-lock.txt)记录本次验证使用的5个第三方包版本。完整程序需要带GUI支持的 `opencv-python` 和可用Tkinter，不能直接使用 `opencv-python-headless` 替代。Windows驱动控制及启动脚本按Windows实现，其他平台尚未完成适配验证。

## 启动

首次创建环境：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

需要按本次已验证包版本安装时，用下面命令替代上述依赖安装命令：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import cv2, numpy, tkinter, reportlab; print('dependencies OK')"
```

安装Python时保留Tcl/Tk组件，并在Windows隐私设置中允许桌面应用访问摄像头。运行VS Code调试项还需要安装Microsoft Python与Python Debugger扩展。

双击 `start_app.cmd`，或运行：

```powershell
.\.venv\Scripts\python.exe run_app.py
```

VS Code选择 **Run full app (pose UI + live camera)** 后按F5；Ctrl+Shift+B也会启动完整程序。默认同时打开地图位姿UI和相机预览，启用Z=0地面测距、自动场景预设和视觉位姿解算。

- `--index 1`：更换摄像头索引。
- `--scene start-facing / front-facing / general`：手动选择侧面、正面、通用预设。
- `--camera <ID>`：选择独立相机配置；`python -m camera_profiles list`查看已登记配置。
- `--no-ranging`：只识别；`--no-visual-pose`：保留测距，关闭视觉位姿解算。
- 预览按 **P** 打开参数窗口，**S** 保存原图、预览及测距报告，**Q/Esc** 退出。
- 关闭地图窗口或相机窗口会结束本次完整程序；日志在 `output/logs/`。

每个相机ID固定设备、分辨率、旋转和镜头模型，使用自己目录下的内外参。已有LINGYAN和SeekFree预设仍可显式选择；它们不保证具有有效的测距外参。

## 当前识别与定位流程

1. 分割紫色区域，检查外侧白色场地及中性色核心，筛选有效轮廓片段。
2. 合并同边碎段、拟合直线，由两条支持边求交生成直接或遮挡补偿角点。三种场景共用此检测器，仅颜色与亮度预设不同。
3. 对所有标出点做Z=0几何测距，选水平距离最近点，再检查定位量程，默认1000 mm。该点失败时不换第二近点。
4. 保留输入位置默认200 mm误差圆、固定输入航向、标定视野和测距上限的安全区预筛：某区至少一地图角可能可见，就保留该区。
5. 沿最近角点的真实支持轮廓投影到车体地面，拟合两条有向边。用输入航向匹配唯一地图角点，默认方向、正交和双边航向一致性容差均15°。
6. 身份唯一后，以地图边方向减去视觉边方向反算视觉航向，再由地图角点及车体系角点反算绝对XY。输入位姿文件不会被校正结果覆盖。

地图坐标原点在左下，X向右、Y向上；航向0°朝+X、90°朝+Y。车体系X向前、Y向左、Z向上。安全区编号和外框坐标在 `safety_templates.json`。

蓝线是分割轮廓，绿线是有效边界观测，红线/L#是拟合边组，黄点是直接角点，紫红点是补偿角点，青色圆环是最近点。地图红箭头表示输入位姿，青箭头表示当前有效的视觉校正位姿。

当前按帧判定。支持边配对变动或门限失败会清除当前身份；本次文件清理没有加入时序稳定或改变角点取舍规则。

## 项目结构

| 路径 | 用途 |
|---|---|
| `run_app.py`、`__main__.py`、`start_app.cmd` | 完整程序入口 |
| `purple_camera_segmentation.py` | 相机主循环、分割和场景参数 |
| `boundary_geometry.py` | 有效边、拟合线及直接/补偿角点 |
| `corner_ranging.py` | 标定投影与车体坐标测距 |
| `nearest_corner_pose.py` | 最近角点身份及视觉绝对位姿 |
| `ground_corner_association.py` | 安全区标定视野预筛 |
| `safety_identity.py`、`safety_templates.json` | 地图角点定义、观察方向预设及结果数据结构 |
| `pose_map_gui.py`、`ground_corner_map.py` | 地图输入及视觉反馈 |
| `pose_io.py`、`pose_association.py` | 原子位姿文件和逐帧回传 |
| `camera_profiles.py/json`、`camera_controls.py`、`camera_orientation.py` | 相机配置、控制及方向 |
| `field_zones.py/json` | 地图观察分区及预设选择 |
| `preview_display.py` | 预览、参数窗口与角点文字 |
| `calibration/` | 内参、安装外参、人工采点与独立误差检验工具 |
| `test_*.py` | 现行功能回归测试 |
| `doc/` | 当前规则与标定说明 |
| `照片/field_map.png` | 地图UI必需资源 |
| `selected_pose.json` | 本机实时输入，自动生成，不提交 |
| `照片/`、`calibration/data/` | 本机原图与采点数据，不提交 |
| `output/` | 本机日志、报告、诊断图和打印棋盘格，不提交 |
| `.venv/`、`__pycache__/`、`tmp/`、`outputs/` | 环境、缓存和临时产物，不提交 |

版本库保留相机的 `calibration/results/<ID>/intrinsics.json` 和 `mount_pose.json`；原始采图、验证报告、历史备份和预览保持本机数据。停止Git跟踪不删除本机环境或输入位姿。

## 标定与误差检验

`start_calibration_points.cmd` 使用当前USB相机采集安装标定点；`start_geometry_validation.cmd` 用同一相机采集独立误差检验点。也可用VS Code中 **Calibration: ... selected camera ...** 任务选择相机ID。

```powershell
.\.venv\Scripts\python.exe -m calibration.collect_mount_points --camera usb0bda3035_1080p
.\.venv\Scripts\python.exe -m calibration.fit_mount_pose --camera usb0bda3035_1080p
.\.venv\Scripts\python.exe -m calibration.validate_ranging --camera usb0bda3035_1080p
```

安装外参绑定内参SHA256；换镜头、采集模式或相机安装姿态后，需要使用相应的重新标定结果。

当前提交内参的数值摘要：`fx=961.2179 px`、`fy=959.5245 px`、`cx=951.4811 px`、`cy=543.8929 px`；畸变顺序为 `k1,k2,p1,p2,k3`，完整系数以[内参文件](calibration/results/usb0bda3035_1080p/intrinsics.json)为准。该次内参使用9×6内角点、实测格长19.2 mm，重投影RMS约0.7191 px；这是标定拟合指标，不是车体定位精度。

当前拟合的相机车体系位置约为 `(17.085, −6.392, 174.277) mm`，旋转矩阵及内参哈希绑定见[安装外参文件](calibration/results/usb0bda3035_1080p/mount_pose.json)。这些参数只适用于记录中的相机模式及安装状态，另一台同硬件ID相机也需要独立标定。

## 规则与文档

- [边界、直线与角点规则](doc/安全区轮廓直线与角点重构规则.md)
- [侧面模式当前规则](doc/安全区侧面参数及规则.md)
- [最近角点身份与视觉绝对位姿](doc/最近角点身份与视觉绝对位姿.md)
- [当前USB相机标定配置](doc/USB_0BDA_3035_1080p标定配置.md)
- [实时角点测距](doc/实时角点测距.md)
- [多相机标定与切换](doc/多相机独立标定与切换.md)
- [多点测距误差检验](doc/多点几何测距误差检验.md)
- [本次文件整理说明](doc/项目结构与清理说明.md)

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -p "test_*.py"
```

VS Code也可运行 **Test: all project checks**。测试使用合成图像或替代相机，不启动物理摄像头。离线回放和测试通过不等于实地定位精度证明。
