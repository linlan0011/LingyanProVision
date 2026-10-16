# Lingyan Pro Vision

当前项目用于安全区轮廓识别、角点测距、地图角点身份判断，以及独立视觉绝对位姿解算。默认相机为 `usb0bda3035_1080p`，原图1920×1080。

## 启动

首次创建环境：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
