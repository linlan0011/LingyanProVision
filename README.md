# Lingyan Pro Vision

This is a standalone VS Code project for testing the purple safety-zone detector.
The current camera choice is the original LINGYAN Pro at 640 x 480 without
rotation. The SeekFree 彩钻风 Pro remains an explicit alternative profile.
Press `q` or `Esc` to stop the preview.

## 一次启动完整程序（默认入口）

在 VS Code 打开这个文件夹后，运行/调试选择第一项
**Run full app (pose UI + live camera)**，按 F5 或 Ctrl+F5。
Ctrl+Shift+B 的默认任务也会启动整套程序。
如果 VS Code 记住了以前的单窗口启动项，首次需手动切换到这一项。

也可以双击 `start_app.cmd`，或在项目目录运行：

```powershell
.venv\Scripts\python.exe .
```

统一入口为 `run_app.py`：同时打开地图选点 UI 和实时相机画面；
启动时恢复上次位姿，两个窗口共享同一份绝对路径位姿文件。
默认开启实时输出和自动场景选择，选点/拖动后识别结果更新在相机画面中。
关闭地图窗口，或在相机窗口按 q/Esc、关闭相机窗口，都会结束整套程序。
相机故障会提示日志位置，日志保存在 `output/logs/`。
更换摄像头序号可运行 `run_app.py --index 1`；SeekFree 使用 `--camera seekfree`。

## 场地地图位姿选择器

运行 `pose_map_gui.py`，或在 VS Code 中选择 **Field map pose selector**。
程序会自动加载 `照片/field_map.png`：

```powershell
.venv\Scripts\python.exe pose_map_gui.py
```

界面默认使用当前救援场地的坐标约定：左下角为原点，X/Y 范围均为
`0..3000 mm`，`0°` 指向 `+X`，`90°` 指向 `+Y`，逆时针为正。
地图与实际范围不一致时，可在右侧修改四个坐标边界并点击“应用坐标范围”。
当前图片的场地外框按原图像素 `(左=27, 上=22, 右=749, 下=743)` 标定。
图片外围的尺寸箭头和文字保留可见，但光标只有在场地外框内才显示场地坐标。
如果换用其他地图，程序默认将整张图片作为场地，也可在右侧修改图片外框像素位置。

- 光标悬停时，底部实时显示对应的 X/Y 坐标。
- 单击地图选择位置；从该点按住左键拖向目标方向可设置航向角。
- 手动修改 X、Y、航向角后，地图上的点和箭头会实时更新。
- 默认勾选“实时输出位姿”：单击选点自动同步，拖动航向时约每 50 ms
  输出最新位姿，松开立即提交；数值输入停顿约 200 ms 后同步。
  位姿原子写入 `selected_pose.json`，视觉程序每帧读取更新。
  可关闭实时输出，改用“立即输出给识别程序”按钮。
  非法数字、越界坐标不会输出；此时视觉端仍使用上次成功提交的位姿。

### 识别情境分区

点击主窗口的“识别情境分区管理…”，再点击“开始在地图上框选”，在地图内
按住左键拖出矩形。管理窗口会自动填写 `X 最小/最大` 和 `Y 最小/最大`，
可手动微调。填写名称、参考航向角和航向容差，点击“保存分区”。
保存的分区会在地图上显示范围、名称和航向箭头；在列表中选中分区可修改
名称、坐标与角度，也可删除。数据自动保存到 `field_zones.json`，重启后恢复。

每个分区是轴对齐矩形；角度描述车辆进入该区时的参考航向，不表示矩形本身的旋转。
坐标边界和角度容差均包含端点。允许不同航向的分区重叠；一辆车可能同时匹配多个分区。
例如中心角 `179°`、容差 `5°` 的分区也能匹配航向 `-179°`。

`purple_camera_segmentation.py` 默认轮询这个文件，并在左侧相机画面顶部显示
`INPUT POSE`。它还会读取 `field_zones.json`，用当前位姿匹配分区；
相机画面显示匹配分区的序号，终端在匹配结果变化时输出中文分区名称。
分区名称不决定识别规则；每个分区新增 `safety_zone_id` 和 `view_side`，
在“识别情境分区管理”中可选择。`--scene auto` 根据位姿模板选择正面或侧面规则。
也可以分别指定路径：

```powershell
.venv\Scripts\python.exe pose_map_gui.py --output runtime\pose.json --zones runtime\zones.json
.venv\Scripts\python.exe purple_camera_segmentation.py --pose-file runtime\pose.json --zones-file runtime\zones.json
```

交换格式为 `lingyan.pose.v1` JSON，核心字段是 `x_mm`、`y_mm` 和
`heading_deg`。找不到地图图片时，界面会显示一张等比例坐标网格。
分区文件使用 `lingyan.field_zones.v1` 格式；每项包含名称、`x_min_mm`、
`x_max_mm`、`y_min_mm`、`y_max_mm`、`heading_deg` 和
`heading_tolerance_deg`。

## 安全区和角点编号识别

上方安全区为 **1**，下方为 **2**；各区从地图左上角起顺时针编号：
`.1=西北角 NW`、`.2=东北角 NE`、`.3=东南角 SE`、`.4=西南角 SW`。
编号基于外框，固定在地图上，不随相机画面的左右变化。默认几何为：
安全区 1：X=1170..1830、Y=2640..3000 mm；
安全区 2：X=1170..1830、Y=0..360 mm。几何、编号映射与阈值统一保存在
`safety_templates.json`，UI 和视觉程序读取同一个文件。

需要分别调试时，也可单独运行两个程序（正常使用推荐上方统一入口）：

```powershell
.venv\Scripts\python.exe pose_map_gui.py
.venv\Scripts\python.exe purple_camera_segmentation.py --camera lingyan --scene auto
```

VS Code 新增 **Safety identity (LINGYAN auto pose templates)** 调试项和相应任务。
原有正面/侧面启动项仍手动固定检测模式；若与位姿模板冲突，编号显示待定。
UI 地图显示两个安全区和全部八个角点；“安全区编号与模板规则…”展示坐标和规则。
默认选点后自动输出，开启实时输出时无需点击确认。UI 底部显示的是位姿候选，
不是相机已经确认的识别结果。

观察区域按半场划分：`Y >= 1500 mm` 只归安全区 1，`Y < 1500 mm` 只归安全区 2。
每个半场都有五个位置区：西侧、中央正面、东侧，以及两个向中央安全区看的斜视区。
分区只决定当前安全区和视角类别；其后的模板再以该视角独立的航向容差选择检测器。

| 观察区 | 安全区 | 检测器 | 角点编号规则 |
| --- | --- | --- | --- |
| 1-W、1-E、2-W、2-E | 对应半场 | `start-facing` 侧面 | 固定边方向模板 |
| 1-S、2-N | 对应半场 | `front-facing` 正面 | 固定边方向模板 |
| 1-SW、1-SE、2-NW、2-NE | 对应半场 | `general` 通用斜视 | 四点齐全时按画面左/上/右/下映射；缺角时使用位姿投影 |

不再要求位置的横向投影落在安全区外框内。比如位于安全区 1 东南方向、
车头朝向西北的位姿仍会命中 `1-SE`，选择通用斜视检测，而不是直接显示无候选。
斜视画面中同一分割区域的四个角都有效时，优先使用实拍确定的画面极值映射。
例如 `1-SW` 的左、上、右、下角依次为 `1.1、1.2、1.3、1.4`。缺角时，按四个地图角
相对车头的方位投影到画面横坐标，并在容差内选择唯一编号；水平视场与匹配容差在
`safety_templates.json` 的 `diagonal_pose_projection` 中可实拍校正。

角点规则使用从角点沿**有效边界实际延伸**的方向，图像右为 0°、上为 90°：

| 模板位置 | 横向边（容差 ±20°） | 纵深边（容差 ±35°） |
| --- | --- | --- |
| 近左 | 0°，向右 | 45°，右上 |
| 近右 | 180°，向左 | 135°，左上 |
| 远左 | 0°，向右 | -135°，左下 |
| 远右 | 180°，向左 | -45°，右下 |

支持线必须来自检测器的有效边界，靠近角点端部，不能拿穿过角点的无限直线
充当边界射线。640 像素宽时，线距离上限 7 px，可见点端部距离上限 18 px，
推断点为 64 px，最短边为 8 px；其他图像宽度同比缩放。直接检测角点使用
所属分量的边，正面遮挡推断角点允许使用相邻已接受分量的支撑边。

只出现一个有效角点也可以关联，但需要两条合适的支撑边。边不足、多个编号
都能匹配、两个检测点争用同一个编号时均显示 `?`。每帧重新匹配，不沿用上帧标签。
相机和二值图的角点旁显示如 `2.2 NE`；底部 `POSE candidates` 是位姿候选，
`SAFETY ZONE / CORNERS` 才是当前视觉模板匹配结果。推断点另标 `inferred`。
模板变化自动重读；无位姿、模板损坏、非默认图像旋转或灰度帧不输出确认编号。

当前阈值是工程初始值，尚需覆盖实际位置、航向、遮挡、光照的实拍验证。
这一步实现身份关联，不执行惯导位置修正。内外框、高度和相机安装必须与模板一致。

验证命令：

```powershell
.venv\Scripts\python.exe -m unittest test_pose_tools test_calibration_tools test_safety_identity
```

测试覆盖十个观察分区、上下半场边界、八个角点映射、单角点、缺边、歧义、重复编号、绑定保存，
并以合成相机帧通过实际主循环验证位姿文件→检测模式→分割→角点→编号。
`照片/synthetic_*.png` 是合成测试结果，不代表实拍准确率。

## Run in VS Code

1. Open this folder in VS Code.
2. Select the interpreter at `.venv\Scripts\python.exe` if VS Code has not
   selected it automatically.
3. Select **Run full app (pose UI + live camera)** and press `F5`,
   or run the default build task with `Ctrl+Shift+B`.

The default command is:

```powershell
.venv\Scripts\python.exe run_app.py
```

`--camera lingyan` uses the earlier 640 × 480, unrotated capture and the
original scene rules. Both profiles have separate VS Code front (`front-facing`)
and side (`start-facing`) run/debug entries. The optional `--camera seekfree`
profile captures 768 × 1024 and rotates to 1024 × 768 before recognition.
On the sample safety-zone view, the SeekFree front profile uses a finer contour
approximation and accepts corners up to 160°; the older profile values remain
unchanged. The sample results are
[SeekFree front](照片/seekfree_front_adaptation_preview.png) (three visible corners;
the left near corner is obscured) and
[SeekFree side](照片/seekfree_side_adaptation_preview.png) (four visible corners).

`--scene front-facing` is the current test for the aligned frontal view of a
safety zone. It accepts visible near-edge corners of the shallow trapezoid
without requiring a complete quadrilateral; under occlusion or dark lighting,
only one corner may pass the two-sided white-field check. The completed launch-side scene
is preserved as `--scene start-facing`; see [安全区侧面参数及规则.md](安全区侧面参数及规则.md)
for its camera values and rules. `--scene general` retains the wider corner
rules and green-line intersection inference. The default `--scene auto` selects
front/side rules from the applicable pose template. Explicit scene flags retain
manual mode. Auto mode keeps each scene's trackbar settings independently.

The scenes have separate parameter presets. `start-facing` keeps its verified
`white S max=179`; `front-facing` starts at `white S max=80` to keep the blue
safety-zone interior out of the white-field mask. Changes to the frontal
corner rules do not alter the saved side-view preset.
In the frontal scene, two nearby candidates on the same left or right end
are grouped symmetrically and only the lower one is kept. At 640 × 480 the
frontal grouping range is about 51 px horizontally and 64 px vertically;
the completed side-view mode retains its own 32 px horizontal range.
When a visible contour turn and a valid inferred corner occupy the same
frontal corner region, the inferred corner is retained because the physical
corner is occluded. An inferred point made by extending past an already
confirmed near-edge corner is rejected before this comparison.
Every frontal corner region produces at most one result across all accepted
purple components. If the region contains one or more valid inferred points,
only the lowest inferred point is kept and all nearby visible or inferred
duplicates are removed.
Only `front-facing` adapts brightness: it measures the relatively neutral
floor in the lower part of the frame, then scales the effective purple
`V min` from 20 to 40 and white `V min` from 80 to 100. The applied values
appear in the preview. The front scene can follow a chain of large, adjacent
purple components when several objects split the frame; positioning references
are selected across all accepted components.
The side-view mode retains its fixed thresholds and largest-component rule.

In `start-facing`, line-intersection inference is disabled: only directly
observed corners can be marked or selected as positioning references.
`front-facing` can infer a nearby occluded corner from two visible green
boundaries: a short side edge is extended to meet a roughly horizontal near
edge. The two boundaries may come from the same accepted safety-zone component
or from neighbouring accepted components split by an occluding object. They
must have white-field support, meet at a suitable angle, and intersect close
to their visible endpoints. The inferred point must lie near the lower part
of the combined safety-zone region and away from an already visible corner. This rule works from
either the left or right side. `general` retains its separate intersection
rule.
This scene also rejects candidate corners touching the image border, because
cropping can create a false turn. A visible launch-view corner must have white
field alongside both neighbouring contour edges. Its shorter edge may be
relatively short in the image when the safety zone is close to the camera.

The first five trackbars tune the HSV threshold and minimum connected-region
area. `corner min area` and `corner min side` add a separate gate for corner
detection, so small purple noise can remain visible without being labelled as
a safety-zone corner. The right image is the same binary mask used to extract
the blue contour.

Visible corner angles are accepted from 20 to 160 degrees in `general` to
account for perspective. In `start-facing`, the upper limit is 145 degrees:
a shallower turn near the image edge can be part of the raised frame while
the true corner lies outside the camera view. `front-facing` uses 15 to 145
degrees to retain the very acute near corners in a distant frontal view.

The overlay uses these colors:

- Blue: directly observed purple-mask contour.
- Green: contour segments that pass the white-field adjacency check and are
  treated as valid safety-zone boundaries. Green is drawn along the original
  blue segmented contour points; it does not connect simplified endpoints
  with a shortcut across bends or occlusion gaps.
- Orange: straight lines fitted from the green contour samples. Collinear
  green pieces separated by a limited gap are merged, and nearby fitted sides
  may extend to their intersection to complete the safety-zone polygon. The
  angle, endpoint gap, and extension limits prevent weakly converging opposite
  sides from forming a false triangle.
- Yellow: visible corners detected from strong turns that pass the green
  boundary check. A
  complete four-sided contour is not required; one or two visible corners can
  be shown.
- Magenta: an inferred corner obtained from the intersection of two green
  boundary lines when the physical corner itself is hidden. The extrapolation
  is limited to the nearby component and is suppressed when the two lines
  meet at an already visible corner.
- Cyan ring: a corner eligible as a nearby positioning reference. The yellow
  or magenta centre still shows whether the corner was observed or inferred.

`near band %` controls the relative-nearness gate. The lowest corner in the
current image is the reference; every corner within the selected percentage
of image height above it receives a cyan ring. The default is 10%, so at
480 px image height the band is 48 px and can contain several corners. This
is an image-based distance proxy for the current fixed camera view, not a
calibrated ground-plane distance.

In `front-facing`, the near edge must span at least 25% of its component's
image width and be approximately horizontal. The side edge must extend by at
least 8 px but no more than 40 px or 80% of the component height, whichever
is larger. Normally the near edge may extend by at most 6 px or 15% of its own
length. When both observed edges are long and the near edge has strong white
support, the near edge may extend by up to 60% of its visible length (within
the same overall extension limit). This handles a wider occlusion gap.
The front mode displays green boundaries at 0.35 white support so a short,
partly occluded side can participate, while directly visible corners still
require 0.40 support on both adjacent edges. At 640 px image width, a directly
visible frontal corner also needs at least 20 px of each adjacent edge; this
rejects the short cut end formed when a white object hides a purple strip.
The first 18 px next to a visible corner must also have white support on both
sides; a gap endpoint next to a coloured block therefore cannot count as a
directly visible corner just because the rest of its long edge touches white.
Two fitted sides in the same uninterrupted green chain produce a directly
visible corner and are never used to create a magenta point. A magenta corner
is permitted only when a real break separates the supporting green pieces.
The shorter green edge may still be used to infer the hidden corner. In
`front-facing`, a near-edge endpoint that is already a directly observed
corner is never extended again to create a second inferred corner. In
`general`, both green
segments must be long enough, their line intersection must be outside the
visible segment ends, and neither extension may exceed about 1.2 times the
component's short image dimension. If a view does not provide two suitable
green segments, no inferred point is shown.

While the preview is running, press `s` to save the current unmarked camera
frame to `照片/latest_live_raw_seekfree.png` or `照片/latest_live_raw_lingyan.png`, and
the annotated two-panel view to the corresponding `照片/latest_live_preview_*.png`
file. This captures the exact frame behind
a questionable marker without reopening the camera.

The preview also builds a white-field mask (`white S max` and `white V min`).
In `general`, at least one neighbouring purple contour segment must have white
pixels alongside most of its middle section. In both named scenes, both
segments must pass this check. The white thresholds are
trackbars because exposure and indoor lighting can change the apparent
brightness of the field.
The boundary test samples outward from each purple segment and checks the
surface immediately beyond its blurred edge. It does not count white pixels
on the opposite side of a purple strip. Saturated warm pixels from orange
blocks are excluded even when `white S max` is set to 179; dim, nearly neutral
field pixels get a small brightness allowance.

In `start-facing` and `general`, only the largest qualified purple component
can produce yellow corners. `front-facing` can also use nearby large pieces
of the same safety-zone frame. Corner candidates must be convex outer turns;
inward turns remain blue-only.

When an object hides part of the safety zone, the mask contour can contain a
short pair of segments at the occlusion break. In `general`, those segments
are rejected by an adaptive edge-support gate: each side of a candidate
corner must be at least 65% of the component's shorter image dimension (and
at least the `corner min side` value). Both named scenes use a smaller length
threshold; both named scenes require white on both sides of a corner.

## Create or repair the environment

From the VS Code terminal in this folder:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project is independent of the STM32 firmware checkout.

## Camera calibration and metric ranging

The project includes a calibration workflow under `calibration/`, now defaulting
to the original LINGYAN Pro 640 x 480 camera with no rotation. A print-ready
9 x 6 inner-corner, 20 mm square A4 board is available as a
[print-ready PDF](output/pdf/lingyan_chessboard_9x6_A4_20mm.pdf). Print at actual size and
measure the square size before solving intrinsics. Camera images and results
use separate `calibration/data/lingyan/` and `calibration/results/lingyan/`
folders. See
[摄像头标定执行计划.md](摄像头标定执行计划.md) before
collecting data. VS Code tasks are available for chessboard capture, intrinsic
calibration, hold-out validation, and fitted camera-to-body mounting pose.
