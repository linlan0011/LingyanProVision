# Lingyan Pro Vision

This is a standalone VS Code project for testing the purple safety-zone detector.
The current camera choice is the original LINGYAN Pro at 640 x 480 without
rotation. The SeekFree 彩钻风 Pro remains an explicit alternative profile.
Press `q` or `Esc` to stop the preview.

## 场地地图位姿选择器

运行 `pose_map_gui.py`，或在 VS Code 中选择 **Field map pose selector**。
程序会自动加载你提供的 `field_map.png`：

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
- 点击“确认并输出给识别程序”后，位姿会原子写入
  `selected_pose.json`，避免识别程序读到半份 JSON。

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
自定义分区名称不会自动改变 `--scene` 的视觉算法规则。
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

## Run in VS Code

1. Open this folder in VS Code.
2. Select the interpreter at `.venv\Scripts\python.exe` if VS Code has not
   selected it automatically.
3. Open `purple_camera_segmentation.py` and press `F5`, or run the task
   **Run purple camera segmentation (LINGYAN legacy front)**.

The default command is:

```powershell
.venv\Scripts\python.exe purple_camera_segmentation.py --camera lingyan --index 0 --scene front-facing
```

`--camera lingyan` uses the earlier 640 × 480, unrotated capture and the
original scene rules. Both profiles have separate VS Code front (`front-facing`)
and side (`start-facing`) run/debug entries. The optional `--camera seekfree`
profile captures 768 × 1024 and rotates to 1024 × 768 before recognition.
On the sample safety-zone view, the SeekFree front profile uses a finer contour
approximation and accepts corners up to 160°; the older profile values remain
unchanged. The sample results are
[SeekFree front](seekfree_front_adaptation_preview.png) (three visible corners;
the left near corner is obscured) and
[SeekFree side](seekfree_side_adaptation_preview.png) (four visible corners).

`--scene front-facing` is the current test for the aligned frontal view of a
safety zone. It accepts visible near-edge corners of the shallow trapezoid
without requiring a complete quadrilateral; under occlusion or dark lighting,
only one corner may pass the two-sided white-field check. The completed launch-side scene
is preserved as `--scene start-facing`; see [安全区侧面参数及规则.md](安全区侧面参数及规则.md)
for its camera values and rules. `--scene general` retains the wider corner
rules and green-line intersection inference. Scene selection is still manual
in this PC prototype; the displayed pose file does not yet switch scenes.

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
frame to `latest_live_raw_seekfree.png` or `latest_live_raw_lingyan.png`, and
the annotated two-panel view to the corresponding `latest_live_preview_*.png`
file in this folder. This captures the exact frame behind
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
