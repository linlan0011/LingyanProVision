"""Map feedback for nearest-corner identity and independent visual pose."""
from __future__ import annotations

import math
import time
import numpy as np

from ground_corner_association import _clip_polygon
from pose_association import validated_corrected_pose


STATUS_NAMES = {
    'corrected': '视觉位姿已更新', 'no_valid_corner': '无有效测距角点',
    'calibration_unavailable': '地面测距标定不可用', 'waiting_for_prior': '等待输入位姿',
    'map_unavailable': '地图不可用', 'no_candidate_safety_zone': '视野内无候选安全区',
    'nearest_out_of_range': '最近点超出定位量程', 'nearest_unusable': '最近点不可用于定位',
    'two_support_edges_required': '最近点缺少唯一的两条支持边',
    'insufficient_edge_samples': '有效边采样不足', 'edge_too_short': '地面边过短',
    'edge_fit_unreliable': '地面边拟合不可靠', 'support_crosses_corner': '支持边穿过角点，方向不确定',
    'edge_direction_unreliable': '边延伸方向不可靠', 'edges_not_orthogonal': '两条地面边不满足正交容差',
    'identity_no_match': '边方向未匹配到地图角', 'ambiguous_corner': '角点身份多解',
    'ambiguous_safety_zone': '两个安全区均相容，身份待定',
    'visual_heading_inconsistent': '两条边反算的视觉航向不一致',
}


def draw_visual_feedback(canvas, transform, data, prior):
    report = data['visual_pose']
    view = report.get('candidate_view') or {}
    tags = ('live_identity', 'visual_pose')

    def outline(values, color, extra):
        polygon = np.asarray(values, float).reshape(-1, 2)
        for normal, offset in (([1.,0.],-transform.x_min),([-1.,0.],transform.x_max),
                              ([0.,1.],-transform.y_min),([0.,-1.],transform.y_max)):
            polygon = _clip_polygon(polygon, np.asarray(normal), offset)
        if len(polygon) >= 3:
            points = [transform.world_to_canvas(*point) for point in polygon]
            canvas.create_line(*(value for point in points+[points[0]] for value in point),
                               fill=color, width=2, dash=(5,3), tags=tags+(extra,))

    outline(view.get('central_footprint_map_mm', ()), '#42a5f5', 'central_view')
    outline(view.get('possible_footprint_map_mm', ()), '#b388ff', 'possible_view')
    radius = view.get('prior_radius_mm', 200.)
    left, top = transform.world_to_canvas(prior.x_mm-radius, prior.y_mm+radius)
    right, bottom = transform.world_to_canvas(prior.x_mm+radius, prior.y_mm-radius)
    canvas.create_oval(left, top, right, bottom, outline='#ffcc66', width=2,
                       dash=(5,3), tags=tags+('pose_uncertainty',))
    pose = validated_corrected_pose(data)
    corner_id = report.get('corner_id') if pose is not None else None
    selected = report.get('selected_corner')
    for landmark in view.get('landmarks', ()):
        x,y = transform.world_to_canvas(*landmark['map_xy_mm'])
        if landmark['corner_id'] == corner_id:
            color = '#ffa000' if selected and selected['inferred'] else '#00cc66'
            canvas.create_oval(x-11,y-11,x+11,y+11,outline=color,width=3,tags=tags)
            canvas.create_text(x+13,y-15,text=corner_id,anchor='w',fill=color,tags=tags)
        elif landmark['possibly_visible']:
            canvas.create_oval(x-6,y-6,x+6,y+6,outline='#b388ff',width=2,tags=tags)
    if pose is not None:
        x,y = transform.world_to_canvas(pose.x_mm, pose.y_mm)
        # Use the same map transform for the arrow so scaling and canvas Y are consistent.
        ahead = transform.world_to_canvas(pose.x_mm+200.*math.cos(math.radians(pose.heading_deg)),
                                          pose.y_mm+200.*math.sin(math.radians(pose.heading_deg)))
        dx,dy = ahead[0]-x,ahead[1]-y
        scale = 64./max(1e-9, math.hypot(dx,dy))
        arrow_tags = tags+('corrected_pose',)
        canvas.create_oval(x-7,y-7,x+7,y+7,fill='#00dddd',outline='white',width=2,tags=arrow_tags)
        canvas.create_line(x,y,x+dx*scale,y+dy*scale,fill='#00dddd',width=4,arrow='last',
                           arrowshape=(13,16,6),tags=arrow_tags)
        canvas.create_text(x+10,y+15,text=f'视觉校正 ({pose.x_mm:.1f}, {pose.y_mm:.1f}, {pose.heading_deg:.2f}°)',
                           anchor='nw',fill='#00dddd',tags=arrow_tags)
    ids = ', '.join(map(str, view.get('possible_safety_ids', ()))) or '无'
    limit = view.get('max_distance_mm')
    range_text = f'{limit:g} mm' if limit is not None else '不限距离'
    summary = [f'候选安全区：{ids}；视野位置半径：{radius:g} mm',
               f'定位量程：{range_text}；基准角编号：{corner_id or "?"}',
               STATUS_NAMES.get(report['status'], report['status']),
               f'惯导输入：X={prior.x_mm:.1f} Y={prior.y_mm:.1f} θ={prior.heading_deg:.2f}°']
    if selected is not None:
        summary.append(f'最近点（{"补偿" if selected["inferred"] else "可见"}）：r={selected["range_mm"]:.1f} mm，'
                       f'车体系 XY=({selected["body_xy_mm"][0]:.1f}, {selected["body_xy_mm"][1]:.1f})')
    if pose is not None:
        summary.append(f'视觉校正：X={pose.x_mm:.1f} Y={pose.y_mm:.1f} θ={pose.heading_deg:.2f}°')
        summary.append(f'差值：ΔX={pose.x_mm-prior.x_mm:+.1f} ΔY={pose.y_mm-prior.y_mm:+.1f} '
                       f'Δθ={(pose.heading_deg-prior.heading_deg+180.)%360.-180.:+.2f}°')
        summary.append('青色箭头为视觉校正；红色箭头为输入位姿')
    else:
        summary.append('本帧未更新视觉位姿')
    try:
        age = time.time()-float(data['created_at_unix'])
        age_text = f'{max(0.,age):.1f} s' if math.isfinite(age) else '未知'
    except (KeyError,TypeError,ValueError):
        age_text = '未知'
    summary.append(f'回传帧：{data.get("frame_number", "?")}；反馈年龄：{age_text}（非拍摄时间）')
    return summary
