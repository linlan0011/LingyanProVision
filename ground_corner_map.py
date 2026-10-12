"""Read-only map feedback for the bounded-pose lower-corner association."""
from __future__ import annotations

import numpy as np

from ground_corner_association import _clip_polygon


def draw_ground_feedback(canvas, transform, data, prior):
    report = data['ground_association']
    config = report['config']
    radius = config['prior_radius_mm']
    tags = ('live_identity', 'ground_association')

    def circle(x, y, r, color, extra=()):
        left, top = transform.world_to_canvas(x-r, y+r)
        right, bottom = transform.world_to_canvas(x+r, y-r)
        canvas.create_oval(left, top, right, bottom, outline=color, width=2,
                           dash=(5, 3), tags=tags+extra)

    def outline(values, color, extra=()):
        polygon = np.asarray(values, float).reshape(-1, 2)
        for normal, offset in (([1.,0.],-transform.x_min),([-1.,0.],transform.x_max),
                               ([0.,1.],-transform.y_min),([0.,-1.],transform.y_max)):
            polygon = _clip_polygon(polygon, np.asarray(normal), offset)
        if len(polygon) >= 3:
            points = [transform.world_to_canvas(*point) for point in polygon]
            canvas.create_line(*(value for point in points+[points[0]] for value in point),
                               fill=color, width=2, dash=(5,3), tags=tags+extra)

    outline(report['possible_footprint_map_mm'], '#b388ff', ('possible_view',))
    outline(report['central_footprint_map_mm'], '#42a5f5', ('central_view',))
    circle(prior.x_mm, prior.y_mm, radius, '#ffcc66', ('pose_uncertainty',))
    for observation in report['observations']:
        circle(*observation['map_center_mm'], observation['map_radius_mm'], '#559974')
    names = {observation['corner_id']: observation for observation in report['observations']
             if observation['corner_id']}
    for landmark in report['landmarks']:
        x,y = transform.world_to_canvas(*landmark['map_xy_mm'])
        observation = names.get(landmark['corner_id'])
        if observation is not None:
            color = '#ffa000' if observation['inferred'] else '#00cc66'
            canvas.create_oval(x-11,y-11,x+11,y+11,outline=color,width=3,tags=tags)
            text = landmark['corner_id']+(' 补偿' if observation['inferred'] else ' 可见')
            canvas.create_text(x+13,y-15,text=text,anchor='w',fill=color,tags=tags)
        elif landmark['possibly_visible']:
            canvas.create_oval(x-6,y-6,x+6,y+6,outline='#b388ff',width=2,tags=tags)
    estimate = report['vision_vehicle_position_mm']
    if estimate is not None:
        x,y = transform.world_to_canvas(*estimate)
        canvas.create_line(x-7,y-7,x+7,y+7,fill='#00dddd',width=3,tags=tags)
        canvas.create_line(x-7,y+7,x+7,y-7,fill='#00dddd',width=3,tags=tags)
        canvas.create_text(x+10,y-10,text='视觉反算位置',anchor='w',fill='#00dddd',tags=tags)
    status_names = {'unique':'身份唯一', 'partial_unique':'部分身份唯一', 'ambiguous':'身份多解',
                    'no_match':'未匹配', 'waiting_for_ground_corners':'等待有效下沿点',
                    'waiting_for_prior':'等待仿真位姿', 'map_unavailable':'地图不可用',
                    'calibration_unavailable':'标定不可用',
                    'search_limit_no_confirmation':'候选过多，暂不确认'}
    ids = sorted(names)
    summary = ['下沿编号：'+('、'.join(ids) if ids else '?（未确认）'),
               f"位置圆半径：{radius:g} mm；量程：{report['max_distance_mm']:g} mm"
               if report['max_distance_mm'] is not None else '等待地面测距',
               status_names.get(report['status'],report['status'])]
    safety_ids = report['possible_safety_ids']
    summary.append('候选安全区：'+('、'.join(map(str,safety_ids)) if safety_ids else '无'))
    for observation in report['observations'][:2]:
        name = observation['corner_id'] or '?'
        kind = '补偿' if observation['inferred'] else '可见'
        summary.append(f"{name}（{kind}）：{observation['range_mm']:.1f} mm")
    if estimate is not None:
        summary.append(f'视觉反算：({estimate[0]:.1f}, {estimate[1]:.1f}) mm')
    return summary
