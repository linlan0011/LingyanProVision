"""Display-only layout; detection, identity and ranging keep native pixels."""
from __future__ import annotations

import cv2
import numpy as np


HEADER_HEIGHT = 72


def preview_dimensions(image_shape):
    """Fit the desktop work area, with a 2:1 colour/mask width ratio."""
    screen_width, screen_height = 1700, 1000
    try:
        import ctypes
        from ctypes import wintypes
        area = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
            screen_width, screen_height = area.right-area.left, area.bottom-area.top
    except (AttributeError, OSError):
        pass
    height, width = image_shape[:2]
    available_width = max(640, screen_width-60)
    available_height = max(360, screen_height-120-HEADER_HEIGHT)
    colour_width = min(round(available_width*2/3), round(available_height*width/height))
    colour_width = max(320, colour_width)
    mask_width = colour_width//2
    return colour_width+mask_width, round(colour_width*height/width)+HEADER_HEIGHT


class ParameterPanel:
    """Detachable trackbars; closing the panel keeps its last values."""
    name = 'HSV / Geometry Parameters - P to hide'
    maxima = {'H low':179, 'H high':179, 'S min':255, 'V min':255,
              'min area':5000, 'corner min area':10000, 'corner min side':200,
              'white S max':255, 'white V min':255,
              'pose radius mm':3000, 'identity tol deg':44,
              'orthogonal tol deg':44, 'heading tol deg':44}

    def __init__(self, values, visible=False):
        self.values = dict(values)
        self.visible = False
        if visible:
            self.toggle()

    def poll(self):
        if not self.visible:
            return
        try:
            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                self.visible = False
                return
            for name in self.values:
                self.values[name] = cv2.getTrackbarPos(name, self.name)
        except cv2.error:
            self.visible = False

    def get(self, name):
        return self.values[name]

    def set(self, name, value):
        self.values[name] = int(value)
        if self.visible:
            cv2.setTrackbarPos(name, self.name, int(value))

    def toggle(self):
        if self.visible:
            self.poll()
            if self.visible:
                cv2.destroyWindow(self.name)
            self.visible = False
            return
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, 460, 26)
        for name, value in self.values.items():
            cv2.createTrackbar(name, self.name, value, self.maxima[name], lambda _: None)
        cv2.imshow(self.name, np.full((26, 460, 3), 35, np.uint8))
        self.visible = True


def _text(image, text, xy, colour=(230,230,230), scale=.65, thickness=2):
    # A dark outline keeps text readable on both the scene and binary mask.
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, thickness, cv2.LINE_AA)


def _corner_labels(image, recognitions, identities, readings, pixel_scale, visual_pose=None,
                   geometry_readings=()):
    identity = {tuple(map(int, r.observation.point)):r.corner_id for r in identities}
    ranges = {tuple(map(int, r.pixel)):r for r in readings}
    geometry = {tuple(map(int, np.rint(r.pixel))):r for r in geometry_readings}
    selected = visual_pose.get('selected_corner') if visual_pose else None
    selected_pixel = tuple(selected['pixel']) if selected is not None else None
    occupied = []
    for _, visible, hidden in recognitions:
        for inferred, values in ((False,visible),(True,hidden)):
            for point, angle in values:
                key = tuple(map(int,point))
                centre = np.rint(np.asarray(point)*pixel_scale).astype(int)
                colour = (255,0,255) if inferred else (0,255,255)
                cv2.circle(image, tuple(centre), 5, colour, -1, cv2.LINE_AA)
                label = f"{identity.get(key) or '?'} {'EST' if inferred else 'V'} a={angle:.0f}"
                if key == selected_pixel:
                    cv2.circle(image, tuple(centre), 10, (255,255,0), 2, cv2.LINE_AA)
                    label = 'NEAREST ' + label
                reading = ranges.get(key)
                lines = [label]
                if reading is not None and reading.valid:
                    lines.append(f"r={reading.distance_mm:.0f}mm")
                elif reading is not None and reading.reason == 'beyond_distance_limit':
                    ranked = geometry.get(key)
                    lines.append(f'geom r={ranked.distance_mm:.0f}mm | range gate'
                                 if ranked is not None and ranked.valid else 'OUT OF RANGE')
                text_width = max(cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,.70,2)[0][0] for t in lines)
                box_height = 26*len(lines)+6
                choices = []
                for dx,dy in ((12,-box_height-10),(12,12),(-text_width-14,-box_height-10),
                              (-text_width-14,12),(12,-box_height-42),(12,44)):
                    x = max(3,min(int(centre[0])+dx,image.shape[1]-text_width-8))
                    y = max(3,min(int(centre[1])+dy,image.shape[0]-box_height-3))
                    box = (x,y,x+text_width+5,y+box_height)
                    overlap = sum(max(0,min(box[2],b[2])-max(box[0],b[0]))
                                  *max(0,min(box[3],b[3])-max(box[1],b[1])) for b in occupied)
                    choices.append((overlap,abs(dx)+abs(dy),box))
                _,_,box = min(choices)
                occupied.append(box)
                for index,text in enumerate(lines):
                    _text(image,text,(box[0]+2,box[1]+22+26*index),colour,.70,2)


def compose_preview(overlay, mask_image, recognitions, identities, readings, *,
                    size, title, parameters, pose_text, status, boundary_result=None,
                    ranger=None, visual_pose=None, geometry_readings=()):
    total_width, _ = size
    colour_width = round(total_width*2/3)
    mask_width = total_width-colour_width
    height,width = overlay.shape[:2]
    colour_height = round(height*colour_width/width)
    mask_height = round(height*mask_width/width)
    canvas = np.full((HEADER_HEIGHT+colour_height,total_width,3),24,np.uint8)
    colour = cv2.resize(overlay,(colour_width,colour_height),interpolation=cv2.INTER_AREA)
    binary = cv2.resize(mask_image,(mask_width,mask_height),interpolation=cv2.INTER_AREA)
    _corner_labels(colour,recognitions,identities,readings,colour_width/width,visual_pose,geometry_readings)
    _corner_labels(binary,recognitions,identities,[],mask_width/width,visual_pose)
    if boundary_result is not None:
        for line in boundary_result.lines:
            if line.observed_length < boundary_result.config.minimum_edge:
                continue
            point = max(line.members,key=lambda p:p.length).fitted_origin
            if boundary_result.config.point_geometry is not None:
                point = boundary_result.config.point_geometry.to_pixels([point])[0]
            for pane,factor in ((colour,colour_width/width),(binary,mask_width/width)):
                if line.length*factor < 60:
                    continue  # Keep tiny segments drawn without crowding corner labels.
                x,y = np.rint(point*factor).astype(int)
                _text(pane,f'L{line.line_id}',(max(3,min(x+4,pane.shape[1]-55)),
                                            max(20,min(y-7,pane.shape[0]-5))),
                      (0,70,255),.65,2)
    canvas[HEADER_HEIGHT:,:colour_width] = colour
    canvas[HEADER_HEIGHT:HEADER_HEIGHT+mask_height,colour_width:] = binary
    cv2.line(canvas,(colour_width,HEADER_HEIGHT),(colour_width,canvas.shape[0]),(85,85,85),1)
    _text(canvas,title,(12,26),(100,255,255),.70,2)
    _text(canvas,parameters+'  |  P: parameters  S: save  Q/Esc: quit',(12,56),scale=.58)
    sidebar = canvas[HEADER_HEIGHT+mask_height:,colour_width:]
    y = 25
    def row(text,colour=(220,220,220),scale=.58,advance=25):
        nonlocal y
        if y <= sidebar.shape[0]-6:
            _text(sidebar,text,(10,y),colour,scale)
        y += advance
    ids = sorted({r.corner_id for r in identities if r.corner_id})
    row('IDs: '+(', '.join(ids) or 'pending')
        +(f' | Z={ranger.target_z_mm:g}' if ranger is not None else ''),(100,255,120),.65,28)
    if visual_pose is not None:
        view = visual_pose['candidate_view'] or {}
        radius = view.get('prior_radius_mm', 200.)
        limit = view.get('max_distance_mm')
        range_text = (f'range<={limit:g}mm' if limit is not None else 'range unlimited')
        if visual_pose['status'] == 'calibration_unavailable':
            range_text = 'ranging required'
        row(f'ZONE VIEW: R={radius:g}mm {range_text}',
            (100,255,120),.51,21)
        row('Zones: '+(', '.join(map(str,view.get('possible_safety_ids',()))) or 'none'),scale=.50,advance=21)
        row(visual_pose['status'],scale=.49,advance=21)
        pose = visual_pose.get('corrected_pose')
        if visual_pose['status'] == 'corrected' and pose is not None:
            row(f"VIS X={pose['x_mm']:.1f} Y={pose['y_mm']:.1f}",(255,255,0),.53,21)
            row(f"VIS heading={pose['heading_deg']:.2f} deg",(255,255,0),.53,21)
        else:
            row('VIS pose pending',scale=.50,advance=21)
    row('BLUE contour / GREEN observed / RED fit',scale=.51,advance=23)
    row('YELLOW visible / MAGENTA estimate / CYAN ref',scale=.51,advance=23)
    if ranger is not None:
        available = max(0,(sidebar.shape[0]-y-20)//44)
        for reading in readings[:available]:
            colour = (0,180,255) if reading.inferred else (100,255,120)
            row(f'{reading.label}: r={reading.distance_mm:.0f}mm' if reading.valid
                else f'{reading.label}: {reading.reason}',colour,.60,23)
            row(f'X={reading.x_mm:.0f} Y={reading.y_mm:.0f} b={reading.bearing_deg:+.1f}'
                if reading.valid else 'estimate unavailable',colour,.54,21)
        if len(readings)>available:
            row(f'+{len(readings)-available} more (S saves all results)',scale=.52)
        elif y+23 < sidebar.shape[0]:
            row('BODY X forward / Y left | range estimates',scale=.49)
    else:
        row(status,scale=.52)
    # Keep the selected pose available without covering the image.
    if y+23 < sidebar.shape[0]:
        row(pose_text,scale=.48)
    return canvas
