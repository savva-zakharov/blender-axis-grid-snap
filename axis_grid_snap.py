bl_info = {
    "name": "Axis Grid Snap",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Grid Snap  |  G in Object Mode",
    "description": "Move objects with separate snapping increments per axis, "
                   "with a ruler (single axis) or grid floor (plane) overlay",
    "category": "3D View",
}

import math

import bpy
import gpu
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy_extras import view3d_utils
from bl_operators.presets import AddPresetBase
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector, geometry

AXIS_VECS = (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
AXIS_COLORS = ((1.0, 0.21, 0.32), (0.54, 0.86, 0.0), (0.17, 0.56, 1.0))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class AxisGridSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(
        name="Use for G",
        description="Replace the G key in Object Mode with Axis Grid Move. "
                    "When off, G falls through to Blender's normal move",
        default=True,
    )
    step_x: FloatProperty(name="X", default=1.0, min=1e-5, soft_max=100.0,
                          subtype='DISTANCE', precision=4)
    step_y: FloatProperty(name="Y", default=1.0, min=1e-5, soft_max=100.0,
                          subtype='DISTANCE', precision=4)
    step_z: FloatProperty(name="Z", default=1.0, min=1e-5, soft_max=100.0,
                          subtype='DISTANCE', precision=4)
    use_double: BoolProperty(
        name="Double Grid",
        description="Alternate between two spacings on each axis (A, B, A, B, ...), "
                    "e.g. module + joint. Set B equal to A to keep an axis single",
        default=False,
    )
    step_x2: FloatProperty(name="X B", default=0.25, min=1e-5, soft_max=100.0,
                           subtype='DISTANCE', precision=4)
    step_y2: FloatProperty(name="Y B", default=0.25, min=1e-5, soft_max=100.0,
                           subtype='DISTANCE', precision=4)
    step_z2: FloatProperty(name="Z B", default=0.25, min=1e-5, soft_max=100.0,
                           subtype='DISTANCE', precision=4)
    absolute: BoolProperty(
        name="Snap to World Grid",
        description="Snap the pivot onto world grid lines (on) or move in "
                    "step-sized increments from where it started (off)",
        default=True,
    )
    typed_direction: EnumProperty(
        name="Typed Direction",
        description="How the sign of a typed distance is interpreted",
        items=(
            ('RELATIVE', "Relative",
             "Typed values follow the direction you dragged with the mouse on each axis "
             "(drag toward -X, type 1 -> moves -1)"),
            ('ABSOLUTE', "Absolute",
             "Typed values use world axis directions (1 is always +, -1 always -)"),
        ),
        default='RELATIVE',
    )
    extent: IntProperty(
        name="Overlay Extent",
        description="Number of increments drawn on each side of the pivot",
        default=20, min=2, max=200,
    )

    def grids(self):
        a = (self.step_x, self.step_y, self.step_z)
        b = (self.step_x2, self.step_y2, self.step_z2) if self.use_double else a
        return tuple(AxisGrid(a[i], b[i]) for i in range(3))


class AxisGrid:
    """Grid lines along one axis with alternating spacings a, b (a == b is a plain grid).

    Line m sits at: m even -> (m/2)*(a+b), m odd -> that + a. So lines run
    ..., -(a+b), -b, 0, a, a+b, 2a+b, ... measured from the grid base.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.double = abs(a - b) > 1e-9
        self.scale = (a + b) * 0.5  # representative size for overlay tick lengths

    def pos(self, m):
        return (m // 2) * (self.a + self.b) + (self.a if m % 2 else 0.0)

    def nearest(self, t):
        """Index of the grid line closest to t."""
        k = math.floor(t / (self.a + self.b))
        return min((2 * k, 2 * k + 1, 2 * k + 2), key=lambda m: abs(self.pos(m) - t))

    def snap(self, t):
        return self.pos(self.nearest(t))

    def is_major(self, m):
        # Every 5th period on a double grid (10 lines), every 5th line on a single one.
        return m % (10 if self.double else 5) == 0


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

# All overlay geometry is built in the move's orientation space (see _orientation)
# and converted to world space just before drawing.

def _grid_base(op, axis):
    """Orientation-space coordinate of grid line 0 along an axis."""
    return 0.0 if op.absolute else op.pivot[axis]


def _fade(k, center, extent):
    return max(0.0, 1.0 - abs(k - center) / (extent + 1))


def _ruler_lines(op, axis, grid, extent, view_dir):
    pos, col = [], []
    a = AXIS_VECS[axis]
    perp = a.cross(view_dir)
    if perp.length < 1e-6:
        perp = AXIS_VECS[(axis + 1) % 3].copy()
    perp.normalize()

    r, g, b = AXIS_COLORS[axis]
    base = _grid_base(op, axis)
    center = grid.nearest(op.target[axis] - base)
    size = grid.scale

    def at(m):
        p = op.target.copy()
        p[axis] = base + grid.pos(m)
        return p

    # Faint infinite constraint line
    far = 10000.0
    pos += [op.pivot - a * far, op.pivot + a * far]
    col += [(r, g, b, 0.25)] * 2

    # Ruler spine, one segment per interval. On a double grid the A intervals are
    # drawn bright and the B intervals dim, so the alternation reads at a glance.
    for m in range(center - extent, center + extent):
        alpha = 0.9 if not grid.double or m % 2 == 0 else 0.35
        pos += [at(m), at(m + 1)]
        col += [(r, g, b, alpha)] * 2

    for m in range(center - extent, center + extent + 1):
        p = at(m)
        alpha = _fade(m, center, extent)
        if m == center:
            length, c = 0.6 * size, (1.0, 1.0, 1.0, 1.0)
        elif grid.is_major(m):
            length, c = 0.35 * size, (r, g, b, alpha)
        else:
            length, c = 0.15 * size, (1.0, 1.0, 1.0, 0.7 * alpha)
        pos += [p - perp * length, p + perp * length]
        col += [c, c]
    return pos, col


def _plane_lines(op, normal_axis, grids, extent):
    pos, col = [], []
    u, v = [i for i in range(3) if i != normal_axis]
    base_u, base_v = _grid_base(op, u), _grid_base(op, v)
    cu = grids[u].nearest(op.target[u] - base_u)
    cv = grids[v].nearest(op.target[v] - base_v)
    u_lo, u_hi = base_u + grids[u].pos(cu - extent), base_u + grids[u].pos(cu + extent)
    v_lo, v_hi = base_v + grids[v].pos(cv - extent), base_v + grids[v].pos(cv + extent)

    def point(pu, pv):
        p = op.pivot.copy()
        p[u], p[v] = pu, pv
        return p

    # Lines running along u (one per v grid line) take v's axis colour, and vice versa,
    # so you can tell which spacing is which.
    for axis, c0, other_lo, other_hi, base in (
        (v, cv, u_lo, u_hi, base_v),
        (u, cu, v_lo, v_hi, base_u),
    ):
        grid = grids[axis]
        r, g, b = AXIS_COLORS[axis]
        for m in range(c0 - extent, c0 + extent + 1):
            t = base + grid.pos(m)
            fade = _fade(m, c0, extent)
            if m == c0:
                c = (1.0, 1.0, 1.0, 0.9)
            elif grid.is_major(m):
                c = (r, g, b, 0.8 * fade)
            elif grid.double and m % 2:
                c = (0.8, 0.8, 0.8, 0.2 * fade)  # second line of each A/B pair
            else:
                c = (0.8, 0.8, 0.8, 0.4 * fade)
            if axis == v:
                pos += [point(other_lo, t), point(other_hi, t)]
            else:
                pos += [point(t, other_lo), point(t, other_hi)]
            col += [c, c]
    return pos, col


def _cross_lines(op, grids):
    pos, col = [], []
    for i, a in enumerate(AXIS_VECS):
        r, g, b = AXIS_COLORS[i]
        h = a * (grids[i].scale * 0.5)
        pos += [op.target - h, op.target + h]
        col += [(r, g, b, 1.0)] * 2
    return pos, col


def draw_overlay(op):
    rv3d = bpy.context.region_data
    if rv3d is None:
        return
    view_dir = op.rot_inv @ (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))
    grids, extent = op.grids, op.extent

    kind, axis = op.constraint if op.constraint else (None, None)
    if kind == 'AXIS':
        pos, col = _ruler_lines(op, axis, grids[axis], extent, view_dir)
    elif kind == 'PLANE':
        pos, col = _plane_lines(op, axis, grids, min(extent, 60))
    else:
        pos, col = _cross_lines(op, grids)

    # Travel line from start to current position
    pos += [op.pivot, op.target]
    col += [(1.0, 1.0, 1.0, 0.5)] * 2
    pos = [op.to_world(p) for p in pos]

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": pos, "color": col})
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(1.0)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


# ---------------------------------------------------------------------------
# Move operator
# ---------------------------------------------------------------------------

def _root_objects(objs):
    """Drop objects whose ancestor is also being moved (they follow the parent)."""
    sel = set(objs)
    roots = []
    for o in objs:
        p = o.parent
        while p is not None and p not in sel:
            p = p.parent
        if p is None:
            roots.append(o)
    return roots


def _orientation(context):
    """(label, rotation 3x3, grid origin) for the scene's current transform orientation.

    Snapping happens in this frame: X/Y/Z constraints, increments, typed values and the
    world-grid option all refer to its axes. Only Cursor moves the grid origin (to the cursor).
    """
    scene = context.scene
    slot = scene.transform_orientation_slots[0]
    kind = slot.type
    ob = context.active_object
    origin = Vector()
    if kind in {'LOCAL', 'NORMAL', 'GIMBAL'} and ob is not None:
        # In Object Mode Normal equals Local; Gimbal is approximated by Local.
        rot = ob.matrix_world.to_quaternion().to_matrix()
    elif kind == 'PARENT' and ob is not None and ob.parent is not None:
        rot = ob.parent.matrix_world.to_quaternion().to_matrix()
    elif kind == 'VIEW' and context.region_data is not None:
        rot = context.region_data.view_rotation.to_matrix()
    elif kind == 'CURSOR':
        rot = scene.cursor.matrix.to_quaternion().to_matrix()
        origin = scene.cursor.location.copy()
    elif slot.custom_orientation is not None:  # custom orientations report their name as type
        rot = slot.custom_orientation.matrix.to_quaternion().to_matrix()
    else:
        kind = 'GLOBAL'
        rot = Matrix.Identity(3)
    return kind.title() if kind.isupper() else kind, rot, origin


def _unit_scale(us):
    return us.scale_length if us.system != 'NONE' else 1.0


def _fmt(context, value):
    us = context.scene.unit_settings
    try:
        return bpy.utils.units.to_string(us.system, 'LENGTH', value * _unit_scale(us), precision=4)
    except Exception:
        return f"{value:.4g}"


# Unitless typed numbers are read in the scene's chosen length unit, like Blender's own fields.
UNIT_ABBR = {
    'KILOMETERS': "km", 'METERS': "m", 'CENTIMETERS': "cm", 'MILLIMETERS': "mm",
    'MICROMETERS': "um", 'MILES': "mi", 'FEET': "ft", 'INCHES': "in", 'THOU': "thou",
}

NUMPAD_CHARS = {
    **{f"NUMPAD_{d}": str(d) for d in range(10)},
    'NUMPAD_PERIOD': ".", 'NUMPAD_MINUS': "-", 'NUMPAD_PLUS': "+",
    'NUMPAD_SLASH': "/", 'NUMPAD_ASTERIX': "*",
}
# Characters that may start a typed value; once typing, unit letters etc. are accepted too.
NUM_START_CHARS = set("0123456789.-+(")
NUM_CHARS = NUM_START_CHARS | set(",)*/ '\"") | set("abcdefghijklmnopqrstuvw")


def _parse_length(context, text):
    """Typed text -> length in Blender units, or None if it doesn't evaluate (yet)."""
    us = context.scene.unit_settings
    ref = UNIT_ABBR.get(us.length_unit) if us.system != 'NONE' else None
    try:
        value = bpy.utils.units.to_value(us.system, 'LENGTH', text,
                                         str_ref_unit=("1" + ref) if ref else None)
    except ValueError:
        return None
    return value / _unit_scale(us)


class VIEW3D_OT_axis_grid_move(bpy.types.Operator):
    """Move selected objects, snapping each axis to its own increment"""
    bl_idname = "view3d.axis_grid_move"
    bl_label = "Axis Grid Move"
    bl_options = {'REGISTER', 'UNDO'}

    offset: FloatVectorProperty(name="Offset", subtype='TRANSLATION', size=3)

    @classmethod
    def poll(cls, context):
        return (context.mode == 'OBJECT'
                and context.area is not None and context.area.type == 'VIEW_3D'
                and context.region is not None and context.region.type == 'WINDOW')

    # Redo panel / scripted use: apply the stored offset directly.
    def execute(self, context):
        t = Matrix.Translation(Vector(self.offset))
        for o in _root_objects(list(context.selected_editable_objects)):
            o.matrix_world = t @ o.matrix_world
        return {'FINISHED'}

    def invoke(self, context, event):
        settings = context.scene.axis_grid
        if not settings.enabled:
            return {'PASS_THROUGH'}  # let Blender's own G handle it

        self.objs = _root_objects(list(context.selected_editable_objects))
        if not self.objs:
            return {'PASS_THROUGH'}
        self.start_mw = [o.matrix_world.copy() for o in self.objs]

        active = context.active_object
        if active in self.objs:
            pivot_world = active.matrix_world.translation.copy()
        else:
            pivot_world = sum((mw.translation for mw in self.start_mw), Vector()) / len(self.start_mw)

        # pivot / target / delta are kept in orientation space; see to_world().
        self.orient_name, self.rot, self.origin = _orientation(context)
        self.rot_inv = self.rot.transposed()
        self.pivot_world = pivot_world
        self.pivot = self.to_local(pivot_world)

        self.grids = settings.grids()
        self.absolute = settings.absolute
        self.relative_dir = settings.typed_direction == 'RELATIVE'
        self.extent = settings.extent
        self.constraint = None          # None | ('AXIS', i) | ('PLANE', normal_i)
        self.snap = True
        self.num_text = ""              # typed distance; when non-empty it overrides the mouse
        self.mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
        self.mouse = self.mouse_start.copy()
        self.delta = Vector()
        self.target = self.pivot.copy()

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay, (self,), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "LMB/Enter: confirm   RMB/Esc: cancel   X/Y/Z: axis   "
            "Shift+X/Y/Z: plane   Ctrl: disable snap   "
            "Type: distance (a,b for planes)   Backspace: edit")
        self.update(context)
        return {'RUNNING_MODAL'}

    # -- geometry ----------------------------------------------------------

    def to_local(self, p):
        return self.rot_inv @ (p - self.origin)

    def to_world(self, p):
        return self.origin + self.rot @ p

    def active_axes(self):
        if self.constraint is None:
            return (0, 1, 2)
        kind, i = self.constraint
        return (i,) if kind == 'AXIS' else tuple(j for j in range(3) if j != i)

    def project(self, context, coord):
        """Mouse position -> orientation-space point on the current constraint line/plane."""
        region, rv3d = context.region, context.region_data
        ro = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        rd = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        p0 = self.pivot_world
        if self.constraint and self.constraint[0] == 'AXIS':
            axis = self.rot @ AXIS_VECS[self.constraint[1]]
            hit = geometry.intersect_line_line(p0, p0 + axis, ro, ro + rd)
            return self.to_local(hit[0]) if hit else None
        if self.constraint:
            normal = self.rot @ AXIS_VECS[self.constraint[1]]
        else:
            normal = rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
        hit = geometry.intersect_line_plane(ro, ro + rd, p0, normal)
        return self.to_local(hit) if hit else None

    def typed_delta(self, context, axes):
        """Comma-separated typed values mapped onto the active axes in X, Y, Z order.

        Returns None while the text doesn't evaluate (e.g. half-typed '1'6').
        """
        parts = self.num_text.split(",")
        if len(parts) > len(axes):
            return None
        # Relative: typed values follow the direction the mouse has dragged along each axis,
        # so "1" after dragging toward -X moves -1 (and "-1" moves back toward +X).
        signs = Vector((1.0, 1.0, 1.0))
        if self.relative_dir:
            mouse = self.mouse_delta(context, axes)
            if mouse is not None:
                for i in axes:
                    if mouse[i] < 0.0:
                        signs[i] = -1.0
        delta = Vector()
        for i, text in zip(axes, parts):
            text = text.strip()
            if text in {"", "-", "+"}:
                continue  # not typed yet -> 0
            value = _parse_length(context, text)
            if value is None:
                return None
            delta[i] = value * signs[i]
        return delta

    def mouse_delta(self, context, axes):
        """Unsnapped mouse movement on the active axes, or None if it can't be projected."""
        a = self.project(context, self.mouse_start)
        b = self.project(context, self.mouse)
        if a is None or b is None:
            return None
        delta = b - a
        for i in range(3):
            if i not in axes:
                delta[i] = 0.0
        return delta

    def update(self, context):
        axes = self.active_axes()

        if self.num_text:
            delta = self.typed_delta(context, axes)
            if delta is None:
                self.set_header(context, axes, invalid=True)
                return
            self.apply(context, delta, axes)
            return

        delta = self.mouse_delta(context, axes)
        if delta is None:
            return

        if self.snap:
            if self.absolute:
                target = self.pivot + delta
                for i in axes:
                    target[i] = self.grids[i].snap(target[i])
                delta = target - self.pivot
            else:
                for i in axes:
                    delta[i] = self.grids[i].snap(delta[i])

        self.apply(context, delta, axes)

    def apply(self, context, delta, axes):
        self.delta = delta
        self.target = self.pivot + delta
        self.world_delta = self.rot @ delta
        t = Matrix.Translation(self.world_delta)
        for o, mw in zip(self.objs, self.start_mw):
            o.matrix_world = t @ mw
        self.set_header(context, axes)

    def set_header(self, context, axes, invalid=False):
        label = "Free"
        if self.constraint:
            kind, i = self.constraint
            label = "XYZ"[i] if kind == 'AXIS' else "".join("XYZ"[j] for j in axes)
        text = f"Axis Grid Move [{label} {self.orient_name}]   "
        if self.num_text:
            text += f"Input: {self.num_text}|" + ("  (invalid)" if invalid else "") + "   "
        text += "   ".join(f"{'XYZ'[i]}: {_fmt(context, self.delta[i])}" for i in axes)
        if not self.snap and not self.num_text:
            text += "   (snap off)"
        context.area.header_text_set(text)

    # -- modal -------------------------------------------------------------

    def modal(self, context, event):
        context.area.tag_redraw()
        t = event.type

        if t in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN',
                 'TRACKPADZOOM'} or t.startswith('NDOF'):
            return {'PASS_THROUGH'}  # allow navigating while moving

        if t == 'MOUSEMOVE':
            self.mouse = Vector((event.mouse_region_x, event.mouse_region_y))
            self.snap = not event.ctrl
            self.update(context)

        elif t in {'LEFT_CTRL', 'RIGHT_CTRL'}:
            self.snap = event.value == 'RELEASE'
            self.update(context)

        elif t in {'X', 'Y', 'Z'} and event.value == 'PRESS':
            new = ('PLANE' if event.shift else 'AXIS', 'XYZ'.index(t))
            self.constraint = None if self.constraint == new else new
            self.update(context)

        elif t == 'BACK_SPACE' and event.value == 'PRESS' and self.num_text:
            # Ctrl+Backspace clears; emptying the input hands control back to the mouse.
            self.num_text = "" if event.ctrl else self.num_text[:-1]
            self.update(context)

        elif event.value == 'PRESS' and not event.ctrl and not event.alt and self.type_char(
                NUMPAD_CHARS.get(t) or event.unicode):
            self.update(context)

        elif t in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if self.num_text and self.typed_delta(context, self.active_axes()) is None:
                return {'RUNNING_MODAL'}  # don't confirm a half-typed value
            self.offset = self.world_delta
            self.cleanup(context)
            return {'FINISHED'}

        elif t in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            for o, mw in zip(self.objs, self.start_mw):
                o.matrix_world = mw
            self.cleanup(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def type_char(self, char):
        """Append a typed character to the numeric input; returns True if it was used."""
        if not char:
            return False
        allowed = NUM_CHARS if self.num_text else NUM_START_CHARS
        if char not in allowed:
            return False
        self.num_text += char
        return True

    def cleanup(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        context.workspace.status_text_set(None)
        context.area.tag_redraw()


# ---------------------------------------------------------------------------
# Presets + UI
# ---------------------------------------------------------------------------

class AXISGRID_OT_execute_preset(bpy.types.Operator):
    """Load a grid spacing preset"""
    bl_idname = "axis_grid.execute_preset"
    bl_label = "Load Spacing Preset"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})

    def execute(self, context):
        # Presets saved before Double Grid existed don't set it; treat them as single grids
        # rather than inheriting whatever the previous preset left switched on.
        try:
            with open(self.filepath, encoding="utf-8") as f:
                legacy = "use_double" not in f.read()
        except OSError:
            self.report({'ERROR'}, f"Preset not found: {self.filepath}")
            return {'CANCELLED'}
        if legacy:
            context.scene.axis_grid.use_double = False
        # Menu.path_menu only fills menu_idname for script.execute_preset, so pass it here.
        return bpy.ops.script.execute_preset(filepath=self.filepath, menu_idname="AXISGRID_MT_presets")


class AXISGRID_MT_presets(bpy.types.Menu):
    bl_label = "Spacing Presets"
    preset_subdir = "axis_grid_snap"
    preset_operator = "axis_grid.execute_preset"
    draw = bpy.types.Menu.draw_preset


class AXISGRID_OT_preset_add(AddPresetBase, bpy.types.Operator):
    """Save or remove a grid spacing preset"""
    bl_idname = "axis_grid.preset_add"
    bl_label = "Add Spacing Preset"
    preset_menu = "AXISGRID_MT_presets"
    preset_subdir = "axis_grid_snap"
    preset_defines = ["s = bpy.context.scene.axis_grid"]
    preset_values = ["s.step_x", "s.step_y", "s.step_z",
                     "s.use_double", "s.step_x2", "s.step_y2", "s.step_z2"]


class VIEW3D_PT_axis_grid(bpy.types.Panel):
    bl_label = "Axis Grid Snap"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Grid Snap"

    def draw(self, context):
        s = context.scene.axis_grid
        layout = self.layout
        layout.prop(s, "enabled")

        row = layout.row(align=True)
        row.menu("AXISGRID_MT_presets", text=AXISGRID_MT_presets.bl_label)
        row.operator("axis_grid.preset_add", text="", icon='ADD')
        row.operator("axis_grid.preset_add", text="", icon='REMOVE').remove_active = True

        layout.prop(s, "use_double")
        col = layout.column(align=True)
        if s.use_double:
            row = col.row(align=True)
            row.label(text="")
            row.label(text="A")
            row.label(text="B")
            for axis in "xyz":
                row = col.row(align=True)
                row.label(text=axis.upper())
                row.prop(s, f"step_{axis}", text="")
                row.prop(s, f"step_{axis}2", text="")
        else:
            col.label(text="Increment per axis:")
            col.prop(s, "step_x")
            col.prop(s, "step_y")
            col.prop(s, "step_z")

        layout.prop(s, "absolute")
        layout.label(text="Typed Direction:")
        layout.row().prop(s, "typed_direction", expand=True)
        layout.prop(s, "extent")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    AxisGridSettings,
    VIEW3D_OT_axis_grid_move,
    AXISGRID_OT_execute_preset,
    AXISGRID_MT_presets,
    AXISGRID_OT_preset_add,
    VIEW3D_PT_axis_grid,
)
addon_keymaps = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.axis_grid = PointerProperty(type=AxisGridSettings)

    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="Object Mode", space_type='EMPTY')
        kmi = km.keymap_items.new(VIEW3D_OT_axis_grid_move.bl_idname, 'G', 'PRESS')
        addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    del bpy.types.Scene.axis_grid
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
