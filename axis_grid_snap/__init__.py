# Packaged as a Blender extension (see blender_manifest.toml, which takes precedence).
# bl_info is kept so the folder still installs as a legacy add-on.
bl_info = {
    "name": "Axis Grid Snap",
    "author": "Savva",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Grid Snap  |  G in Object / Mesh Edit Mode",
    "description": "Move with separate grid spacing per axis, double grids, ruler and grid "
                   "overlays, typed and CAD-style point-to-point moves",
    "category": "3D View",
}

import math

import blf
import bmesh
import bpy
import gpu
import numpy as np
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

def _spread_major_every(settings):
    for axis in "xyz":
        setattr(settings, f"major_{axis}", settings.major_every)


MAJOR_DESC = ("Draw a coloured major line every N grid lines on this axis "
              "(every N A+B pairs on a double grid). 0 turns major lines off")


class AxisGridSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(
        name="Use for G",
        description="Replace the G key in Object and Mesh Edit Mode with Axis Grid Move. "
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
    major_x: IntProperty(name="X Major", default=5, min=0, soft_max=50, description=MAJOR_DESC)
    major_y: IntProperty(name="Y Major", default=5, min=0, soft_max=50, description=MAJOR_DESC)
    major_z: IntProperty(name="Z Major", default=5, min=0, soft_max=50, description=MAJOR_DESC)
    # Superseded by major_x/y/z; kept so presets that set it still load (sets all three).
    major_every: IntProperty(
        name="Major Every", default=5, min=0, options={'HIDDEN'},
        update=lambda self, _ctx: _spread_major_every(self),
    )
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
    snap_typed: BoolProperty(
        name="Snap Typed Values",
        description="Round typed distances to the nearest grid line, the same way mouse "
                    "movement snaps (follows the snap toggle and Ctrl). Off: typed values are exact",
        default=False,
    )
    auto_lock: BoolProperty(
        name="Auto Plane Lock",
        description="When moving without an axis lock and the view looks almost straight "
                    "along an axis (e.g. top view), lock to the plane facing the view",
        default=True,
    )
    auto_lock_source: EnumProperty(
        name="Lock From",
        description="What is compared with the axes to decide the automatic plane lock",
        items=(
            ('VIEW', "View", "Lock when the view looks almost straight along an axis"),
            ('NORMAL', "Normal",
             "Lock when the selected geometry's normal is close to an axis, so a face moves "
             "within its own plane (Edit Mode; Object Mode uses the view)"),
            ('BOTH', "Both",
             "Use the selection's normal when it is close to an axis, otherwise the view"),
        ),
        default='VIEW',
    )
    auto_lock_angle: FloatProperty(
        name="Within",
        description="How close the view must be to looking straight along an axis",
        default=math.radians(15.0), min=0.0, max=math.radians(45.0), subtype='ANGLE',
    )
    show_values: BoolProperty(
        name="Show Ruler Values",
        description="Label the ruler's major lines and the current position with their values",
        default=True,
    )
    extent: IntProperty(
        name="Overlay Extent",
        description="Number of increments drawn on each side of the pivot",
        default=20, min=2, max=200,
    )

    def grids(self):
        a = (self.step_x, self.step_y, self.step_z)
        b = (self.step_x2, self.step_y2, self.step_z2) if self.use_double else a
        major = (self.major_x, self.major_y, self.major_z)
        return tuple(AxisGrid(a[i], b[i], major[i]) for i in range(3))


class AxisGrid:
    """Grid lines along one axis with alternating spacings a, b (a == b is a plain grid).

    Line m sits at: m even -> (m/2)*(a+b), m odd -> that + a. So lines run
    ..., -(a+b), -b, 0, a, a+b, 2a+b, ... measured from the grid base.
    """

    def __init__(self, a, b, major_every=5):
        self.a, self.b = a, b
        self.major_every = major_every
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
        # Every Nth line on a single grid, every Nth A+B pair (2N lines) on a double one.
        if self.major_every <= 0:
            return False
        return m % (self.major_every * (2 if self.double else 1)) == 0


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

# All overlay geometry is built in the move's orientation space (see _orientation)
# and converted to world space just before drawing.

def _grid_base(op, axis):
    """Orientation-space coordinate of grid line 0 along an axis."""
    return 0.0 if op.grid_absolute else op.anchor[axis]


MAX_OVERLAY_LINES = 400  # per axis, so a long move can't build a huge overlay


class _Span:
    """Grid lines drawn along one axis. The overlay stays anchored at op.anchor (the move's
    start point, or the CAD base point); the range covers it and the current position plus `extent`
    lines either side, and lines fade with their distance from that stretch."""

    def __init__(self, op, axis, grid, extent):
        self.grid = grid
        self.extent = extent
        self.base = _grid_base(op, axis)
        self.origin = grid.nearest(op.anchor[axis] - self.base)   # line at the start point
        self.current = grid.nearest(op.target[axis] - self.base)  # line at the object now
        near, far = sorted((self.origin, self.current))
        self.lo, self.hi = near - extent, far + extent
        if self.hi - self.lo > MAX_OVERLAY_LINES:  # very long move: keep the area around the object
            half = MAX_OVERLAY_LINES // 2
            self.lo, self.hi = self.current - half, self.current + half
        self._near, self._far = near, far

    def coord(self, m):
        return self.base + self.grid.pos(m)

    def lines(self):
        return range(self.lo, self.hi + 1)

    def fade(self, m):
        gap = self._near - m if m < self._near else (m - self._far if m > self._far else 0)
        return max(0.0, 1.0 - gap / (self.extent + 1))

    def closeness(self, m):
        """Label priority: current position first, then nearest to it."""
        return abs(m - self.current)


def _ruler_frame(op, axis, grid, view_dir, extent):
    """Shared ruler geometry: (axis vector, tick direction, span,
    at(m) -> orientation-space point of grid line m on the ruler through the start point)."""
    a = AXIS_VECS[axis]
    perp = a.cross(view_dir)
    if perp.length < 1e-6:
        perp = AXIS_VECS[(axis + 1) % 3].copy()
    perp.normalize()
    span = _Span(op, axis, grid, extent)

    def at(m):
        p = op.anchor.copy()
        p[axis] = span.coord(m)
        return p

    return a, perp, span, at


def _ruler_lines(op, axis, grid, extent, view_dir):
    pos, col = [], []
    a, perp, span, at = _ruler_frame(op, axis, grid, view_dir, extent)
    r, g, b = AXIS_COLORS[axis]
    size = grid.scale

    # Faint infinite constraint line
    far = 10000.0
    pos += [op.anchor - a * far, op.anchor + a * far]
    col += [(r, g, b, 0.25)] * 2

    # Ruler spine, one segment per interval. On a double grid the A intervals are
    # drawn bright and the B intervals dim, so the alternation reads at a glance.
    for m in range(span.lo, span.hi):
        alpha = 0.9 if not grid.double or m % 2 == 0 else 0.35
        pos += [at(m), at(m + 1)]
        col += [(r, g, b, alpha)] * 2

    for m in span.lines():
        p = at(m)
        alpha = span.fade(m)
        if m == span.current:
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
    spans = {u: _Span(op, u, grids[u], extent), v: _Span(op, v, grids[v], extent)}

    def point(pu, pv):
        p = op.anchor.copy()
        p[u], p[v] = pu, pv
        return p

    # Lines running along u (one per v grid line) take v's axis colour, and vice versa,
    # so you can tell which spacing is which.
    for axis, other in ((v, u), (u, v)):
        span, other_span = spans[axis], spans[other]
        other_lo, other_hi = other_span.coord(other_span.lo), other_span.coord(other_span.hi)
        grid = grids[axis]
        r, g, b = AXIS_COLORS[axis]
        for m in span.lines():
            t = span.coord(m)
            fade = span.fade(m)
            if m == span.current:
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
    if rv3d is None or op.cad == 'BASE':
        return  # nothing moves while picking the CAD base point; draw_labels shows the marker
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
    pos += [op.anchor, op.target]
    col += [(1.0, 1.0, 1.0, 0.5)] * 2
    pos = [op.to_world(p) for p in pos]

    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": pos, "color": col})
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(1.0)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


CAD_MARKER_COLORS = {
    'VERTEX': (1.0, 0.6, 0.1, 1.0),   # orange square: snapped to a vertex / object origin
    'SURFACE': (0.3, 0.8, 1.0, 1.0),  # blue diamond: on a surface under the cursor
    'FREE': (0.85, 0.85, 0.85, 1.0),  # grey cross: in space (view or constraint plane)
}


def _draw_cad_markers(op):
    """CAD move markers in screen space: the picked base point (white) and what the
    cursor would snap to right now (coloured by snap kind)."""
    context = bpy.context
    region, rv3d = context.region, context.region_data
    if rv3d is None:
        return
    s = 6 * context.preferences.system.ui_scale
    lines = []  # (2D points as LINES pairs, colour)

    def marker(world, kind, color):
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, world)
        if p is None:
            return
        x, y = p
        if kind == 'VERTEX' or kind == 'BASE':
            pts = [(x - s, y - s), (x + s, y - s), (x + s, y - s), (x + s, y + s),
                   (x + s, y + s), (x - s, y + s), (x - s, y + s), (x - s, y - s)]
        elif kind == 'SURFACE':
            pts = [(x, y - s), (x + s, y), (x + s, y), (x, y + s),
                   (x, y + s), (x - s, y), (x - s, y), (x, y - s)]
        else:
            pts = [(x - s, y), (x + s, y), (x, y - s), (x, y + s)]
        lines.append((pts, color))

    if op.cad == 'TARGET':
        marker(op.cad_base_world, 'BASE', (1.0, 1.0, 1.0, 1.0))
    if op.cad_pick is not None:
        world, kind = op.cad_pick
        marker(world, kind, CAD_MARKER_COLORS[kind])

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    for pts, color in lines:
        shader.uniform_float("color", color)
        batch_for_shader(shader, 'LINES', {"pos": pts}).draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_edge_lengths(op):
    """Edge Length overlay for the edges adjacent to the moved vertices.

    Blender labels these itself only during its own transform; outside one it labels just
    selected edges (which it keeps doing here), so the unselected neighbours are added
    in the same theme colour. Lengths are in world space.
    """
    side_edges = getattr(op.mover, "side_edges", None)
    context = bpy.context
    space = context.space_data
    if (not side_edges or space is None or not space.overlay.show_overlays
            or not space.overlay.show_extra_edge_length):
        return
    region, rv3d = context.region, context.region_data
    if rv3d is None:
        return
    ui = context.preferences.system.ui_scale
    color = (*context.preferences.themes[0].view_3d.extra_edge_len, 1.0)
    font = 0
    blf.size(font, 11 * ui)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.8)
    blf.shadow_offset(font, 1, -1)
    blf.color(font, *color)
    for ob, edges in side_edges:
        mw = ob.matrix_world
        for e in edges:
            a, b = mw @ e.verts[0].co, mw @ e.verts[1].co
            p = view3d_utils.location_3d_to_region_2d(region, rv3d, (a + b) / 2)
            if p is None:
                continue  # behind the view
            text = _fmt(context, (b - a).length)
            w, h = blf.dimensions(font, text)
            blf.position(font, p.x - w / 2, p.y - h / 2, 0)
            blf.draw(font, text)
    blf.disable(font, blf.SHADOW)


def draw_labels(op):
    """Screen-space value labels for the ruler (axis moves) and grid floor (plane moves).

    Each labelled axis is read like a ruler through the move's start point (it stays put
    while the object moves): major lines are labelled on one side in the axis colour, the
    object's current value in white on the other.
    On the grid floor that gives two crossing rulers. Labels that would overlap one already
    drawn are skipped, nearest-to-current first, so zooming out thins them.
    """
    _draw_edge_lengths(op)
    if op.cad:
        _draw_cad_markers(op)
        if op.cad == 'BASE':
            return
    if not op.show_values or not op.constraint:
        return
    context = bpy.context
    region, rv3d = context.region, context.region_data
    if rv3d is None:
        return
    kind, axis = op.constraint
    view_dir = op.rot_inv @ (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))

    # (axis, offset direction for its labels, extent) for each labelled axis
    if kind == 'AXIS':
        _a, perp, _span, _at = _ruler_frame(op, axis, op.grids[axis], view_dir, op.extent)
        rulers = [(axis, perp, op.extent)]
    else:
        u, v = [i for i in range(3) if i != axis]
        extent = min(op.extent, 60)  # same as the drawn grid
        # Labels for one axis sit beside its line, pushed along the plane's other axis.
        rulers = [(u, AXIS_VECS[v], extent), (v, AXIS_VECS[u], extent)]

    font = 0
    ui = context.preferences.system.ui_scale
    blf.size(font, 11 * ui)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.8)
    blf.shadow_offset(font, 1, -1)
    drawn = []

    def place(at, grid, m, perp, side, color, tick_len):
        p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, op.to_world(at(m)))
        q2 = view3d_utils.location_3d_to_region_2d(
            region, rv3d, op.to_world(at(m) + perp * (tick_len * side)))
        if p2 is None or q2 is None:
            return  # behind the view
        d = q2 - p2
        d = d.normalized() if d.length > 1e-3 else Vector((0.0, float(side)))
        text = _fmt(context, grid.pos(m))
        w, h = blf.dimensions(font, text)
        # Push the label out past the tick end, by its half-extent in the push direction.
        c = q2 + d * (6 * ui + abs(d.x) * w / 2 + abs(d.y) * h / 2)
        rect = (c.x - w / 2 - 4, c.y - h / 2 - 2, c.x + w / 2 + 4, c.y + h / 2 + 2)
        if rect[2] < 0 or rect[0] > region.width or rect[3] < 0 or rect[1] > region.height:
            return
        if any(rect[0] < o[2] and o[0] < rect[2] and rect[1] < o[3] and o[1] < rect[3] for o in drawn):
            return
        drawn.append(rect)
        blf.color(font, *color)
        blf.position(font, c.x - w / 2, c.y - h / 2, 0)
        blf.draw(font, text)

    # Build every label first, then place current values before majors and nearer before
    # farther, so the most useful labels win when space runs out.
    jobs = []
    for ax, perp, extent in rulers:
        grid = op.grids[ax]
        _a, _p, span, at = _ruler_frame(op, ax, grid, view_dir, extent)
        r, g, b = AXIS_COLORS[ax]
        size = grid.scale
        jobs.append((-1, (at, grid, span.current, perp, -1, (1.0, 1.0, 1.0, 1.0), 0.6 * size)))
        for m in span.lines():
            if m != span.current and grid.is_major(m):
                color = (r, g, b, max(0.4, span.fade(m)))
                jobs.append((span.closeness(m), (at, grid, m, perp, 1, color, 0.35 * size)))
    for _order, args in sorted(jobs, key=lambda j: j[0]):
        place(*args)
    blf.disable(font, blf.SHADOW)


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


def _selection_normal(ob, unit=True):
    """World-space average normal of an edit mesh's selected faces (else vertices).

    With unit=False the summed, unnormalised vector is returned so several objects'
    normals can be combined. None when there is no usable normal.
    """
    if ob.type != 'MESH':
        return None
    bm = bmesh.from_edit_mesh(ob.data)
    faces = [f for f in bm.faces if f.select and not f.hide]
    elems = faces or [v for v in bm.verts if v.select and not v.hide]
    n = sum((e.normal for e in elems), Vector())
    if n.length < 1e-9:
        return None
    n = ob.matrix_world.to_3x3().inverted_safe().transposed() @ n  # normals use the inverse transpose
    return n.normalized() if unit else n


def _edit_normal_rotation(ob):
    """Approximation of Blender's Normal orientation for the edit-mesh selection.

    Z follows the selected faces' (else vertices') average normal. Y lies along the longest
    edge of the active face, if there is one, so a face's grid lines up with its edges.
    Returns None when the selection has no usable normal.
    """
    z = _selection_normal(ob)
    if z is None:
        return None
    bm = bmesh.from_edit_mesh(ob.data)
    mw3 = ob.matrix_world.to_3x3()

    y = None
    active = bm.select_history.active
    if isinstance(active, bmesh.types.BMFace) and active.select:
        edge = max(active.edges, key=lambda e: e.calc_length())
        y = mw3 @ (edge.verts[1].co - edge.verts[0].co)
        y -= z * y.dot(z)
    if y is None or y.length < 1e-9:
        return z.to_track_quat('Z', 'Y').to_matrix()
    y.normalize()
    return Matrix((y.cross(z), y, z)).transposed()


def _orientation(context, ob=None):
    """(label, rotation 3x3, grid origin) for the scene's current transform orientation.

    Snapping happens in this frame: X/Y/Z constraints, increments, typed values and the
    world-grid option all refer to its axes. The grid origin is the world origin in Object
    Mode, the edited object's origin in Edit Mode, and the 3D cursor for Cursor orientation.
    `ob` defaults to the active object.
    """
    scene = context.scene
    slot = scene.transform_orientation_slots[0]
    kind = slot.type
    ob = ob or context.active_object
    rv3d = context.region_data or getattr(context.space_data, "region_3d", None)
    if context.mode == 'EDIT_MESH' and ob is not None:
        origin = ob.matrix_world.translation.copy()
    else:
        origin = Vector()
    edit_normal = (_edit_normal_rotation(ob)
                   if kind == 'NORMAL' and context.mode == 'EDIT_MESH' and ob is not None else None)
    if edit_normal is not None:
        rot = edit_normal
    elif kind in {'LOCAL', 'NORMAL', 'GIMBAL'} and ob is not None:
        # In Object Mode Normal equals Local (as it does in Edit Mode with no usable
        # selection normal); Gimbal is approximated by Local.
        rot = ob.matrix_world.to_quaternion().to_matrix()
    elif kind == 'PARENT' and ob is not None and ob.parent is not None:
        rot = ob.parent.matrix_world.to_quaternion().to_matrix()
    elif kind == 'VIEW' and rv3d is not None:
        rot = rv3d.view_rotation.to_matrix()
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


# Metres per unit, and the symbol Blender shows, for Scene > Units > Length.
LENGTH_UNITS = {
    'KILOMETERS': (1000.0, "km"), 'METERS': (1.0, "m"), 'CENTIMETERS': (0.01, "cm"),
    'MILLIMETERS': (0.001, "mm"), 'MICROMETERS': (1e-6, "µm"),
    'MILES': (1609.344, "mi"), 'FEET': (0.3048, "'"), 'INCHES': (0.0254, '"'),
    'THOU': (0.0000254, "thou"),
}


def _fmt(context, value, decimals=4):
    """Format a length in the scene's chosen Length unit (not Blender's adaptive pick).

    Adaptive length or Separate Units fall back to Blender's own formatting.
    """
    us = context.scene.unit_settings
    if us.system == 'NONE':
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    metres = value * _unit_scale(us)
    unit = LENGTH_UNITS.get(us.length_unit)
    if unit is None or us.use_separate:
        try:
            return bpy.utils.units.to_string(us.system, 'LENGTH', metres, precision=decimals,
                                             split_unit=us.use_separate)
        except Exception:
            return f"{value:.{decimals}g}"
    factor, symbol = unit
    x = metres / factor
    if 0 < abs(x) < 1:  # keep ~4 significant digits for small values (e.g. 0.000675 m)
        decimals = min(8, decimals + math.ceil(-math.log10(abs(x))))
    number = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    if number == "-0":
        number = "0"
    return number + symbol if symbol in {"'", '"'} else f"{number} {symbol}"


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
# No 'a' (absolute-position toggle) and no x/y/z (axis locks); no length unit needs them.
NUM_CHARS = NUM_START_CHARS | set(",)*/ '\"") | set("bcdefghijklmnopqrstuvw")


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


# ---------------------------------------------------------------------------
# What gets moved: whole objects (Object Mode) or selected vertices (Edit Mode)
# ---------------------------------------------------------------------------

class ObjectMover:
    def __init__(self, context):
        self.objs = _root_objects(list(context.selected_editable_objects))
        self.start = [o.matrix_world.copy() for o in self.objs]
        self.active = context.active_object

    def __bool__(self):
        return bool(self.objs)

    def pivot(self):
        """Active object's origin, else the median of the moved origins."""
        if self.active in self.objs:
            return self.active.matrix_world.translation.copy()
        return sum((mw.translation for mw in self.start), Vector()) / len(self.start)

    def apply(self, world_delta):
        t = Matrix.Translation(world_delta)
        for o, mw in zip(self.objs, self.start):
            o.matrix_world = t @ mw

    def restore(self):
        for o, mw in zip(self.objs, self.start):
            o.matrix_world = mw


class MeshMover:
    """Selected vertices of every mesh in Edit Mode (multi-object editing included)."""

    def __init__(self, context):
        self.items = []  # (object, bmesh, verts, start coords, world->local 3x3)
        self.side_edges = []  # (object, edges with one end moving)
        self.active = None
        for ob in context.objects_in_mode_unique_data:
            if ob.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(ob.data)
            verts = [v for v in bm.verts if v.select and not v.hide]
            if not verts:
                continue
            self.items.append((ob, bm, verts, [v.co.copy() for v in verts],
                               ob.matrix_world.to_3x3().inverted_safe()))
            # Unselected edges touching the selection: they stretch during the move, and
            # Blender's Edge Length overlay only labels them inside its own transform.
            side = {e for v in verts for e in v.link_edges if not e.select and not e.hide}
            if side:
                self.side_edges.append((ob, list(side)))
            if ob == context.active_object:
                self.active = (ob, bm.select_history.active)

    def __bool__(self):
        return bool(self.items)

    def pivot(self):
        """Active vertex/edge/face (its median), else the median of all selected vertices."""
        if self.active and self.active[1] is not None and self.active[1].select:
            ob, elem = self.active
            vs = [elem] if isinstance(elem, bmesh.types.BMVert) else list(elem.verts)
            return ob.matrix_world @ (sum((v.co for v in vs), Vector()) / len(vs))
        total, count = Vector(), 0
        for ob, _bm, verts, start, _inv in self.items:
            mw = ob.matrix_world
            for co in start:
                total += mw @ co
            count += len(start)
        return total / count

    def apply(self, world_delta):
        for ob, _bm, verts, start, inv in self.items:
            local = inv @ world_delta
            for v, co in zip(verts, start):
                v.co = co + local
            bmesh.update_edit_mesh(ob.data, loop_triangles=True, destructive=False)

    def restore(self):
        self.apply(Vector())


def _snap_world(point, grids, rot, origin):
    """Snap a world-space point onto the grid on all three axes of the given frame."""
    local = rot.transposed() @ (point - origin)
    local = Vector([grids[i].snap(local[i]) for i in range(3)])
    return origin + rot @ local


class VIEW3D_OT_axis_grid_snap_selection(bpy.types.Operator):
    """Snap each selected object origin (Object Mode) or vertex (Edit Mode) to the nearest
    point of the current Axis Grid, using the active transform orientation"""
    bl_idname = "view3d.axis_grid_snap_selection"
    bl_label = "Selection to Axis Grid"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH'}

    def execute(self, context):
        grids = context.scene.axis_grid.grids()
        count = 0
        if context.mode == 'EDIT_MESH':
            # Each mesh snaps on its own grid, starting at that object's origin.
            for ob in context.objects_in_mode_unique_data:
                if ob.type != 'MESH':
                    continue
                _name, rot, origin = _orientation(context, ob)
                mw = ob.matrix_world
                mw_inv = mw.inverted_safe()
                bm = bmesh.from_edit_mesh(ob.data)
                verts = [v for v in bm.verts if v.select and not v.hide]
                for v in verts:
                    v.co = mw_inv @ _snap_world(mw @ v.co, grids, rot, origin)
                count += len(verts)
                bmesh.update_edit_mesh(ob.data, loop_triangles=True, destructive=False)
            what = "vertices"
        else:
            _name, rot, origin = _orientation(context)
            objs = _root_objects(list(context.selected_editable_objects))
            for o in objs:
                loc = o.matrix_world.translation
                o.matrix_world = Matrix.Translation(_snap_world(loc, grids, rot, origin) - loc) @ o.matrix_world
            count = len(objs)
            what = "objects"
        if not count:
            self.report({'WARNING'}, "Nothing selected")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Snapped {count} {what} to the Axis Grid")
        return {'FINISHED'}


def _snap_menu_entry(self, context):
    self.layout.separator()
    self.layout.operator(VIEW3D_OT_axis_grid_snap_selection.bl_idname, icon='SNAP_GRID')


class SnapCloud:
    """World-space points the CAD move can snap to, gathered once when CAD mode starts.

    Visible mesh vertices (evaluated, so modifiers and instances count) and the origins of
    other visible objects. Points on the geometry being moved are kept separately: they are
    valid base points (grab the selection by its own corner) but not targets.
    """

    def __init__(self, context, mover):
        static, moving = [], []
        depsgraph = context.evaluated_depsgraph_get()
        edit_objs = set()

        if isinstance(mover, MeshMover):
            for ob, bm, _verts, _start, _inv in mover.items:
                edit_objs.add(ob)
                mw = ob.matrix_world
                for v in bm.verts:
                    if not v.hide:
                        (moving if v.select else static).append(tuple(mw @ v.co))
            moving_objs = set()
        else:
            moving_objs = set(mover.objs)

        def is_moving(ob):
            while ob is not None:  # children follow a moved parent
                if ob in moving_objs:
                    return True
                ob = ob.parent
            return False

        arrays_static, arrays_moving = [], []
        for inst in depsgraph.object_instances:
            ob = inst.object
            orig = ob.original
            if orig in edit_objs:
                continue  # taken from the edit mesh above
            if not inst.is_instance and not orig.visible_get():
                continue
            parent = inst.parent.original if inst.is_instance and inst.parent else None
            moves = is_moving(orig) or (parent is not None and is_moving(parent))
            mw = np.array(inst.matrix_world, dtype=np.float64)
            if ob.type == 'MESH' and len(ob.data.vertices):
                co = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
                ob.data.vertices.foreach_get("co", co)
                co = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
                (arrays_moving if moves else arrays_static).append(co)
            else:
                (moving if moves else static).append(tuple(mw[:3, 3]))

        def stack(points, arrays):
            parts = arrays + ([np.array(points, dtype=np.float64)] if points else [])
            return np.concatenate(parts) if parts else np.empty((0, 3))

        self.static = stack(static, arrays_static)
        self.moving = stack(moving, arrays_moving)
        self.moving_objs = moving_objs

    def nearest(self, region, rv3d, coord, radius, include_moving):
        """Closest point to the mouse on screen within `radius` pixels, or None."""
        pts = np.concatenate((self.static, self.moving)) if include_moving else self.static
        if not len(pts):
            return None
        m = np.array(rv3d.perspective_matrix, dtype=np.float64)
        clip = pts @ m[:, :3].T + m[:, 3]
        w = clip[:, 3]
        visible = w > 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            sx = (clip[:, 0] / w * 0.5 + 0.5) * region.width
            sy = (clip[:, 1] / w * 0.5 + 0.5) * region.height
        d2 = (sx - coord[0]) ** 2 + (sy - coord[1]) ** 2
        d2[~visible] = np.inf
        i = int(np.argmin(d2))
        if d2[i] > radius * radius:
            return None
        return Vector(pts[i])


def _make_mover(context):
    return MeshMover(context) if context.mode == 'EDIT_MESH' else ObjectMover(context)


class VIEW3D_OT_axis_grid_move(bpy.types.Operator):
    """Move selected objects or mesh elements, snapping each axis to its own increment"""
    bl_idname = "view3d.axis_grid_move"
    bl_label = "Axis Grid Move"
    bl_options = {'REGISTER', 'UNDO'}

    offset: FloatVectorProperty(name="Offset", subtype='TRANSLATION', size=3)

    @classmethod
    def poll(cls, context):
        return (context.mode in {'OBJECT', 'EDIT_MESH'}
                and context.area is not None and context.area.type == 'VIEW_3D'
                and context.region is not None and context.region.type == 'WINDOW')

    # Redo panel / scripted use: apply the stored offset directly.
    def execute(self, context):
        mover = _make_mover(context)
        if mover:
            mover.apply(Vector(self.offset))
        return {'FINISHED'}

    def invoke(self, context, event):
        settings = context.scene.axis_grid
        if not settings.enabled:
            return {'PASS_THROUGH'}  # let Blender's own G handle it

        self.mover = _make_mover(context)
        if not self.mover:
            return {'PASS_THROUGH'}
        pivot_world = self.mover.pivot()

        # pivot / target / delta are kept in orientation space; see to_world().
        self.orient_name, self.rot, self.origin = _orientation(context)
        self.rot_inv = self.rot.transposed()
        self.pivot_world = pivot_world
        self.pivot = self.to_local(pivot_world)

        self.grids = settings.grids()
        self.absolute = settings.absolute
        self.relative_dir = settings.typed_direction == 'RELATIVE'
        self.snap_typed = settings.snap_typed
        self.extent = settings.extent
        self.show_values = settings.show_values
        self.constraint = None          # None | ('AXIS', i) | ('PLANE', normal_i); in effect now
        self.user_constraint = None     # set by X/Y/Z keys; overrides the automatic lock
        self.auto_locked = False
        self.auto_angle = settings.auto_lock_angle if settings.auto_lock else None
        # Lock From Normal / Both: compare the selection's normal (Edit Mode only; without a
        # usable selection normal, e.g. in Object Mode, the view is used).
        self.auto_source = settings.auto_lock_source
        self.auto_normal = None
        if self.auto_source in {'NORMAL', 'BOTH'} and context.mode == 'EDIT_MESH':
            total = sum((n for n in (_selection_normal(ob, unit=False)
                                     for ob in context.objects_in_mode_unique_data)
                         if n is not None), Vector())
            if total.length > 1e-9:
                self.auto_normal = total.normalized()
        self.snap_toggle = context.scene.tool_settings.use_snap  # the header's magnet button
        self.snap = self.snap_toggle
        self.num_text = ""              # typed distance; when non-empty it overrides the mouse
        self.abs_coords = False         # A key: typed values are grid positions, not distances
        self.mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
        self.mouse = self.mouse_start.copy()
        self.delta = Vector()
        self.target = self.pivot.copy()
        # The point offsets are measured from: the pivot, or the CAD base point once picked.
        self.anchor = self.pivot.copy()
        self.anchor_world = pivot_world.copy()
        self.grid_absolute = self.absolute
        # CAD move (M key): None, 'BASE' (picking the base point) or 'TARGET'.
        self.cad = None
        self.cad_cloud = None           # SnapCloud, built on first use
        self.cad_pick = None            # (world point, 'VERTEX'|'SURFACE'|'FREE') under the cursor
        self.cad_base_world = None
        self.cad_exact = False          # target landed on a vertex: don't grid-round it

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay, (self,), 'WINDOW', 'POST_VIEW')
        self._handle_px = bpy.types.SpaceView3D.draw_handler_add(
            draw_labels, (self,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "LMB/Enter: confirm   RMB/Esc: cancel   X/Y/Z: axis   "
            "Shift+X/Y/Z: plane   Ctrl: toggle snap   "
            "Type: distance (a,b for planes)   A: typed = absolute position   Backspace: edit   "
            "M: CAD move (click base point, then target)")
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
        """Mouse position -> orientation-space point on the current constraint line/plane
        through the anchor (the pivot, or the CAD base point)."""
        region, rv3d = context.region, context.region_data
        ro = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        rd = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        p0 = self.anchor_world
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
        # Absolute coordinates (A key) are positions, so the mouse direction doesn't apply.
        signs = Vector((1.0, 1.0, 1.0))
        if self.relative_dir and not self.abs_coords:
            mouse = self.mouse_delta(context, axes)
            if mouse is not None:
                for i in axes:
                    if mouse[i] < 0.0:
                        signs[i] = -1.0
        delta = Vector()
        for i, text in zip(axes, parts):
            text = text.strip()
            if text in {"", "-", "+"}:
                continue  # not typed yet -> stays where it is on this axis
            value = _parse_length(context, text)
            if value is None:
                return None
            if self.abs_coords:
                # Position on the grid, measured from the grid origin in orientation space
                # (the anchor is already relative to that origin). In CAD mode it is the
                # base point that lands there.
                delta[i] = value - self.anchor[i]
            else:
                delta[i] = value * signs[i]
        return delta

    def mouse_delta(self, context, axes):
        """Unsnapped mouse movement on the active axes, or None if it can't be projected."""
        if self.cad == 'TARGET':
            return self.cad_delta(context, axes)
        a = self.project(context, self.mouse_start)
        b = self.project(context, self.mouse)
        if a is None or b is None:
            return None
        delta = b - a
        for i in range(3):
            if i not in axes:
                delta[i] = 0.0
        return delta

    # -- CAD move ------------------------------------------------------------

    def cad_pick_point(self, context, include_moving, allow_surface):
        """What a click would pick: nearest vertex/origin on screen, else the surface under
        the cursor, else a point on the view/constraint plane through the anchor.
        Returns (world point, kind) or None."""
        region, rv3d = context.region, context.region_data
        radius = 12 * context.preferences.system.ui_scale
        p = self.cad_cloud.nearest(region, rv3d, self.mouse, radius, include_moving)
        if p is not None:
            return p, 'VERTEX'
        if allow_surface:
            ro = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse)
            rd = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse)
            hit, loc, _n, _i, obj, _m = context.scene.ray_cast(
                context.evaluated_depsgraph_get(), ro, rd)
            if hit and (include_moving or not self.cad_is_moving(obj)):
                return loc.copy(), 'SURFACE'
        local = self.project(context, self.mouse)
        return (self.to_world(local), 'FREE') if local is not None else None

    def cad_is_moving(self, obj):
        """Whether a ray-cast hit belongs to geometry being moved (not a valid target)."""
        if isinstance(self.mover, MeshMover):
            # Edit Mode: the edited meshes contain the moving selection; skip them to be safe.
            return obj.original in {item[0] for item in self.mover.items}
        ob = obj.original
        while ob is not None:
            if ob in self.cad_cloud.moving_objs:
                return True
            ob = ob.parent
        return False

    def cad_delta(self, context, axes):
        """Offset from the base point to the picked target, on the active axes."""
        # Surface hits only make sense for a free move; with a lock the target is
        # projected onto the axis/plane through the base point instead.
        self.cad_pick = self.cad_pick_point(context, include_moving=False,
                                            allow_surface=self.constraint is None)
        if self.cad_pick is None:
            return None
        world, kind = self.cad_pick
        self.cad_exact = kind == 'VERTEX'
        delta = self.to_local(world) - self.anchor
        for i in range(3):
            if i not in axes:
                delta[i] = 0.0
        return delta

    def enter_cad(self, context):
        if self.cad_cloud is None:
            self.cad_cloud = SnapCloud(context, self.mover)
        self.mover.restore()
        self.cad = 'BASE'
        self.num_text = ""
        self.update(context)

    def exit_cad(self, context):
        self.cad = None
        self.cad_pick = None
        self.cad_exact = False
        self.anchor = self.pivot.copy()
        self.anchor_world = self.pivot_world.copy()
        self.update(context)

    def set_cad_base(self, context):
        world, _kind = self.cad_pick
        self.cad_base_world = world.copy()
        self.anchor_world = world.copy()
        self.anchor = self.to_local(world)
        self.cad = 'TARGET'
        self.update(context)

    def auto_constraint(self, context):
        """Plane lock for a free move when the view looks almost straight along an axis.

        Looking down Z (within the angle setting) -> XY plane, etc., using the move's
        orientation axes. None when disabled or no axis is close enough.
        """
        if self.auto_angle is None:
            return None
        if self.auto_normal is not None:  # fixed for the move: translating keeps normals
            lock = self.plane_facing(self.auto_normal)
            if lock is not None or self.auto_source == 'NORMAL':
                return lock
            # 'BOTH': the normal isn't near an axis, so fall back to the view
        rv3d = context.region_data
        if rv3d is None:
            return None
        return self.plane_facing(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))

    def plane_facing(self, direction):
        """('PLANE', i) if the world direction is within the auto-lock angle of axis i."""
        direction = self.rot_inv @ direction
        i = max(range(3), key=lambda k: abs(direction[k]))
        if abs(direction[i]) >= math.cos(self.auto_angle):
            return ('PLANE', i)
        return None

    def update(self, context):
        # The user's X/Y/Z choice wins; otherwise re-check the view, which can orbit mid-move.
        self.auto_locked = False
        if self.user_constraint is not None:
            self.constraint = self.user_constraint
        else:
            self.constraint = self.auto_constraint(context)
            self.auto_locked = self.constraint is not None
        axes = self.active_axes()

        if self.cad == 'BASE':  # nothing moves yet; just track what a click would pick
            self.cad_pick = self.cad_pick_point(context, include_moving=True, allow_surface=True)
            self.set_header(context, axes)
            return

        if self.num_text:
            delta = self.typed_delta(context, axes)
            if delta is None:
                self.set_header(context, axes, invalid=True)
                return
            snap = self.snap and self.snap_typed  # typed values are exact unless opted in
        else:
            delta = self.mouse_delta(context, axes)
            if delta is None:
                return
            # A CAD target picked on a vertex is exact; anything else follows the grid.
            snap = self.snap and not (self.cad and self.cad_exact)

        if snap:
            # CAD offsets count grid steps from the base point, not world grid lines.
            if self.absolute and not self.cad:
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
        self.target = self.anchor + delta
        self.grid_absolute = self.absolute and not self.cad
        self.world_delta = self.rot @ delta
        self.mover.apply(self.world_delta)
        self.set_header(context, axes)

    def set_header(self, context, axes, invalid=False):
        label = "Free"
        if self.constraint:
            kind, i = self.constraint
            label = "XYZ"[i] if kind == 'AXIS' else "".join("XYZ"[j] for j in axes)
            if self.auto_locked:
                label += " auto"
        if self.cad == 'BASE':
            kind = {'VERTEX': "vertex", 'SURFACE': "surface", 'FREE': "in space"}.get(
                self.cad_pick[1] if self.cad_pick else None, "nothing")
            context.area.header_text_set(
                f"Axis Grid Move (CAD) [{label} {self.orient_name}]   Click the base point "
                f"(under cursor: {kind})   M: back to normal move")
            return
        cad = " (CAD)" if self.cad else ""
        text = f"Axis Grid Move{cad} [{label} {self.orient_name}]   "
        mode = " (absolute)" if self.abs_coords else ""
        if self.num_text or self.abs_coords:
            text += f"Input{mode}: {self.num_text}|" + ("  (invalid)" if invalid else "") + "   "
        if self.abs_coords and self.num_text:
            # Show where it lands on the grid, then the distance moved.
            text += "   ".join(f"{'XYZ'[i]} at {_fmt(context, self.target[i])} "
                               f"(moved {_fmt(context, self.delta[i])})" for i in axes)
        else:
            text += "   ".join(f"{'XYZ'[i]}: {_fmt(context, self.delta[i])}" for i in axes)
        if self.num_text:
            if self.snap and self.snap_typed:
                text += "   (snapped to grid)"
        elif self.cad and self.cad_exact:
            text += "   (on vertex)"
        elif not self.snap:
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
            self.snap = self.snap_toggle != event.ctrl
            self.update(context)

        elif t in {'LEFT_CTRL', 'RIGHT_CTRL'}:
            # Like native snapping: Ctrl inverts the header's snap (magnet) toggle while held.
            self.snap = self.snap_toggle != (event.value == 'PRESS')
            self.update(context)

        elif t in {'X', 'Y', 'Z'} and event.value == 'PRESS':
            new = ('PLANE' if event.shift else 'AXIS', 'XYZ'.index(t))
            # Pressing the same key again releases the lock (back to free / auto lock).
            self.user_constraint = None if self.user_constraint == new else new
            self.update(context)

        elif t == 'A' and event.value == 'PRESS' and not (event.ctrl or event.alt or event.shift):
            self.abs_coords = not self.abs_coords
            self.update(context)

        # M toggles the CAD move, unless a value is being typed ("m" is part of cm/mm units).
        elif (t == 'M' and event.value == 'PRESS' and not self.num_text
              and not (event.ctrl or event.alt or event.shift)):
            if self.cad:
                self.exit_cad(context)
            else:
                self.enter_cad(context)

        elif self.cad == 'BASE':
            if t == 'LEFTMOUSE' and event.value == 'PRESS' and self.cad_pick is not None:
                self.set_cad_base(context)
            elif t in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                self.mover.restore()
                self.cleanup(context)
                return {'CANCELLED'}
            # Everything else (typing, Enter, ...) waits until the base point is picked.

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
            self.mover.restore()
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
        bpy.types.SpaceView3D.draw_handler_remove(self._handle_px, 'WINDOW')
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
        # Presets saved before a setting existed don't set it. Reset those settings to their
        # defaults rather than inheriting whatever the previous preset left behind.
        try:
            with open(self.filepath, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            self.report({'ERROR'}, f"Preset not found: {self.filepath}")
            return {'CANCELLED'}
        s = context.scene.axis_grid
        for prop in ("use_double", "major_x", "major_y", "major_z"):
            if f"s.{prop} " not in text:
                s.property_unset(prop)
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
                     "s.use_double", "s.step_x2", "s.step_y2", "s.step_z2",
                     "s.major_x", "s.major_y", "s.major_z"]


def _draw_spacing(layout, s):
    """Presets + per-axis spacing fields; shared by the sidebar panel and the snap popover."""
    row = layout.row(align=True)
    row.menu("AXISGRID_MT_presets", text=AXISGRID_MT_presets.bl_label)
    row.operator("axis_grid.preset_add", text="", icon='ADD')
    row.operator("axis_grid.preset_add", text="", icon='REMOVE').remove_active = True

    layout.prop(s, "use_double")

    # Table: axis | spacing (A, B on a double grid) | major line interval
    def narrow(row, units):
        sub = row.row(align=True)
        sub.ui_units_x = units
        return sub

    col = layout.column(align=True)
    row = col.row(align=True)
    narrow(row, 1).label(text="")
    if s.use_double:
        row.label(text="A")
        row.label(text="B")
    else:
        row.label(text="Spacing")
    narrow(row, 3).label(text="Major")
    for axis in "xyz":
        row = col.row(align=True)
        narrow(row, 1).label(text=axis.upper())
        row.prop(s, f"step_{axis}", text="")
        if s.use_double:
            row.prop(s, f"step_{axis}2", text="")
        narrow(row, 3).prop(s, f"major_{axis}", text="")


def _draw_snap_popover(self, context):
    """Axis Grid section appended to the header's Snapping popover."""
    if context.mode not in {'OBJECT', 'EDIT_MESH'}:
        return
    s = context.scene.axis_grid
    layout = self.layout
    layout.separator()
    layout.label(text="Axis Grid")
    layout.prop(s, "enabled", text="Use Axis Grid for Move (G)")
    if s.enabled:
        _draw_spacing(layout, s)
    else:
        layout.label(text="Off while using a Snap Target above", icon='INFO')


class VIEW3D_PT_axis_grid(bpy.types.Panel):
    bl_label = "Axis Grid Snap"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Grid Snap"

    def draw(self, context):
        s = context.scene.axis_grid
        layout = self.layout
        layout.prop(s, "enabled")
        _draw_spacing(layout, s)
        layout.prop(s, "absolute")
        layout.label(text="Typed Direction:")
        layout.row().prop(s, "typed_direction", expand=True)
        layout.prop(s, "snap_typed")
        row = layout.row(align=True)
        row.prop(s, "auto_lock")
        sub = row.row(align=True)
        sub.active = s.auto_lock
        sub.prop(s, "auto_lock_angle")
        row = layout.row()
        row.active = s.auto_lock
        row.prop(s, "auto_lock_source", expand=True)
        layout.prop(s, "extent")
        layout.prop(s, "show_values")

        layout.separator()
        layout.operator(VIEW3D_OT_axis_grid_snap_selection.bl_idname, icon='SNAP_GRID')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Picking any native Snap Target in the Snapping popover switches the Axis Grid off, so
# the two work like a choice: either native snapping or Axis Grid. Ticking "Use Axis Grid"
# turns it back on. (msgbus only reports changes made through the UI, not from scripts.)
SNAP_TARGET_PROPS = ("snap_elements", "snap_elements_base", "snap_elements_individual")
_msgbus_owner = object()


def _on_snap_target_changed():
    scene = bpy.context.scene
    if scene is not None and scene.axis_grid.enabled:
        scene.axis_grid.enabled = False


def _subscribe_snap_targets():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    for prop in SNAP_TARGET_PROPS:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.ToolSettings, prop),
            owner=_msgbus_owner,
            args=(),
            notify=_on_snap_target_changed,
        )


@bpy.app.handlers.persistent
def _on_load_post(_dummy):
    _subscribe_snap_targets()  # opening a file drops all msgbus subscriptions


classes = (
    AxisGridSettings,
    VIEW3D_OT_axis_grid_move,
    VIEW3D_OT_axis_grid_snap_selection,
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
    bpy.types.VIEW3D_MT_snap.append(_snap_menu_entry)  # Object/Mesh > Snap menu
    bpy.types.VIEW3D_PT_snapping.append(_draw_snap_popover)  # header magnet popover
    _subscribe_snap_targets()
    bpy.app.handlers.load_post.append(_on_load_post)

    kc =bpy.context.window_manager.keyconfigs.addon
    if kc:
        for name in ("Object Mode", "Mesh"):
            km = kc.keymaps.new(name=name, space_type='EMPTY')
            kmi = km.keymap_items.new(VIEW3D_OT_axis_grid_move.bl_idname, 'G', 'PRESS')
            addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()
    bpy.types.VIEW3D_MT_snap.remove(_snap_menu_entry)
    bpy.types.VIEW3D_PT_snapping.remove(_draw_snap_popover)
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    del bpy.types.Scene.axis_grid
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
