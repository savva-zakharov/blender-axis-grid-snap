# Axis Grid Snap

A Blender extension (4.2+) for moving with a separate grid spacing on each axis, with on-screen rulers and grids, typed distances, and CAD-style point-to-point moves.

## Install

1. Get `axis_grid_snap-<version>.zip` from `dist/` (or build it, below).
2. In Blender: **Edit → Preferences → Get Extensions → ⌄ → Install from Disk…** and pick the zip.
   Dragging the zip into Blender also works.

## Build

```bash
blender --command extension validate axis_grid_snap
blender --command extension build --source-dir axis_grid_snap --output-dir dist
```

Bump `version` in `axis_grid_snap/blender_manifest.toml` (and `bl_info` in `__init__.py`) before each release.

## Use

Settings live in **3D View → Sidebar → Grid Snap**. The spacing and presets also appear at the bottom of the header's **Snapping** popover.

**G** (Object Mode and Mesh Edit Mode) starts an Axis Grid Move:

| Key | Action |
|---|---|
| X / Y / Z | lock to an axis (shows a ruler); press again to release |
| Shift+X / Y / Z | lock to a plane (shows a grid floor) |
| Ctrl | toggle snapping while held (follows the header magnet) |
| type a number | exact distance; `a,b` for both axes of a plane; units like `90cm`, `1'6"` work |
| A | typed values become absolute positions on the grid |
| M | CAD move: click a base point, then a target (snaps to vertices, surfaces, or the grid) |
| Enter / LMB | confirm |
| Esc / RMB | cancel |

Other features:

- **Spacing:** separate spacing per axis, optional double grid (alternating A/B spacing), and a major-line interval per axis. All of these save in presets.
- **Orientation and origin:** the move follows the active transform orientation. In Edit Mode the grid starts at the object origin.
- **Auto plane lock:** locks to a plane when the view or the selection normal is close to an axis.
- **Selection to Axis Grid** (Sidebar, or Object/Mesh → Snap): snaps origins or vertices to the grid.
- **Snap targets:** picking any native Snap Target in the Snapping popover switches Axis Grid off. Tick *Use Axis Grid for Move (G)* to switch it back on.
