# mjlab/sensor/warp_hybrid_raycast_sensor.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import quat_from_matrix
from mjlab.sensor.raycast_sensor import RayCastData

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    # q = [w, x, y, z] in MuJoCo
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=torch.float32)


def extract_geom_mesh(
    mj_model: mujoco.MjModel,
    geom_id: int,
    export_path: str | None = None,
    export_format: Literal["obj", "ply"] = "obj",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract world-space triangle mesh from MuJoCo geom.

    Supports mjGEOM_MESH, mjGEOM_HFIELD, mjGEOM_BOX, mjGEOM_PLANE.
    Returned vertices are in WORLD coordinates.
    """
    geom_type = mj_model.geom_type[geom_id]
    geom_pos = torch.tensor(mj_model.geom_pos[geom_id], dtype=torch.float32)
    geom_quat = torch.tensor(mj_model.geom_quat[geom_id], dtype=torch.float32)
    geom_rot = quat_to_matrix(geom_quat)  # 3x3

    # ---------------- Mesh ----------------
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = mj_model.geom_dataid[geom_id]
        vadr = mj_model.mesh_vertadr[mesh_id]
        vnum = mj_model.mesh_vertnum[mesh_id]
        fadr = mj_model.mesh_faceadr[mesh_id]
        fnum = mj_model.mesh_facenum[mesh_id]

        verts_local = torch.tensor(
            mj_model.mesh_vert[vadr : vadr + 3 * vnum].reshape(-1, 3),
            dtype=torch.float32,
        )
        verts_world = (geom_rot @ verts_local.T).T + geom_pos

        faces = torch.tensor(
            mj_model.mesh_face[fadr : fadr + 3 * fnum].reshape(-1, 3),
            dtype=torch.int32,
        )
        return verts_world, faces

    # ---------------- Height Field ----------------
    if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
        hfield_id = mj_model.geom_dataid[geom_id]
        nrow = mj_model.hfield_nrow[hfield_id]
        ncol = mj_model.hfield_ncol[hfield_id]
        size = mj_model.hfield_size[hfield_id]  # [sx, sy, sz, ...]
        adr = mj_model.hfield_adr[hfield_id]

        heights = torch.tensor(
            mj_model.hfield_data[adr : adr + nrow * ncol],
            dtype=torch.float32,
        ).view(nrow, ncol)

        xs = torch.linspace(-size[0], size[0], ncol)
        ys = torch.linspace(-size[1], size[1], nrow)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        z = heights * size[2]

        verts_local = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z.reshape(-1)], dim=1)
        verts_world = (geom_rot @ verts_local.T).T + geom_pos

        faces = []
        for i in range(nrow - 1):
            for j in range(ncol - 1):
                v0 = i * ncol + j
                v1 = v0 + 1
                v2 = v0 + ncol
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = torch.tensor(faces, dtype=torch.int32)
        return verts_world, faces

    # ---------------- Box ----------------
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = torch.tensor(mj_model.geom_size[geom_id], dtype=torch.float32)
        signs = torch.tensor([
            [-1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [-1,  1,  1],
            [ 1, -1, -1],
            [ 1, -1,  1],
            [ 1,  1, -1],
            [ 1,  1,  1],
        ], dtype=torch.float32)
        verts_local = signs * size

        faces = torch.tensor([
            [0, 1, 2], [1, 3, 2],  # -X
            [4, 6, 5], [5, 6, 7],  # +X
            [0, 4, 1], [1, 4, 5],  # -Y
            [2, 3, 6], [3, 7, 6],  # +Y
            [0, 2, 4], [2, 6, 4],  # -Z
            [1, 5, 3], [3, 5, 7],  # +Z
        ], dtype=torch.int32)

        verts_world = (geom_rot @ verts_local.T).T + geom_pos
        return verts_world, faces

    # ---------------- Plane ----------------
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        sx, sy = mj_model.geom_size[geom_id][:2]
        
        # MuJoCo planes are infinite by default (size may be 0)
        # Use a large bounded mesh for raycast BVH
        if sx == 0.0:
            sx = 500.0  # Default large plane size
        if sy == 0.0:
            sy = 500.0
            
        verts_local = torch.tensor([
            [-sx, -sy, 0],
            [ sx, -sy, 0],
            [-sx,  sy, 0],
            [ sx,  sy, 0],
        ], dtype=torch.float32)

        faces = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=torch.int32)

        verts_world = (geom_rot @ verts_local.T).T + geom_pos
        print(f"[extract_geom_mesh] Info: extracted plane geom as bounded mesh ({sx*2:.1f} x {sy*2:.1f}m)")
        return verts_world, faces

    raise NotImplementedError(f"Unsupported geom type {geom_type}")


RayAlignment = Literal["base", "yaw", "world"]

vec6 = wp.types.vector(length=6, dtype=float)



@dataclass
class WarpRayCastSensorCfg(SensorCfg):

    @dataclass
    class VizCfg:
        """Visualization settings for debug rendering."""

        hit_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8)
        """RGBA color for rays that hit a surface."""

        miss_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.4)
        """RGBA color for rays that miss."""

        hit_sphere_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
        """RGBA color for spheres drawn at hit points."""

        hit_sphere_radius: float = 0.5
        """Radius of spheres drawn at hit points (multiplier of meansize)."""

        show_rays: bool = False
        """Whether to draw ray arrows."""

        show_normals: bool = False
        """Whether to draw surface normals at hit points."""

        normal_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
        """RGBA color for surface normal arrows."""

        normal_length: float = 5.0
        """Length of surface normal arrows (multiplier of meansize)."""

    frame: ObjRef
    pattern: object
    ray_alignment: RayAlignment = "base"
    max_distance: float = 10.0

    exclude_parent_body: bool = True
    include_geom_groups: tuple[int, ...] | None = None

    debug_vis: bool = False
    viz: VizCfg = field(default_factory=VizCfg)
    """Visualization settings."""

    export_mesh_path: str | None = None
    """Optional path to export the built mesh for debugging (e.g., '/tmp/warp_mesh.obj')."""

    def build(self) -> "WarpRayCastSensor":
        return WarpRayCastSensor(self)


# ============================================================
# Warp BVH kernel (mesh / hfield)
# ============================================================

@wp.kernel
def _raycast_mesh_kernel(
    mesh_id: wp.uint64,
    ray_o: wp.array(dtype=wp.vec3),
    ray_d: wp.array(dtype=wp.vec3),
    max_dist: float,
    out_dist: wp.array(dtype=float),
    out_n: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    q = wp.mesh_query_ray(mesh_id, ray_o[tid], ray_d[tid], max_dist)
    if q.t < max_dist:
        out_dist[tid] = q.t
        out_n[tid] = q.normal
    else:
        out_dist[tid] = -1.0
        out_n[tid] = wp.vec3(0.0, 0.0, 0.0)


# ============================================================
# Sensor
# ============================================================

class WarpRayCastSensor(Sensor[RayCastData]):
    """
    Raycast sensor (only for static geometry) using Warp BVH acceleration structure.:

      - Mesh /Box / HField / Plane → Warp BVH

    """

    def __init__(self, cfg: WarpRayCastSensorCfg):
        self.cfg = cfg

        self._mj_model: mujoco.MjModel | None = None
        self._model: mjwarp.Model | None = None
        self._data: mjwarp.Data | None = None
        self._device: str | None = None

        self._frame_type: Literal["body", "site", "geom"] = "body"
        self._frame_body_id: int | None = None
        self._frame_site_id: int | None = None
        self._frame_geom_id: int | None = None

        self._local_offsets: torch.Tensor | None = None
        self._local_dirs: torch.Tensor | None = None
        self._num_rays: int = 0

        # ---------- Warp mesh
        self._mesh: wp.Mesh | None = None
        self._mesh_id: wp.uint64 | None = None

        self._dirty = True

    # --------------------------------------------------------

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        self._mj_model = mj_model
        self._model = model
        self._data = data
        self._device = device

        frame = self.cfg.frame
        name = frame.prefixed_name()

        if frame.type == "body":
            self._frame_body_id = mj_model.body(name).id
            self._frame_type = "body"
        elif frame.type == "site":
            self._frame_site_id = mj_model.site(name).id
            self._frame_body_id = int(mj_model.site_bodyid[self._frame_site_id])
            self._frame_type = "site"
        else:
            self._frame_geom_id = mj_model.geom(name).id
            self._frame_body_id = int(mj_model.geom_bodyid[self._frame_geom_id])
            self._frame_type = "geom"

        self._local_offsets, self._local_dirs = self.cfg.pattern.generate_rays(
            mj_model, device
        )
        self._num_rays = self._local_offsets.shape[0]

        self._build_geometry()

    # --------------------------------------------------------

    def _build_geometry(self) -> None:
        mj = self._mj_model
        assert mj is not None

        verts_all = []
        faces_all = []
        voff = 0

        for gid in range(mj.ngeom):
            body_id = int(mj.geom_bodyid[gid])
            if self.cfg.exclude_parent_body and body_id == self._frame_body_id:
                continue
            if self.cfg.include_geom_groups is not None and mj.geom_group[gid] not in self.cfg.include_geom_groups:
                continue

            gtype = mj.geom_type[gid]
            if gtype not in (
                mujoco.mjtGeom.mjGEOM_MESH,
                mujoco.mjtGeom.mjGEOM_HFIELD,
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_PLANE,
            ):
                continue

            v, f = extract_geom_mesh(mj, gid)
            verts_all.append(v)
            faces_all.append(f + voff)
            voff += v.shape[0]

        if verts_all:
            verts = torch.cat(verts_all, 0)
            faces = torch.cat(faces_all, 0)

            self._mesh = wp.Mesh(
                points=wp.array(verts, dtype=wp.vec3, device=self._device),
                indices=wp.array(faces.flatten(), dtype=int, device=self._device),
            )
            self._mesh_id = self._mesh.id

            # Export mesh for debugging if requested
            if self.cfg.export_mesh_path:
                self._export_mesh_obj(verts, faces, self.cfg.export_mesh_path)
                print(f"[WarpRayCastSensor] Exported mesh to {self.cfg.export_mesh_path}")
                print(f"  - Vertices: {verts.shape[0]}")
                print(f"  - Faces: {faces.shape[0]}")

    # --------------------------------------------------------

    def _export_mesh_obj(self, verts: torch.Tensor, faces: torch.Tensor, path: str) -> None:
        """Export mesh to OBJ format for debugging."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "w") as f:
            f.write(f"# Warp Mesh Export\n")
            f.write(f"# Vertices: {verts.shape[0]}\n")
            f.write(f"# Faces: {faces.shape[0]}\n\n")
            
            # Write vertices
            verts_cpu = verts.cpu().numpy()
            for v in verts_cpu:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-based indexing)
            faces_cpu = faces.cpu().numpy()
            for face in faces_cpu:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # --------------------------------------------------------

    def _perform_raycast(self) -> None:
        assert self._data is not None
        assert self._local_offsets is not None
        assert self._local_dirs is not None

        data = self._data
        if self._frame_type == "body":
            pos = data.xpos[:, self._frame_body_id]
            mat = data.xmat[:, self._frame_body_id].view(-1, 3, 3)
        elif self._frame_type == "site":
            pos = data.site_xpos[:, self._frame_site_id]
            mat = data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
        else:
            pos = data.geom_xpos[:, self._frame_geom_id]
            mat = data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)

        B = pos.shape[0]
        N = self._num_rays
        rot = self._compute_alignment_rotation(mat)

        origins = pos[:, None] + torch.einsum("bij,nj->bni", rot, self._local_offsets)
        dirs = torch.einsum("bij,nj->bni", rot, self._local_dirs)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)

        O = origins.reshape(-1, 3).contiguous()
        D = dirs.reshape(-1, 3).contiguous()

        best_dist = torch.full((O.shape[0],), self.cfg.max_distance, device=O.device)
        best_n = torch.zeros_like(O)

        # ---------- Warp mesh only
        if self._mesh_id is not None:
            ro = wp.array(O, dtype=wp.vec3, device=self._device)
            rd = wp.array(D, dtype=wp.vec3, device=self._device)
            out_d = wp.empty(O.shape[0], dtype=float, device=self._device)
            out_n = wp.empty(O.shape[0], dtype=wp.vec3, device=self._device)

            wp.launch(
                _raycast_mesh_kernel,
                dim=O.shape[0],
                inputs=[self._mesh_id, ro, rd, self.cfg.max_distance, out_d, out_n],
                device=self._device,
            )

            md = wp.to_torch(out_d)
            mn = wp.to_torch(out_n)
            hit = (md >= 0) & (md < best_dist)
            best_dist[hit] = md[hit]
            best_n[hit] = mn[hit]

        best_dist[best_dist >= self.cfg.max_distance] = -1.0

        self._distances = best_dist.view(B, N)
        self._normals_w = best_n.view(B, N, 3)
        self._hit_pos_w = origins + dirs * self._distances[..., None]
        self._pos_w = pos
        self._quat_w = quat_from_matrix(mat)

    # --------------------------------------------------------

    @property
    def data(self) -> RayCastData:
        if self._dirty:
            self._perform_raycast()
            self._dirty = False
        return RayCastData(
            distances=self._distances,
            normals_w=self._normals_w,
            hit_pos_w=self._hit_pos_w,
            pos_w=self._pos_w,
            quat_w=self._quat_w,
        )

    def update(self, dt: float) -> None:
        self._dirty = True

    def reset(self, env_ids=None) -> None:
        self._dirty = True

    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        del scene_spec, entities

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self.cfg.debug_vis:
            return
        assert self._data is not None
        assert self._local_offsets is not None
        assert self._local_dirs is not None

        env_idx = visualizer.env_idx
        data = self.data

        if self._frame_type == "body":
            frame_pos = self._data.xpos[env_idx, self._frame_body_id].cpu().numpy()
            frame_mat_tensor = self._data.xmat[env_idx, self._frame_body_id].view(3, 3)
        elif self._frame_type == "site":
            frame_pos = self._data.site_xpos[env_idx, self._frame_site_id].cpu().numpy()
            frame_mat_tensor = self._data.site_xmat[env_idx, self._frame_site_id].view(3, 3)
        else:
            frame_pos = self._data.geom_xpos[env_idx, self._frame_geom_id].cpu().numpy()
            frame_mat_tensor = self._data.geom_xmat[env_idx, self._frame_geom_id].view(3, 3)

        # Apply ray alignment for visualization.
        rot_mat_tensor = self._compute_alignment_rotation(frame_mat_tensor.unsqueeze(0))[0]
        rot_mat = rot_mat_tensor.cpu().numpy()

        local_offsets_np = self._local_offsets.cpu().numpy()
        local_dirs_np = self._local_dirs.cpu().numpy()
        hit_positions_np = data.hit_pos_w[env_idx].cpu().numpy()
        distances_np = data.distances[env_idx].cpu().numpy()
        normals_np = data.normals_w[env_idx].cpu().numpy()

        meansize = visualizer.meansize
        ray_width = 0.1 * meansize
        sphere_radius = self.cfg.viz.hit_sphere_radius * meansize
        normal_length = self.cfg.viz.normal_length * meansize
        normal_width = 0.1 * meansize

        for i in range(self._num_rays):
            origin = frame_pos + rot_mat @ local_offsets_np[i]
            hit = distances_np[i] >= 0

            if hit:
                end = hit_positions_np[i]
                color = self.cfg.viz.hit_color
            else:
                direction = rot_mat @ local_dirs_np[i]
                end = origin + direction * min(0.5, self.cfg.max_distance * 0.05)
                color = self.cfg.viz.miss_color

            if self.cfg.viz.show_rays:
                visualizer.add_arrow(
                    start=origin,
                    end=end,
                    color=color,
                    width=ray_width,
                    label=f"{self.cfg.name}_ray_{i}",
                )

            if hit:
                visualizer.add_sphere(
                    center=end,
                    radius=sphere_radius,
                    color=self.cfg.viz.hit_sphere_color,
                    label=f"{self.cfg.name}_hit_{i}",
                )
                if self.cfg.viz.show_normals:
                    normal_end = end + normals_np[i] * normal_length
                    visualizer.add_arrow(
                        start=end,
                        end=normal_end,
                        color=self.cfg.viz.normal_color,
                        width=normal_width,
                        label=f"{self.cfg.name}_normal_{i}",
                    )

    # --------------------------------------------------------

    def _compute_alignment_rotation(self, mat: torch.Tensor) -> torch.Tensor:
        if self.cfg.ray_alignment == "base":
            return mat
        elif self.cfg.ray_alignment == "yaw":
            return self._extract_yaw_rotation(mat)
        else:
            return torch.eye(3, device=mat.device).unsqueeze(0).expand(mat.shape[0], 3, 3)

    def _extract_yaw_rotation(self, mat: torch.Tensor) -> torch.Tensor:
        x = mat[:, :, 0]
        x[:, 2] = 0
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)

        yaw = torch.zeros_like(mat)
        yaw[:, 0, 0] = x[:, 0]
        yaw[:, 1, 0] = x[:, 1]
        yaw[:, 0, 1] = -x[:, 1]
        yaw[:, 1, 1] = x[:, 0]
        yaw[:, 2, 2] = 1
        return yaw


# ============================================================
# Utils
# ============================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ],
        dtype=q.dtype,
        device=q.device,
    )
