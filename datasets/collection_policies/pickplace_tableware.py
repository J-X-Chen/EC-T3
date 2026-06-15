import mujoco
import numpy as np
import os

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class PickPlaceTablewareMarkovOracle(MarkovOracle):
    """Tableware oracle that grasps rims/edges instead of object centers."""

    def __init__(
        self,
        max_step=650,
        pre_push_lift_steps=8,
        pre_push_lift_delta=0.07,
        pre_push_xy_retreat=0.02,
        push_burst_steps=240,
        push_settle_steps=20,
        push_settle_lift=0.02,
        push_lookahead_min=0.012,
        push_lookahead_max=0.045,
        push_lateral_correction=0.35,
        push_contact_penetration=0.018,
        push_force_contact_threshold=0.04,
        push_contact_forward_tolerance=0.014,
        push_contact_lateral_tolerance=0.03,
        push_contact_z_tolerance=0.035,
        push_contact_margin=0.002,
        push_contact_outset=0.004,
        push_approach_clearance=0.03,
        push_transit_z=0.18,
        push_step_distance=0.24,
        push_follow_through_fraction=0.75,
        push_drive_gain=5.0,
        push_slowdown_dist=0.06,
        push_slow_drive_gain=3.0,
        push_fine_dist=0.035,
        push_fine_drive_gain=3.0,
        push_stop_dist=0.035,
        push_stop_lift=0.12,
        push_stop_backoff=0.06,
        bowl_push_height_offset=0.014,
        plate_push_height_offset=0.008,
        fallback_bowl_radius=0.084,
        fallback_plate_radius=0.028,
        fallback_mug_radius=0.05,
        fallback_fork_radius=0.08,
        fallback_knife_radius=0.11,
        max_visual_center_correction=0.02,
        success_dist=0.02,
        use_push_for_bottom_stack=True,
        edge_grasp_inset=0.006,
        edge_grasp_lift=0.14,
        edge_grasp_clearance=0.08,
        edge_grasp_xy_threshold=0.018,
        edge_grasp_z_threshold=0.018,
        edge_grasp_close_steps=18,
        plate_edge_grasp_close_steps=18,
        edge_grasp_release_steps=8,
        edge_release_hold_gain=2.0,
        edge_release_max_arm_action=0.18,
        edge_release_max_yaw_action=0.12,
        edge_grasp_yaw_offset=np.pi / 2.0,
        edge_grasp_min_other_clearance=0.036, #问题一，夹爪宽度为0.8，所以要设一个大概0.3的安全距离，防止夹到其他物体。问题二，永远都是推plate优先，要是盘子落点被挡住了就work不了了
        plate_grasp_top_offset=0.006,
        bowl_grasp_top_offset=0.020,
        mug_grasp_top_offset=0.028,
        flatware_grasp_top_offset=0.006,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_step = int(max_step)
        self._pre_push_lift_steps = int(pre_push_lift_steps)
        self._pre_push_lift_delta = float(pre_push_lift_delta)
        self._pre_push_xy_retreat = float(pre_push_xy_retreat)
        self._push_burst_steps = int(push_burst_steps)
        self._push_settle_steps = int(push_settle_steps)
        self._push_settle_lift = float(push_settle_lift)
        self._push_lookahead_min = float(push_lookahead_min)
        self._push_lookahead_max = float(push_lookahead_max)
        self._push_lateral_correction = float(push_lateral_correction)
        self._push_contact_penetration = float(push_contact_penetration)
        self._push_force_contact_threshold = float(push_force_contact_threshold)
        self._push_contact_forward_tolerance = float(push_contact_forward_tolerance)
        self._push_contact_lateral_tolerance = float(push_contact_lateral_tolerance)
        self._push_contact_z_tolerance = float(push_contact_z_tolerance)
        self._push_contact_margin = float(push_contact_margin)
        self._push_contact_outset = float(push_contact_outset)
        self._push_approach_clearance = float(push_approach_clearance)
        self._push_transit_z = float(push_transit_z)
        self._push_step_distance = float(push_step_distance)
        self._push_follow_through_fraction = float(push_follow_through_fraction)
        self._push_drive_gain = float(push_drive_gain)
        self._push_slowdown_dist = float(push_slowdown_dist)
        self._push_slow_drive_gain = float(push_slow_drive_gain)
        self._push_fine_dist = float(push_fine_dist)
        self._push_fine_drive_gain = float(push_fine_drive_gain)
        self._push_stop_dist = float(push_stop_dist)
        self._push_stop_lift = float(push_stop_lift)
        self._push_stop_backoff = float(push_stop_backoff)
        self._bowl_push_height_offset = float(bowl_push_height_offset)
        self._plate_push_height_offset = float(plate_push_height_offset)
        self._fallback_bowl_radius = float(fallback_bowl_radius)
        self._fallback_plate_radius = float(fallback_plate_radius)
        self._fallback_mug_radius = float(fallback_mug_radius)
        self._fallback_fork_radius = float(fallback_fork_radius)
        self._fallback_knife_radius = float(fallback_knife_radius)
        self._max_visual_center_correction = float(max_visual_center_correction)
        self._success_dist = float(success_dist)
        self._use_push_for_bottom_stack = bool(use_push_for_bottom_stack)
        self._edge_grasp_inset = float(edge_grasp_inset)
        self._edge_grasp_lift = float(edge_grasp_lift)
        self._edge_grasp_clearance = float(edge_grasp_clearance)
        self._edge_grasp_xy_threshold = float(edge_grasp_xy_threshold)
        self._edge_grasp_z_threshold = float(edge_grasp_z_threshold)
        self._edge_grasp_close_steps = int(edge_grasp_close_steps)
        self._plate_edge_grasp_close_steps = int(plate_edge_grasp_close_steps)
        self._edge_grasp_release_steps = int(edge_grasp_release_steps)
        self._edge_release_hold_gain = float(edge_release_hold_gain)
        self._edge_release_max_arm_action = max(0.0, float(edge_release_max_arm_action))
        self._edge_release_max_yaw_action = max(0.0, float(edge_release_max_yaw_action))
        self._edge_grasp_yaw_offset = float(edge_grasp_yaw_offset)
        self._edge_grasp_min_other_clearance = float(edge_grasp_min_other_clearance)
        self._plate_grasp_top_offset = float(plate_grasp_top_offset)
        self._bowl_grasp_top_offset = float(bowl_grasp_top_offset)
        self._mug_grasp_top_offset = float(mug_grasp_top_offset)
        self._flatware_grasp_top_offset = float(flatware_grasp_top_offset)
        self._mode = 'edge_move_above'
        self._mode_step = 0
        self._pre_push_target_z = 0.0
        self._mode_start_effector = np.zeros(3, dtype=np.float32)
        self._mode_start_yaw = 0.0
        self._active_push_plan = None
        self._active_grasp_plan = None
        self._active_target_block = None
        self._push_settle_pose = np.zeros(3, dtype=np.float32)
        self._push_drive_started = False
        self._edge_release_hold_pos = None
        self._edge_release_hold_yaw = 0.0
        self._final_pos = np.zeros(3, dtype=np.float32)
        self._final_yaw = 0.0

    def _extract_target_info(self, info):
        try:
            target_block = int(info['privileged/target_block'])
            block_pos = np.asarray(info[f'privileged/block_{target_block}_pos'], dtype=np.float32)
            target_pos = np.asarray(info['privileged/target_block_pos'], dtype=np.float32)
            return target_block, block_pos, target_pos
        except Exception:
            return -1, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def _push_debug_enabled(self):
        return os.environ.get('TABLEWARE_PUSH_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}

    def _push_debug(self, message):
        if self._push_debug_enabled():
            print(f'[tableware-push] {message}')

    def _is_bottom_push_target(self, info):
        target_block = int(info.get('privileged/target_block', -1))
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        stacking = bool(getattr(env_unwrapped, '_stacking', False))
        stack_targets = getattr(env_unwrapped, '_stack_targets', None)
        if (not stacking) or (stack_targets is None) or (len(stack_targets) == 0):
            return False
        return target_block == int(stack_targets[0])

    def _set_mode(self, mode, ob, info):
        self._mode = mode
        self._mode_step = 0
        self._mode_start_effector = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        self._mode_start_yaw = float(np.asarray(info.get('proprio/effector_yaw', [0.0]))[0])
        self._active_push_plan = None
        if mode != 'push_settle':
            self._push_drive_started = False
        if mode == 'pre_push_lift':
            start_z = float(self._mode_start_effector[2])
            self._pre_push_target_z = start_z + self._pre_push_lift_delta
        if mode == 'edge_release':
            self._edge_release_hold_pos = self._mode_start_effector.copy()
            self._edge_release_hold_yaw = self._mode_start_yaw
        elif mode != 'edge_retreat':
            self._edge_release_hold_pos = None

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._mode = 'edge_move_above'
        self._mode_step = 0
        self._pre_push_target_z = 0.0
        self._mode_start_effector = np.zeros(3, dtype=np.float32)
        self._mode_start_yaw = 0.0
        self._active_push_plan = None
        self._active_grasp_plan = None
        self._active_target_block = None
        self._push_settle_pose = np.zeros(3, dtype=np.float32)
        self._push_drive_started = False
        self._edge_release_hold_pos = None
        self._edge_release_hold_yaw = 0.0
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        arm_bounds = getattr(env_unwrapped, '_arm_sampling_bounds', None)
        if arm_bounds is not None:
            self._final_pos = np.random.uniform(*arm_bounds).astype(np.float32)
        else:
            self._final_pos = np.array([0.425, 0.0, 0.28], dtype=np.float32)
        self._final_yaw = float(np.random.uniform(-np.pi, np.pi))

    def _get_object_kind(self, target_block):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        object_kinds = getattr(env_unwrapped, '_tableware_object_kinds', None)
        if object_kinds is not None and 0 <= target_block < len(object_kinds):
            return object_kinds[target_block]
        if getattr(env_unwrapped, '_object_type', 'cube') != 'cube':
            return 'bowl'
        return 'plate'

    def _fallback_push_radius(self, target_block):
        object_kind = self._get_object_kind(target_block)
        if object_kind == 'bowl':
            return self._fallback_bowl_radius
        if object_kind == 'mug':
            return self._fallback_mug_radius
        if object_kind == 'fork':
            return self._fallback_fork_radius
        if object_kind == 'knife':
            return self._fallback_knife_radius
        return self._fallback_plate_radius

    def _get_object_profile(self, target_block):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        profiles = getattr(env_unwrapped, '_tableware_object_profiles', None)
        if profiles is not None and 0 <= target_block < len(profiles):
            return profiles[target_block]

        kind = self._get_object_kind(target_block)
        if kind == 'plate':
            return {'bottom_z': -0.01, 'top_z': 0.025, 'horizontal_radius': self._fallback_plate_radius}
        if kind == 'mug':
            return {'bottom_z': -0.05, 'top_z': 0.045, 'horizontal_radius': self._fallback_mug_radius}
        if kind == 'fork':
            return {'bottom_z': -0.01, 'top_z': 0.01, 'horizontal_radius': self._fallback_fork_radius}
        if kind == 'knife':
            return {'bottom_z': -0.01, 'top_z': 0.01, 'horizontal_radius': self._fallback_knife_radius}
        return {'bottom_z': -0.03, 'top_z': 0.035, 'horizontal_radius': self._fallback_bowl_radius}

    def _fallback_grasp_radius(self, target_block):
        profile = self._get_object_profile(target_block)
        return float(profile.get('horizontal_radius', self._fallback_push_radius(target_block)))

    def _grasp_top_offset(self, target_block):
        kind = self._get_object_kind(target_block)
        if kind == 'plate':
            return self._plate_grasp_top_offset
        if kind == 'mug':
            return self._mug_grasp_top_offset
        if kind in {'fork', 'knife'}:
            return self._flatware_grasp_top_offset
        return self._bowl_grasp_top_offset

    def _grasp_z_relative_to_body(self, target_block):
        profile = self._get_object_profile(target_block)
        bottom_z = float(profile.get('bottom_z', -0.03))
        top_z = float(profile.get('top_z', 0.035))
        grasp_z = top_z - self._grasp_top_offset(target_block)
        return float(np.clip(grasp_z, bottom_z + 0.012, top_z + 0.004))

    def _get_push_upper_z_limit(self):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        arm_bounds = getattr(env_unwrapped, '_arm_sampling_bounds', None)
        if arm_bounds is not None:
            return float(arm_bounds[1][2])
        workspace_bounds = getattr(env_unwrapped, '_workspace_bounds', None)
        if workspace_bounds is not None:
            return float(workspace_bounds[1][2])
        return 0.35

    def _push_height_offset(self, target_block):
        object_kind = self._get_object_kind(target_block)
        if object_kind == 'bowl':
            return self._bowl_push_height_offset
        if object_kind == 'plate':
            profile = self._get_object_profile(target_block)
            bottom_z = float(profile.get('bottom_z', -0.01))
            top_z = float(profile.get('top_z', 0.025))
            side_offset = bottom_z + 0.52 * max(top_z - bottom_z, 0.0)
            return float(np.clip(side_offset, 0.006, self._plate_push_height_offset))
        return self._plate_push_height_offset

    def _normalize_xy(self, vec_xy, fallback=(1.0, 0.0)):
        vec_xy = np.asarray(vec_xy, dtype=np.float32)
        norm = float(np.linalg.norm(vec_xy))
        if norm < 1e-6:
            return np.asarray(fallback, dtype=np.float32)
        return (vec_xy / norm).astype(np.float32)

    def _move_effector_action(self, effector_pos, desired_pos, gripper_cmd, gain_pos=8.0):
        diff = np.asarray(desired_pos, dtype=np.float32) - np.asarray(effector_pos, dtype=np.float32)
        action = np.zeros(5, dtype=np.float32)
        action[:3] = np.clip(diff * gain_pos, -1.0, 1.0)
        action[3] = 0.0
        action[4] = float(gripper_cmd)
        return action

    def _move_pose_action(self, info, desired_pos, desired_yaw, gripper_cmd, gain_pos=6.0, gain_yaw=3.0):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        effector_yaw = float(np.asarray(info.get('proprio/effector_yaw', [0.0]))[0])
        diff = np.asarray(desired_pos, dtype=np.float32) - effector_pos
        diff = self.shape_diff(diff)
        action = np.zeros(5, dtype=np.float32)
        action[:3] = np.clip(diff[:3] * float(gain_pos), -1.0, 1.0)
        yaw_target = self.shortest_yaw(effector_yaw, float(desired_yaw), n=2)
        action[3] = np.clip((yaw_target - effector_yaw) * float(gain_yaw), -1.0, 1.0)
        action[4] = float(gripper_cmd)
        return action

    def _edge_release_gripper_cmd(self):
        release_steps = max(1, int(self._edge_grasp_release_steps))
        open_frac = min(1.0, float(self._mode_step + 1) / float(release_steps))
        return -open_frac

    def _edge_release_hold_action(self, info, fallback_pos, fallback_yaw):
        if self._edge_release_hold_pos is None:
            self._edge_release_hold_pos = np.asarray(fallback_pos, dtype=np.float32).copy()
            self._edge_release_hold_yaw = float(fallback_yaw)

        action = self._move_pose_action(
            info,
            self._edge_release_hold_pos,
            self._edge_release_hold_yaw,
            gripper_cmd=self._edge_release_gripper_cmd(),
            gain_pos=self._edge_release_hold_gain,
            gain_yaw=1.0,
        )
        action[:3] = np.clip(
            action[:3],
            -self._edge_release_max_arm_action,
            self._edge_release_max_arm_action,
        )
        action[3] = np.clip(
            action[3],
            -self._edge_release_max_yaw_action,
            self._edge_release_max_yaw_action,
        )
        return action

    def _cross_2d(self, a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def _convex_hull_xy(self, points):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return None
        if len(points) == 1:
            return points.astype(np.float32)

        rounded = np.round(points, decimals=8)
        unique = np.unique(rounded, axis=0)
        if len(unique) <= 2:
            return unique.astype(np.float32)

        order = np.lexsort((unique[:, 1], unique[:, 0]))
        pts = [tuple(p) for p in unique[order]]

        def build_half(iter_points):
            half = []
            for pt in iter_points:
                while len(half) >= 2:
                    a = np.asarray(half[-2], dtype=np.float64)
                    b = np.asarray(half[-1], dtype=np.float64)
                    c = np.asarray(pt, dtype=np.float64)
                    if self._cross_2d(b - a, c - b) > 0.0:
                        break
                    half.pop()
                half.append(pt)
            return half

        lower = build_half(pts)
        upper = build_half(reversed(pts))
        hull = lower[:-1] + upper[:-1]
        return np.asarray(hull, dtype=np.float32)

    def _build_circle_polygon(self, center_xy, radius, num_points=24):
        angles = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False, dtype=np.float64)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(radius)
        return (np.asarray(center_xy, dtype=np.float64)[None, :] + circle).astype(np.float32)

    def _build_capsule_polygon(self, point_a_xy, point_b_xy, radius, num_cap_points=12):
        point_a_xy = np.asarray(point_a_xy, dtype=np.float64)
        point_b_xy = np.asarray(point_b_xy, dtype=np.float64)
        seg = point_b_xy - point_a_xy
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm <= 1e-8:
            return self._build_circle_polygon(point_a_xy, radius, num_points=max(2 * int(num_cap_points), 12))

        tangent = seg / seg_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        theta_a = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, int(num_cap_points), endpoint=False, dtype=np.float64)
        theta_b = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(num_cap_points), endpoint=False, dtype=np.float64)

        cap_a = point_a_xy[None, :] + (
            np.cos(theta_a)[:, None] * tangent[None, :] + np.sin(theta_a)[:, None] * normal[None, :]
        ) * float(radius)
        cap_b = point_b_xy[None, :] + (
            np.cos(theta_b)[:, None] * tangent[None, :] + np.sin(theta_b)[:, None] * normal[None, :]
        ) * float(radius)
        return np.concatenate([cap_a, cap_b], axis=0).astype(np.float32)

    def _geom_footprint_polygon(self, geom_id):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        model = env_unwrapped._model
        data = env_unwrapped._data

        geom_type = int(model.geom_type[geom_id])
        geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        center_xy = geom_pos[:2]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            corners_xy = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        offset_local = np.array([sx, sy, sz], dtype=np.float64) * geom_size[:3]
                        offset_world = geom_rot @ offset_local
                        corners_xy.append(center_xy + offset_world[:2])
            return self._convex_hull_xy(corners_xy)

        if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius = float(geom_size[0])
            half_length = float(geom_size[1]) if geom_size.shape[0] > 1 else 0.0
            axis_xy = geom_rot[:2, 2] * half_length
            point_a_xy = center_xy - axis_xy
            point_b_xy = center_xy + axis_xy
            return self._build_capsule_polygon(point_a_xy, point_b_xy, radius)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return self._build_circle_polygon(center_xy, float(geom_size[0]))

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id >= 0:
                vert_adr = int(model.mesh_vertadr[mesh_id])
                vert_num = int(model.mesh_vertnum[mesh_id])
                if vert_num > 0:
                    verts = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
                    world_verts = geom_pos[None, :] + verts @ geom_rot.T
                    return self._convex_hull_xy(world_verts[:, :2])

        return self._build_circle_polygon(center_xy, float(model.geom_rbound[geom_id]))

    def _ray_segment_intersection_distance(self, ray_origin_xy, ray_dir_xy, seg_start_xy, seg_end_xy):
        ray_origin_xy = np.asarray(ray_origin_xy, dtype=np.float64)
        ray_dir_xy = np.asarray(ray_dir_xy, dtype=np.float64)
        seg_start_xy = np.asarray(seg_start_xy, dtype=np.float64)
        seg_end_xy = np.asarray(seg_end_xy, dtype=np.float64)

        seg_dir_xy = seg_end_xy - seg_start_xy
        denom = self._cross_2d(ray_dir_xy, seg_dir_xy)
        if abs(denom) <= 1e-8:
            return None

        rel_xy = seg_start_xy - ray_origin_xy
        ray_t = self._cross_2d(rel_xy, seg_dir_xy) / denom
        seg_u = self._cross_2d(rel_xy, ray_dir_xy) / denom
        if ray_t < -1e-8 or seg_u < -1e-8 or seg_u > 1.0 + 1e-8:
            return None
        return float(max(ray_t, 0.0))

    def _estimate_push_ray_distance(self, target_block, center_xy, push_dir):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        model = env_unwrapped._model
        geom_ids = getattr(env_unwrapped, '_object_geom_ids_list', None)

        approach_dir = self._normalize_xy(-np.asarray(push_dir, dtype=np.float32))
        center_xy = np.asarray(center_xy, dtype=np.float32)

        if geom_ids is None or not (0 <= target_block < len(geom_ids)):
            return self._fallback_push_radius(target_block)

        best_distance = None
        for geom_id in geom_ids[target_block]:
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                continue
            polygon_xy = self._geom_footprint_polygon(geom_id)
            if polygon_xy is None or len(polygon_xy) < 2:
                continue
            for idx in range(len(polygon_xy)):
                dist = self._ray_segment_intersection_distance(
                    center_xy,
                    approach_dir,
                    polygon_xy[idx],
                    polygon_xy[(idx + 1) % len(polygon_xy)],
                )
                if dist is None:
                    continue
                if best_distance is None or dist > best_distance:
                    best_distance = dist

        if best_distance is None or best_distance <= 1e-5:
            return self._fallback_push_radius(target_block)
        return float(best_distance)

    def _object_footprint_hull(self, target_block):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        model = env_unwrapped._model
        geom_ids = getattr(env_unwrapped, '_object_geom_ids_list', None)
        if geom_ids is None or not (0 <= target_block < len(geom_ids)):
            return None

        points = []
        for geom_id in geom_ids[target_block]:
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                continue
            polygon_xy = self._geom_footprint_polygon(geom_id)
            if polygon_xy is not None and len(polygon_xy) > 0:
                points.extend(np.asarray(polygon_xy, dtype=np.float32).tolist())

        if len(points) < 3:
            return None
        return self._convex_hull_xy(points)

    def _push_center_xy(self, target_block, block_pos):
        center_xy = np.asarray(block_pos[:2], dtype=np.float32)
        if self._get_object_kind(target_block) not in {'plate', 'fork', 'knife'}:
            return center_xy

        hull_xy = self._object_footprint_hull(target_block)
        if hull_xy is None or len(hull_xy) < 3:
            return center_xy

        hull_xy = np.asarray(hull_xy, dtype=np.float32)
        geom_center_xy = 0.5 * (np.min(hull_xy, axis=0) + np.max(hull_xy, axis=0))
        correction = geom_center_xy - center_xy
        correction_norm = float(np.linalg.norm(correction))
        max_correction = max(0.0, self._max_visual_center_correction)
        if correction_norm > max_correction > 0.0:
            correction = correction * (max_correction / correction_norm)
        return (center_xy + correction).astype(np.float32)

    def _estimate_edge_ray_distance(self, target_block, center_xy, edge_dir):
        center_xy = np.asarray(center_xy, dtype=np.float32)
        edge_dir = self._normalize_xy(edge_dir)
        hull_xy = self._object_footprint_hull(target_block)
        if hull_xy is None or len(hull_xy) < 3:
            return self._fallback_grasp_radius(target_block)

        intersections = []
        for idx in range(len(hull_xy)):
            dist = self._ray_segment_intersection_distance(
                center_xy,
                edge_dir,
                hull_xy[idx],
                hull_xy[(idx + 1) % len(hull_xy)],
            )
            if dist is not None and dist > 1e-5:
                intersections.append(float(dist))

        if len(intersections) == 0:
            return self._fallback_grasp_radius(target_block)
        return float(min(intersections))

    def _grasp_center_xy(self, target_block, block_pos):
        center_xy = np.asarray(block_pos[:2], dtype=np.float32)
        hull_xy = self._object_footprint_hull(target_block)
        if hull_xy is None or len(hull_xy) < 3:
            return center_xy

        hull_xy = np.asarray(hull_xy, dtype=np.float32)
        return (0.5 * (np.min(hull_xy, axis=0) + np.max(hull_xy, axis=0))).astype(np.float32)

    def _point_segment_distance_xy(self, point_xy, seg_start_xy, seg_end_xy):
        point_xy = np.asarray(point_xy, dtype=np.float64)
        seg_start_xy = np.asarray(seg_start_xy, dtype=np.float64)
        seg_end_xy = np.asarray(seg_end_xy, dtype=np.float64)
        seg_xy = seg_end_xy - seg_start_xy
        seg_len_sq = float(np.dot(seg_xy, seg_xy))
        if seg_len_sq <= 1e-12:
            return float(np.linalg.norm(point_xy - seg_start_xy))
        t = float(np.clip(np.dot(point_xy - seg_start_xy, seg_xy) / seg_len_sq, 0.0, 1.0))
        closest_xy = seg_start_xy + t * seg_xy
        return float(np.linalg.norm(point_xy - closest_xy))

    def _point_in_polygon_xy(self, point_xy, polygon_xy):
        polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
        if len(polygon_xy) < 3:
            return False

        x, y = np.asarray(point_xy, dtype=np.float64)
        inside = False
        prev_idx = len(polygon_xy) - 1
        for idx in range(len(polygon_xy)):
            xi, yi = polygon_xy[idx]
            xj, yj = polygon_xy[prev_idx]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
                if x < x_cross:
                    inside = not inside
            prev_idx = idx
        return inside

    def _point_polygon_distance_xy(self, point_xy, polygon_xy):
        polygon_xy = np.asarray(polygon_xy, dtype=np.float32)
        if len(polygon_xy) == 0:
            return float('inf')
        if len(polygon_xy) == 1:
            return float(np.linalg.norm(np.asarray(point_xy, dtype=np.float32) - polygon_xy[0]))
        if self._point_in_polygon_xy(point_xy, polygon_xy):
            return 0.0

        edge_count = len(polygon_xy) if len(polygon_xy) >= 3 else len(polygon_xy) - 1
        distances = [
            self._point_segment_distance_xy(point_xy, polygon_xy[idx], polygon_xy[(idx + 1) % len(polygon_xy)])
            for idx in range(edge_count)
        ]
        return float(min(distances)) if distances else float('inf')

    def _edge_grasp_other_clearance(self, target_block, grasp_xy, info):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        num_objects = int(getattr(env_unwrapped, 'num_objects', 0))
        grasp_xy = np.asarray(grasp_xy, dtype=np.float32)
        best_clearance = None

        for other_block in range(num_objects):
            if other_block == target_block:
                continue

            hull_xy = self._object_footprint_hull(other_block)
            if hull_xy is not None and len(hull_xy) > 0:
                clearance = self._point_polygon_distance_xy(grasp_xy, hull_xy)
            else:
                other_pos = info.get(f'privileged/block_{other_block}_pos', None)
                if other_pos is None:
                    continue
                other_xy = np.asarray(other_pos, dtype=np.float32)[:2]
                clearance = float(np.linalg.norm(grasp_xy - other_xy) - self._fallback_grasp_radius(other_block))

            if best_clearance is None or clearance < best_clearance:
                best_clearance = float(clearance)

        return float('inf') if best_clearance is None else float(best_clearance)

    def _choose_two_point_edge_grasp(self, info, target_block, block_pos, target_pos, effector_pos):
        center_xy = self._grasp_center_xy(target_block, block_pos)
        effector_delta_xy = np.asarray(effector_pos[:2] - center_xy, dtype=np.float32)
        target_delta_xy = np.asarray(target_pos[:2] - block_pos[:2], dtype=np.float32)

        if float(np.linalg.norm(effector_delta_xy)) > 1e-4:
            nearest_dir = self._normalize_xy(effector_delta_xy)
        elif float(np.linalg.norm(target_delta_xy)) > 1e-4:
            nearest_dir = self._normalize_xy(-target_delta_xy)
        else:
            nearest_dir = np.array([1.0, 0.0], dtype=np.float32)

        nearest_edge_distance = self._estimate_edge_ray_distance(target_block, center_xy, nearest_dir)
        grasp_radius = max(nearest_edge_distance - self._edge_grasp_inset, nearest_edge_distance * 0.55)

        candidates = []
        for choice_name, edge_dir in (('nearest', nearest_dir), ('opposite', -nearest_dir)):
            edge_dir = self._normalize_xy(edge_dir)
            # Only these two points are allowed; they are exact reflections through the footprint center.
            grasp_xy = center_xy + edge_dir * float(grasp_radius)
            candidates.append(
                {
                    'choice': choice_name,
                    'edge_dir': edge_dir.astype(np.float32),
                    'edge_distance': float(nearest_edge_distance),
                    'grasp_offset_xy': np.asarray(grasp_xy - block_pos[:2], dtype=np.float32),
                    'other_clearance': self._edge_grasp_other_clearance(target_block, grasp_xy, info),
                }
            )

        nearest_candidate, opposite_candidate = candidates
        min_clearance = max(0.0, self._edge_grasp_min_other_clearance)
        if min_clearance <= 0.0:
            return nearest_candidate

        nearest_clearance = float(nearest_candidate['other_clearance'])
        opposite_clearance = float(opposite_candidate['other_clearance'])
        if nearest_clearance <= 0.0 and opposite_clearance > nearest_clearance:
            return opposite_candidate
        if nearest_clearance < min_clearance and opposite_clearance >= min_clearance:
            return opposite_candidate
        return nearest_candidate

    def _build_edge_grasp_plan(self, info):
        target_block, block_pos, target_pos = self._extract_target_info(info)
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        grasp_choice = self._choose_two_point_edge_grasp(info, target_block, block_pos, target_pos, effector_pos)
        edge_dir = np.asarray(grasp_choice['edge_dir'], dtype=np.float32)
        edge_distance = float(grasp_choice['edge_distance'])
        grasp_offset_xy = np.asarray(grasp_choice['grasp_offset_xy'], dtype=np.float32)
        z_rel = self._grasp_z_relative_to_body(target_block)

        raw_yaw = float(np.arctan2(edge_dir[1], edge_dir[0]) + self._edge_grasp_yaw_offset)
        return {
            'target_block': target_block,
            'object_kind': self._get_object_kind(target_block),
            'grasp_offset_xy': grasp_offset_xy,
            'edge_dir': edge_dir.astype(np.float32),
            'edge_distance': float(edge_distance),
            'z_rel': float(z_rel),
            'grasp_yaw': raw_yaw,
            'grasp_choice': grasp_choice['choice'],
            'other_clearance': float(grasp_choice['other_clearance']),
            'target_pos_at_plan': np.asarray(target_pos, dtype=np.float32).copy(),
        }

    def _edge_plan_poses(self, info, plan):
        _, block_pos, target_pos = self._extract_target_info(info)
        upper_z_limit = self._get_push_upper_z_limit()
        offset_xy = np.asarray(plan['grasp_offset_xy'], dtype=np.float32)
        z_rel = float(plan['z_rel'])
        locked_grasp_xy = plan.get('locked_grasp_xy', None)
        locked_place_xy = plan.get('locked_place_xy', None)

        grasp_xy = (
            np.asarray(locked_grasp_xy, dtype=np.float32)
            if locked_grasp_xy is not None
            else np.asarray(block_pos[:2], dtype=np.float32) + offset_xy
        )
        place_xy = (
            np.asarray(locked_place_xy, dtype=np.float32)
            if locked_place_xy is not None
            else np.asarray(target_pos[:2], dtype=np.float32) + offset_xy
        )
        grasp_pos = np.array(
            [grasp_xy[0], grasp_xy[1], block_pos[2] + z_rel],
            dtype=np.float32,
        )
        place_pos = np.array(
            [place_xy[0], place_xy[1], target_pos[2] + z_rel],
            dtype=np.float32,
        )
        if 'locked_above_z' in plan:
            above_z = float(plan['locked_above_z'])
        else:
            above_z = float(
                np.clip(
                    max(grasp_pos[2], place_pos[2]) + self._edge_grasp_lift,
                    max(0.18, grasp_pos[2] + self._edge_grasp_clearance),
                    upper_z_limit - 0.01,
                )
            )
        above_grasp_pos = np.array([grasp_pos[0], grasp_pos[1], above_z], dtype=np.float32)
        above_place_pos = np.array([place_pos[0], place_pos[1], above_z], dtype=np.float32)
        return grasp_pos, above_grasp_pos, place_pos, above_place_pos

    def _lock_edge_grasp_path(self, plan, effector_pos, target_pos, above_z):
        offset_xy = np.asarray(plan['grasp_offset_xy'], dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        plan['locked_grasp_xy'] = np.asarray(effector_pos[:2], dtype=np.float32).copy()
        plan['locked_place_xy'] = (target_pos[:2] + offset_xy).astype(np.float32)
        plan['locked_above_z'] = float(above_z)

    def _edge_close_steps_for_target(self, target_block):
        if self._get_object_kind(target_block) in {'plate', 'fork', 'knife'}:
            return self._plate_edge_grasp_close_steps
        return self._edge_grasp_close_steps

    def _pose_aligned(self, effector_pos, desired_pos, xy_threshold=None, z_threshold=None):
        if xy_threshold is None:
            xy_threshold = self._edge_grasp_xy_threshold
        if z_threshold is None:
            z_threshold = self._edge_grasp_z_threshold
        effector_pos = np.asarray(effector_pos, dtype=np.float32)
        desired_pos = np.asarray(desired_pos, dtype=np.float32)
        xy_aligned = float(np.linalg.norm(effector_pos[:2] - desired_pos[:2])) <= float(xy_threshold)
        z_aligned = abs(float(effector_pos[2] - desired_pos[2])) <= float(z_threshold)
        return xy_aligned and z_aligned

    def _edge_grasp_action(self, ob, info):
        if self._active_grasp_plan is None:
            self._active_grasp_plan = self._build_edge_grasp_plan(info)
            self._mode = 'edge_move_above'
            self._mode_step = 0

        plan = self._active_grasp_plan
        target_block, block_pos, target_pos = self._extract_target_info(info)
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        grasp_pos, above_grasp_pos, place_pos, above_place_pos = self._edge_plan_poses(info, plan)
        target_dist = float(np.linalg.norm(np.asarray(target_pos - block_pos, dtype=np.float32)))
        desired_yaw = float(plan['grasp_yaw'])

        if self._mode == 'edge_retreat' and self._mode_step >= self._edge_grasp_release_steps * 4:
            self._done = True

        if self._mode == 'edge_move_above':
            self.print_phase('edge-1: move above object edge')
            action = self._move_pose_action(info, above_grasp_pos, desired_yaw, gripper_cmd=-1.0, gain_pos=6.0)
            if self._pose_aligned(effector_pos, above_grasp_pos, xy_threshold=0.025, z_threshold=0.025):
                self._set_mode('edge_descend', ob, info)
        elif self._mode == 'edge_descend':
            self.print_phase('edge-2: descend to bbox/rim edge')
            action = self._move_pose_action(info, grasp_pos, desired_yaw, gripper_cmd=-1.0, gain_pos=5.0)
            if self._pose_aligned(effector_pos, grasp_pos):
                self._set_mode('edge_close', ob, info)
        elif self._mode == 'edge_close':
            self.print_phase('edge-3: close on edge')
            action = self._move_pose_action(info, grasp_pos, desired_yaw, gripper_cmd=1.0, gain_pos=4.0)
            gripper_contact = float(np.asarray(info.get('proprio/gripper_contact', [0.0]))[0])
            close_steps = self._edge_close_steps_for_target(target_block)
            if self._get_object_kind(target_block) == 'plate':
                close_ready = self._mode_step >= close_steps
            else:
                min_contact_steps = min(4, close_steps)
                close_ready = self._mode_step >= close_steps or (
                    self._mode_step >= min_contact_steps and gripper_contact > 0.35
                )
            if close_ready:
                self._lock_edge_grasp_path(plan, effector_pos, target_pos, above_grasp_pos[2])
                self._set_mode('edge_lift', ob, info)
        elif self._mode == 'edge_lift':
            self.print_phase('edge-4: lift while holding edge')
            action = self._move_pose_action(info, above_grasp_pos, desired_yaw, gripper_cmd=1.0, gain_pos=6.0)
            if self._pose_aligned(effector_pos, above_grasp_pos, xy_threshold=0.012, z_threshold=0.018):
                self._set_mode('edge_move_target', ob, info)
        elif self._mode == 'edge_move_target':
            self.print_phase('edge-5: move edge grip above target')
            action = self._move_pose_action(info, above_place_pos, desired_yaw, gripper_cmd=1.0, gain_pos=6.0)
            if self._pose_aligned(effector_pos, above_place_pos, xy_threshold=0.025, z_threshold=0.018):
                self._set_mode('edge_descend_target', ob, info)
        elif self._mode == 'edge_descend_target':
            self.print_phase('edge-6: place with preserved edge offset')
            action = self._move_pose_action(info, place_pos, desired_yaw, gripper_cmd=1.0, gain_pos=5.0)
            if self._pose_aligned(effector_pos, place_pos) or target_dist <= self._success_dist:
                self._set_mode('edge_release', ob, info)
        elif self._mode == 'edge_release':
            self.print_phase('edge-7: release')
            action = self._edge_release_hold_action(info, effector_pos, desired_yaw)
            if self._mode_step >= self._edge_grasp_release_steps:
                self._set_mode('edge_retreat', ob, info)
        else:
            self.print_phase('edge-8: retreat above target')
            action = self._move_pose_action(info, above_place_pos, desired_yaw, gripper_cmd=-1.0, gain_pos=5.0)
            final_aligned = self._pose_aligned(effector_pos, above_place_pos, xy_threshold=0.04, z_threshold=0.04)
            if final_aligned and self._mode_step >= 2:
                self._done = True

        if self._push_debug_enabled():
            self._push_debug(
                f"edge_grasp target={target_block} kind={plan['object_kind']} mode={self._mode} "
                f"offset={np.round(plan['grasp_offset_xy'], 4).tolist()} "
                f"edge_dist={plan['edge_distance']:.4f} target_dist={target_dist:.4f}"
            )
        return action

    def _build_push_plan(self, info):
        target_block, block_pos, target_pos = self._extract_target_info(info)
        push_center_xy = self._push_center_xy(target_block, block_pos)
        push_delta_xy = np.asarray(target_pos[:2] - push_center_xy, dtype=np.float32)
        push_dist_xy = float(np.linalg.norm(push_delta_xy))

        if push_dist_xy < 1e-6:
            base_push_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            base_push_dir = (push_delta_xy / push_dist_xy).astype(np.float32)

        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        workspace_bounds = getattr(env_unwrapped, '_workspace_bounds', None)
        if workspace_bounds is not None:
            lower_xy = np.asarray(workspace_bounds[0, :2], dtype=np.float32) + 0.002
            upper_xy = np.asarray(workspace_bounds[1, :2], dtype=np.float32) - 0.002
        else:
            lower_xy = np.array([-np.inf, -np.inf], dtype=np.float32)
            upper_xy = np.array([np.inf, np.inf], dtype=np.float32)

        def rotate_dir(vec, angle):
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float32)

        def build_candidate(candidate_dir):
            candidate_dir = self._normalize_xy(candidate_dir)
            ray = self._estimate_push_ray_distance(target_block, push_center_xy, candidate_dir)
            contour = push_center_xy - candidate_dir * ray
            contact = contour - candidate_dir * self._push_contact_outset
            approach = contact - candidate_dir * self._push_approach_clearance
            return candidate_dir, ray, contour, contact, approach

        def workspace_violation(point_xy):
            below = np.maximum(lower_xy - point_xy, 0.0)
            above = np.maximum(point_xy - upper_xy, 0.0)
            return float(np.linalg.norm(below + above))

        best_score = None
        best_candidate = None
        if self._get_object_kind(target_block) == 'plate':
            angle_candidates = [0.0]
        else:
            angle_candidates = [
                0.0,
                np.pi / 12.0,
                -np.pi / 12.0,
                np.pi / 6.0,
                -np.pi / 6.0,
                np.pi / 4.0,
                -np.pi / 4.0,
                np.pi / 3.0,
                -np.pi / 3.0,
                5.0 * np.pi / 12.0,
                -5.0 * np.pi / 12.0,
            ]
        for angle in angle_candidates:
            candidate = build_candidate(rotate_dir(base_push_dir, angle))
            candidate_dir, _, _, contact_xy_candidate, approach_xy_candidate = candidate
            if float(np.dot(candidate_dir, base_push_dir)) < 0.2:
                continue
            violation = workspace_violation(contact_xy_candidate) + workspace_violation(approach_xy_candidate)
            heading_penalty = 0.2 * (1.0 - float(np.dot(candidate_dir, base_push_dir)))
            score = 2.0 * violation + heading_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            best_candidate = build_candidate(base_push_dir)
        push_dir, ray_distance, contour_xy, contact_xy, approach_xy = best_candidate

        upper_z_limit = self._get_push_upper_z_limit()
        push_z = float(np.clip(block_pos[2] + self._push_height_offset(target_block), 0.02, upper_z_limit - 0.02))
        approach_z = float(np.clip(push_z + 0.015, push_z + 0.005, upper_z_limit - 0.01))
        follow_through = max(float(ray_distance) * self._push_follow_through_fraction, 0.0)
        push_stride = min(push_dist_xy + follow_through + self._push_contact_margin, self._push_step_distance)
        push_goal_xy = contact_xy + push_dir * push_stride

        return {
            'target_block': target_block,
            'block_pos': block_pos,
            'target_pos': target_pos,
            'push_center_xy': push_center_xy.astype(np.float32),
            'push_dir': push_dir,
            'push_dist_xy': push_dist_xy,
            'contour_xy': contour_xy.astype(np.float32),
            'contact_xy': contact_xy.astype(np.float32),
            'approach_xy': approach_xy.astype(np.float32),
            'push_goal_xy': push_goal_xy.astype(np.float32),
            'ray_distance': float(ray_distance),
            'push_z': push_z,
            'approach_z': float(approach_z),
        }

    def _pre_push_lift_action(self, info, push_plan):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        desired_pos = np.array(
            [push_plan['approach_xy'][0], push_plan['approach_xy'][1], push_plan['approach_z']],
            dtype=np.float32,
        )
        desired_yaw = self._push_yaw(push_plan['push_dir'])
        return self._move_pose_action(info, desired_pos, desired_yaw, gripper_cmd=-1.0, gain_pos=8.0)

    def _push_yaw(self, push_dir):
        push_dir = self._normalize_xy(push_dir)
        return float(np.arctan2(push_dir[1], push_dir[0]) + np.pi / 2.0)

    def _set_push_settle_mode(self, info, push_plan=None):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        upper_z_limit = self._get_push_upper_z_limit()
        settle_pose = effector_pos.copy()
        if push_plan is not None:
            push_dir = self._normalize_xy(push_plan['push_dir'])
            settle_pose[:2] = settle_pose[:2] - push_dir * self._push_stop_backoff
        settle_pose[2] = min(
            upper_z_limit - 0.01,
            effector_pos[2] + max(self._push_settle_lift, self._push_stop_lift),
        )
        self._push_settle_pose = settle_pose.astype(np.float32)
        self._mode = 'push_settle'
        self._mode_step = 0

    def _push_settle_action(self, info, current_push_dist_xy):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        if not np.all(np.isfinite(self._push_settle_pose)):
            self._set_push_settle_mode(info)

        self.print_phase('push-5: pause after push and observe')
        action = self._move_effector_action(
            effector_pos,
            self._push_settle_pose,
            gripper_cmd=-1.0,
            gain_pos=3.0,
        )
        if self._mode_step >= self._push_settle_steps:
            if self._done or current_push_dist_xy <= self._push_stop_dist or self._current_target_success(info):
                self._done = True
            else:
                self._mode = 'push'
                self._mode_step = 0
                self._push_drive_started = False
                self._active_push_plan = self._build_push_plan(info)
        return action

    def _current_target_success(self, info):
        target_block = int(info.get('privileged/target_block', -1))
        if target_block < 0:
            return False
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        if not hasattr(env_unwrapped, '_compute_successes'):
            return False
        try:
            successes = env_unwrapped._compute_successes()
        except Exception:
            return False
        return 0 <= target_block < len(successes) and bool(successes[target_block])

    def _smooth_push_drive_action(self, info, push_plan, current_push_dist_xy):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        target_block = int(push_plan.get('target_block', -1))
        _, block_pos, _ = self._extract_target_info(info)
        push_center_xy = self._push_center_xy(target_block, block_pos)
        push_dir = self._normalize_xy(push_plan['push_dir'])
        lateral_dir = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)

        lateral_error = float(np.dot(push_center_xy - effector_pos[:2], lateral_dir))
        lateral_correction = float(np.clip(lateral_error * self._push_lateral_correction, -0.012, 0.012))

        desired_xy = np.asarray(push_plan['push_goal_xy'], dtype=np.float32)
        desired_xy = desired_xy + lateral_dir * lateral_correction

        desired_pos = np.array([desired_xy[0], desired_xy[1], push_plan['push_z']], dtype=np.float32)
        desired_yaw = self._push_yaw(push_dir)
        gain_pos = self._push_drive_gain
        if current_push_dist_xy <= self._push_fine_dist:
            gain_pos = min(gain_pos, self._push_fine_drive_gain)
        elif current_push_dist_xy <= self._push_slowdown_dist:
            gain_pos = min(gain_pos, self._push_slow_drive_gain)

        self.print_phase('push-4: smooth continuous drive')
        return self._move_pose_action(info, desired_pos, desired_yaw, gripper_cmd=-1.0, gain_pos=gain_pos)

    def _gripper_contact_strength(self, info):
        try:
            return float(np.asarray(info.get('proprio/gripper_contact', [0.0]), dtype=np.float32).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return 0.0

    def _target_robot_contact(self, target_block):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        geom_ids_list = getattr(env_unwrapped, '_object_geom_ids_list', None)
        if geom_ids_list is None or not (0 <= target_block < len(geom_ids_list)):
            return False

        target_geom_ids = set(int(gid) for gid in geom_ids_list[target_block])
        model = getattr(env_unwrapped, '_model', None)
        data = getattr(env_unwrapped, '_data', None)
        if model is None or data is None:
            return False

        robot_tokens = ('robotiq', 'ur5e', 'finger', 'pad', 'gripper')
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in target_geom_ids:
                other_geom = geom2
            elif geom2 in target_geom_ids:
                other_geom = geom1
            else:
                continue
            try:
                other_name = str(model.geom(other_geom).name).lower()
            except Exception:
                other_name = ''
            if any(token in other_name for token in robot_tokens):
                return True
        return False

    def _should_start_push_drive(self, info, push_plan, contact_pose, effector_pos):
        contact_strength = self._gripper_contact_strength(info)
        target_block = int(push_plan.get('target_block', -1))
        push_dir = self._normalize_xy(push_plan['push_dir'])
        lateral_dir = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)
        contact_error_xy = np.asarray(contact_pose[:2] - effector_pos[:2], dtype=np.float32)
        forward_error = float(np.dot(contact_error_xy, push_dir))
        lateral_error = abs(float(np.dot(contact_error_xy, lateral_dir)))
        z_error = abs(float(contact_pose[2] - effector_pos[2]))
        near_contact_corridor = (
            -max(0.02, 2.0 * self._push_contact_penetration)
            <= forward_error
            <= self._push_approach_clearance + self._push_contact_forward_tolerance
            and lateral_error <= max(0.05, self._push_contact_lateral_tolerance)
            and z_error <= max(0.06, self._push_contact_z_tolerance)
        )

        if near_contact_corridor and contact_strength >= self._push_force_contact_threshold:
            return True, 'force_contact'

        if near_contact_corridor and self._target_robot_contact(target_block):
            return True, 'mujoco_contact'

        # If the TCP has reached or slightly crossed the object's contact line, start pushing.
        geometric_contact = (
            -max(0.02, 2.0 * self._push_contact_penetration)
            <= forward_error
            <= self._push_contact_forward_tolerance
            and lateral_error <= self._push_contact_lateral_tolerance
            and z_error <= self._push_contact_z_tolerance
        )
        if geometric_contact:
            return True, 'geometric_contact'
        return False, 'waiting_contact'

    def _directional_push_action(self, info, push_plan):
        effector_pos = np.asarray(info.get('proprio/effector_pos', np.zeros(3)), dtype=np.float32)
        _, block_pos, target_pos = self._extract_target_info(info)
        current_push_dist_xy = float(np.linalg.norm(np.asarray(target_pos[:2] - block_pos[:2], dtype=np.float32)))
        target_delta_xy = np.asarray(target_pos[:2] - block_pos[:2], dtype=np.float32)
        overshot_target = float(np.dot(target_delta_xy, push_plan['push_dir'])) < -0.5 * self._push_stop_dist

        if self._mode == 'push_settle':
            self._push_debug(
                f'push-settle xy_dist={current_push_dist_xy:.4f} '
                f'stop_dist={self._push_stop_dist:.4f} mode_step={self._mode_step} done={self._done}'
            )
            if self._current_target_success(info):
                self._done = True
            return self._push_settle_action(info, current_push_dist_xy)

        if overshot_target and current_push_dist_xy > self._push_stop_dist and not self._push_drive_started:
            push_plan = self._build_push_plan(info)
            self._active_push_plan = push_plan

        if current_push_dist_xy <= self._push_stop_dist or self._current_target_success(info):
            self._set_push_settle_mode(info, push_plan)
            self._done = True
            self.print_phase('push-stop: pause at object edge')
            self._push_debug(
                f"push-stop target_block={int(info.get('privileged/target_block', -1))} "
                f'xy_dist={current_push_dist_xy:.4f} stop_dist={self._push_stop_dist:.4f} '
                f'mode={self._mode} mode_step={self._mode_step} done={self._done}'
            )
            return self._push_settle_action(info, current_push_dist_xy)

        approach_pose = np.array(
            [push_plan['approach_xy'][0], push_plan['approach_xy'][1], push_plan['approach_z']],
            dtype=np.float32,
        )
        upper_z_limit = self._get_push_upper_z_limit()
        transit_z = float(np.clip(self._push_transit_z, push_plan['approach_z'] + 0.04, upper_z_limit - 0.01))
        transit_pose = np.array(
            [push_plan['approach_xy'][0], push_plan['approach_xy'][1], transit_z],
            dtype=np.float32,
        )
        contact_pose = np.array(
            [push_plan['contact_xy'][0], push_plan['contact_xy'][1], push_plan['push_z']],
            dtype=np.float32,
        )
        desired_yaw = self._push_yaw(push_plan['push_dir'])
        contact_ready, contact_reason = self._should_start_push_drive(info, push_plan, contact_pose, effector_pos)
        is_plate_push = self._get_object_kind(int(push_plan.get('target_block', -1))) == 'plate'
        approach_xy_dist = float(np.linalg.norm(approach_pose[:2] - effector_pos[:2]))
        contact_xy_dist = float(np.linalg.norm(contact_pose[:2] - effector_pos[:2]))
        transit_xy_threshold = 0.06 if is_plate_push else 0.035
        approach_xy_threshold = 0.03 if is_plate_push else 0.02
        contact_xy_threshold = 0.025 if is_plate_push else 0.015
        transit_xy_aligned = approach_xy_dist <= transit_xy_threshold
        approach_xy_aligned = approach_xy_dist <= approach_xy_threshold
        approach_z_aligned = abs(float(approach_pose[2] - effector_pos[2])) <= 0.02
        contact_xy_aligned = contact_xy_dist <= contact_xy_threshold
        contact_z_aligned = abs(float(contact_pose[2] - effector_pos[2])) <= 0.012

        gain_pos = 5.0
        push_gain_pos = self._push_drive_gain
        if current_push_dist_xy <= self._push_fine_dist:
            push_gain_pos = min(push_gain_pos, self._push_fine_drive_gain)
        elif current_push_dist_xy <= self._push_slowdown_dist:
            push_gain_pos = min(push_gain_pos, self._push_slow_drive_gain)
        if self._push_drive_started:
            action = self._smooth_push_drive_action(info, push_plan, current_push_dist_xy)
        elif contact_ready:
            self._push_debug(
                f'push-contact-start reason={contact_reason} '
                f'contact_strength={self._gripper_contact_strength(info):.4f}'
            )
            self._push_drive_started = True
            action = self._smooth_push_drive_action(info, push_plan, current_push_dist_xy)
        elif is_plate_push and contact_xy_aligned and not contact_z_aligned:
            self.print_phase('push-3: descend directly to plate boundary contact')
            action = self._move_pose_action(info, contact_pose, desired_yaw, gripper_cmd=-1.0, gain_pos=gain_pos)
        elif not transit_xy_aligned:
            self.print_phase('push-1: high transit to approach side')
            action = self._move_pose_action(info, transit_pose, desired_yaw, gripper_cmd=-1.0, gain_pos=gain_pos)
        elif (not self._push_drive_started) and not (approach_xy_aligned and approach_z_aligned):
            self.print_phase('push-2: descend outside object footprint')
            action = self._move_pose_action(info, approach_pose, desired_yaw, gripper_cmd=-1.0, gain_pos=gain_pos)
        elif (not self._push_drive_started) and not (contact_xy_aligned and contact_z_aligned):
            self.print_phase('push-3: move to boundary contact')
            action = self._move_pose_action(info, contact_pose, desired_yaw, gripper_cmd=-1.0, gain_pos=gain_pos)
        else:
            self._push_drive_started = True
            action = self._smooth_push_drive_action(info, push_plan, current_push_dist_xy)

        return action

    def select_action(self, ob, info):
        target_block, block_pos, target_pos = self._extract_target_info(info)
        if self._active_target_block is None:
            self._active_target_block = int(target_block)
        elif int(target_block) != int(self._active_target_block):
            self._active_target_block = int(target_block)
            self._active_push_plan = None
            self._active_grasp_plan = None
            self._mode = 'edge_move_above'
            self._mode_step = 0
            self._push_drive_started = False
        target_dist = float(np.linalg.norm(target_pos - block_pos))
        target_dist_xy = float(np.linalg.norm(np.asarray(target_pos[:2] - block_pos[:2], dtype=np.float32)))
        allow_push = self._use_push_for_bottom_stack and self._is_bottom_push_target(info)
        push_plan = None
        already_at_nonpush_goal = (
            (not allow_push)
            and target_dist <= self._success_dist
            and self._active_grasp_plan is None
            and self._mode == 'edge_move_above'
        )
        if already_at_nonpush_goal:
            self._done = True
            self._step += 1
            self._mode_step += 1
            return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        if not allow_push:
            if not self._mode.startswith('edge_'):
                self._set_mode('edge_move_above', ob, info)
                self._active_grasp_plan = None
        else:
            if self._mode not in {'push', 'push_settle'}:
                self._set_mode('push', ob, info)
                self._active_push_plan = self._build_push_plan(info)
            elif (
                self._mode == 'push'
                and (not self._push_drive_started)
                and self._mode_step >= self._push_burst_steps
            ):
                self._set_push_settle_mode(info, self._active_push_plan)
            if self._mode in {'push', 'push_settle'}:
                if self._active_push_plan is None:
                    self._active_push_plan = self._build_push_plan(info)
                push_plan = self._active_push_plan

        if self._mode in {'push', 'push_settle'}:
            action = self._directional_push_action(info, push_plan)
        elif self._mode == 'pre_push_lift':
            action = self._pre_push_lift_action(info, push_plan)
        else:
            action = self._edge_grasp_action(ob, info)

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self._step += 1
        self._mode_step += 1

        done_dist = self._push_stop_dist if allow_push else self._success_dist
        reached_goal = target_dist_xy <= done_dist if allow_push else target_dist <= done_dist
        if allow_push and self._push_debug_enabled():
            self._push_debug(
                f"select_action target_block={int(info.get('privileged/target_block', -1))} "
                f'mode={self._mode} mode_step={self._mode_step} '
                f'xy_dist={target_dist_xy:.4f} dist3d={target_dist:.4f} '
                f'done_dist={done_dist:.4f} reached_goal={reached_goal} done_before={self._done}'
            )
        if self._step >= self._max_step:
            self._done = True
        if allow_push and self._push_debug_enabled():
            self._push_debug(
                f"post_decision target_block={int(info.get('privileged/target_block', -1))} "
                f'done_after={self._done} action={np.round(action, 4).tolist()}'
            )

        return action
