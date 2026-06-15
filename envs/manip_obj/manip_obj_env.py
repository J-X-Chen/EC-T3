from pathlib import Path
import re

import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
from dm_control import mjcf

import numpy as np
import torch

import ogbench
from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.env import CustomMuJoCoEnv


class ManipObjEnv(CustomMuJoCoEnv):
    """Compositional object environment modified from the OGBench CubeEnv.

    It contains a UR5e robot arm with a Robotiq 2F-85 gripper.
    The default control mode is relative end-effector control. The 5-D action space corresponds to the following:
    - 3-D relative end-effector position (x, y, z).
    - 1-D relative end-effector yaw.
    - 1-D relative gripper opening.

    This environment consists of a single or multiple objects. The goal is to move the objects to target configurations.
    """

    def __init__(
        self,
        ob_type='states',
        multiview=False,
        mode='task',
        reward_mode='step',
        reward_scale=1.0,
        num_cubes=1,
        object_type='cube',
        local_object_name='bowl',
        local_object_pack_root='local_object_packs',
        tableware_stack_base_min_offset=0.08,
        tableware_stack_base_max_offset=0.15,
        tableware_stack_base_clearance=0.015,
        tableware_stack_base_safe_clearance=0,#0.035,
        tableware_preclear_target_attempts=80,
        tableware_ppp_min_offset=0.0,
        tableware_ppp_max_offset=None,
        tableware_init_resample_attempts=5,
        num_colors=6,
        random_colors=False,
        terminate_at_goal=False,
        visualize_info=False,
        physics_timestep=0.002,
        control_timestep=0.05,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            ob_type: Observation type. Either 'states', 'states_clean' or 'pixels'.
                'states' correspond to the full state specified by OGBench.
                'states_clean' correspond to a subset of 'states' that excludes some information that is not necessary to solve the task (e.g., velocities).
            multiview: Whether to use multi-view pixel observations. Only applicable when ob_type is 'pixels'.
            mode: Mode of the environment. Either 'task' or 'data_collection'. In 'task' mode, the environment is used
                for training and evaluation. In 'data_collection' mode, the environment is used for collecting offline
                data. 'task_stack' and 'data_collection_stack' are variants that focus on stacking.
                'task_sort' is a variant for evaluating specific compositional generalization scenarios.
                Hybrid variants follow naming `task_cubeX+Y` / `data_collection_cubeX+Y`:
                X cubes must reach fixed pick-place targets and Y cubes must be stacked.
                (Backward-compatible aliases: task_stack_ppp5 / data_collection_stack_ppp5)
            reward_mode: Either 'sparse', 'step' or 'dense'.
                'sparse': reward of 0 when the task is completed, -1 otherwise.
                'step': stepwise sparse reward for each completed subtask. Reward is -n/N, where n is the number of remaining subtasks and N is the total number of subtasks.
                'dense': dense reward based on the distance between the objects and the target configurations.
            reward_scale: Scale factor for the dense reward.
            num_cubes: Number of objects in the environment.
                For tableware modes named as `...tablewareX+Y`, this can be left as 0 and will be inferred from X+Y.
                In tableware naming, X means pick-place-on-table subtasks and Y means stacking subtasks.
            object_type: Object geometry type. 'cube' uses OGBench cubes; any non-'cube' value uses local object packs.
            local_object_name: Local object folder name under '<local_object_pack_root>/assets/'.
                Tableware modes load semantic assets from '<local_object_pack_root>/assets/{kind}/',
                '<local_object_pack_root>/assets_main/{kind}/00/', or '<local_object_pack_root>/{kind}/00/'.
            local_object_pack_root: Root folder that stores generated local object assets.
            tableware_stack_base_min_offset: Minimum XY movement sampled for the bottom stack object.
            tableware_stack_base_max_offset: Maximum XY movement sampled for the bottom stack object.
            tableware_stack_base_clearance: Extra XY clearance from non-stack objects when sampling stack base targets.
            tableware_stack_base_safe_clearance: Extra XY clearance around the stack base target before pushing it.
                Set to 0 to disable pre-clearing blockers near the stack base target.
            tableware_preclear_target_attempts: Lightweight samples used when moving a blocker out of the stack base target.
            tableware_ppp_min_offset: Minimum XY distance between a tableware pick-place object's start and target.
            tableware_ppp_max_offset: Optional maximum XY distance between a tableware pick-place object's start and target.
            tableware_init_resample_attempts: Reset-time retries when tableware initialization has no valid target.
            num_colors: Number of colors to use for the objects.
            random_colors: Whether to use random colors for the objects.
                If so, colors are randomly assigned to objects at the beginning of each episode without repetition (unless mode=='task_sort').
            terminate_at_goal: Whether to terminate the episode when the goal is reached.
            visualize_info: Whether to visualize the task information (e.g., success status).
            physics_timestep: Physics timestep.
            control_timestep: Control timestep.
            **kwargs: Additional keyword arguments.
        """
                
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )
        self._hidden_collision_geom_group = 4
        self._scene_option.geomgroup[self._hidden_collision_geom_group] = 0

        # Define constants.
        self._desc_dir = Path(ogbench.manipspace.__file__).resolve().parent / 'descriptions'
        self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        self._effector_down_rotation = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))
        self._workspace_bounds = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
        self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._target_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._sort_goal_positions = np.asarray([[0.3, -0.2], [0.3, 0.2], [0.55, -0.2], [0.55, 0.2], [0.425, 0.0]])
        self._object_colors = np.array(
            [
                [0.96, 0.26, 0.33, 1.0],  # red
                [0.06, 0.74, 0.21, 1.0],  # green
                [0.35, 0.55, 0.91, 1.0],  # blue
                [1.00, 0.69, 0.21, 1.0],  # orange
                [0.61, 0.28, 0.82, 1.0],  # purple
                [0.00, 0.85, 0.85, 1.0],  # cyan / teal
                # [0.90, 0.90, 0.90, 1.0],  # white
                # [0.10, 0.10, 0.10, 1.0],  # black
            ]
        )
        self.agent_state_dim = 6 if ob_type == 'states_clean' else 19
        self.object_state_dim = 5 if ob_type == 'states_clean' else 9

        self._ob_type = ob_type
        self._multiview = multiview
        self._mode = mode
        self._reward_mode = reward_mode
        self._reward_scale = reward_scale

        # For tableware modes (e.g., data_collection_tableware2+0), infer object count from mode by default.
        # Naming rule: tablewareX+Y where X is pick-place count and Y is stack count.
        tableware_split = self._parse_tableware_split_from_mode(self._mode)
        if tableware_split is not None:
            inferred_tableware_count = int(tableware_split[0] + tableware_split[1])
            if inferred_tableware_count <= 0:
                raise ValueError(f"Mode '{self._mode}' implies zero tableware objects.")
            if num_cubes <= 0 or int(num_cubes) != inferred_tableware_count:
                num_cubes = inferred_tableware_count

        self.num_cubes = int(num_cubes)
        self.num_objects = self.num_cubes
        self._object_type = object_type.lower()
        self._local_object_name = local_object_name
        self._tableware_stack_base_min_offset = max(0.0, float(tableware_stack_base_min_offset))
        self._tableware_stack_base_max_offset = max(
            self._tableware_stack_base_min_offset,
            float(tableware_stack_base_max_offset),
        )
        self._tableware_stack_base_clearance = max(0.0, float(tableware_stack_base_clearance))
        self._tableware_stack_base_safe_clearance = max(0.0, float(tableware_stack_base_safe_clearance))
        self._tableware_preclear_target_attempts = max(1, int(tableware_preclear_target_attempts))
        self._tableware_ppp_min_offset = max(0.0, float(tableware_ppp_min_offset))
        self._tableware_ppp_max_offset = (
            None
            if tableware_ppp_max_offset is None
            else max(self._tableware_ppp_min_offset, float(tableware_ppp_max_offset))
        )
        self._tableware_init_resample_attempts = max(1, int(tableware_init_resample_attempts))
        self._project_root = Path(__file__).resolve().parents[2]
        local_pack_root = Path(local_object_pack_root).expanduser()
        if not local_pack_root.is_absolute():
            local_pack_root = self._project_root / local_pack_root
        self._local_object_pack_root = local_pack_root.resolve()
        self._num_colors = num_colors
        self._random_colors = random_colors
        self._terminate_at_goal = terminate_at_goal
        self._visualize_info = visualize_info

        # Hybrid mode: naming rule "cubeX+Y" means X fixed PPP targets + Y stack targets.
        # This also supports edge cases "cube0+Y" (pure stack) and "cubeX+0" (pure fixed PPP).
        # Backward compatibility is kept for old names: task_stack_ppp5 / data_collection_stack_ppp5.
        cube_split = self._parse_cube_split_from_mode(self._mode)
        if cube_split is not None:
            self._hybrid_ppp_count, self._hybrid_stack_count = cube_split
            if (self._hybrid_ppp_count + self._hybrid_stack_count) <= 0:
                raise ValueError(f"Mode '{self._mode}' implies zero cubes.")
            self._is_hybrid_mode = True
        elif self._mode in {'task_stack_ppp5', 'data_collection_stack_ppp5'}:
            self._hybrid_ppp_count, self._hybrid_stack_count = 2, 3
            self._is_hybrid_mode = True
        else:
            self._hybrid_ppp_count, self._hybrid_stack_count = 0, 0
            self._is_hybrid_mode = False
        self._stack_block_indices = list(range(self._hybrid_stack_count))
        self._ppp_block_indices = list(range(self._hybrid_stack_count, self._hybrid_stack_count + self._hybrid_ppp_count))
        self._hybrid_stack_base_xy = np.array([0.425, 0.0], dtype=np.float32)
        self._hybrid_stack_height_step = 0.04
        self._hybrid_ppp_target_xys = self._default_ppp_target_xys(self._hybrid_ppp_count)
        self._hybrid_stack_xy_threshold = 0.02
        self._hybrid_stack_z_threshold = 0.02
        self._hybrid_ppp_pos_threshold = 0.04
        self._target_task_group = 'pickplace'
        self._tableware_ppp_target_positions = {}
        self._tableware_ppp_target_oris = {}
        
        assert ob_type in ['states', 'states_clean', 'pixels']
        assert reward_mode in ['sparse', 'step', 'dense'], f"Invalid reward mode: {reward_mode}."
        assert self.num_cubes > 0, "At least one object must be present in the environment."
        assert len(self._object_type) > 0, "object_type must be a non-empty string."
        assert num_colors <= len(self._object_colors), f"Number of colors ({num_colors}) exceeds available colors ({len(self._object_colors)})."
        if self._is_hybrid_mode:
            expected_num_objects = self._hybrid_ppp_count + self._hybrid_stack_count
            assert self.num_objects == expected_num_objects, (
                f"Mode '{self._mode}' expects exactly {expected_num_objects} cubes, got {self.num_objects}."
            )

        self._hybrid_stack_target_positions, self._hybrid_ppp_target_positions = self._build_hybrid_target_positions()

        # Initialize inverse kinematics controller.
        ik_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        # Define action space.
        action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
        self.action_low = -action_range
        self.action_high = action_range

        if 'task' in self._mode:
            # Set task goals.
            self._cur_goal_ob = None
            self._cur_goal_rendered = None
            self._render_goal = False
        else:
            # Target info.
            self._target_task = 'push_cube'
            # The target cube position is stored in the mocap object.
            self._target_block = 0
            self._stacking = False

        self._success = False

    @property
    def observation_space(self):#gymnasium硬性要求
        if self._ob_type == 'pixels':
            image_shape = (self._render_height, self._render_width, 3)
            if self._multiview:
                image_shape = (2, *image_shape)
            return Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
        else:
            ob_dim = self.agent_state_dim + self.num_objects * self.object_state_dim
            return Box(low=-np.inf, high=np.inf, shape=(ob_dim,), dtype=np.float64)#self._ob_type=='states_clean'

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-np.ones(5),
            high=np.ones(5),
            shape=(5,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        """Normalize the action to the range [-1, 1]."""
        action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return np.clip(action, -1, 1)

    def unnormalize_action(self, action):
        """Unnormalize the action to the range [action_low, action_high]."""
        return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low

    def build_mjcf_model(self):
        # Set scene.
        arena_mjcf = mjcf.from_path((self._desc_dir / 'floor_wall.xml').as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # Add UR5e robot arm.
        ur5e_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()
            del light

        # Attach the robotiq gripper to the UR5e flange.
        gripper_mjcf = mjcf.from_path((self._desc_dir / 'robotiq_2f85' / '2f85.xml'), escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach UR5e to the scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        self.add_objects(arena_mjcf)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        # Add bounding boxes to visualize the workspace and object sampling bounds.
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name='arm_bounds',
        )

        return arena_mjcf

    def add_objects(self, arena_mjcf):
        # Add objects.
        self._object_geoms_list = []
        self._object_target_geoms_list = []
        if self.num_cubes > 0:
            if self._object_type == 'cube':
                self.add_cubes(arena_mjcf)
            elif 'tableware' in self._mode:
                self.add_tableware_objects(arena_mjcf)
            else:
                self.add_bowls(arena_mjcf)
        
        # Add cameras.
        cameras = {
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            'side_pixels': {
                'pos': (0.414, -0.753, 0.639),
                'xyaxes': (1.000, 0.000, 0.000, -0.001, 0.628, 0.778),
            },
            'side_pixels_stack': {
                'pos': (0.400, -0.987, 0.509),
                'xyaxes': (1.000, 0.000, 0.000, 0.00, 0.342, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def add_cubes(self, arena_mjcf):
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self.num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self.num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            cube_mjcf.find('geom', 'object_0').density = 8000  # NOTE: change default density of OGBench cube (1240), dataset collected with 8000
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        for i in range(self.num_cubes):
            self._object_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        for i in range(self.num_cubes):
            self._object_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))
        #import pdb; pdb.set_trace()

    def add_bowls(self, arena_mjcf):
        """Add local-pack objects generated by object_generation.py."""
        bowl_assets_dir = self._local_object_pack_root / 'assets' / self._local_object_name
        bowl_outer_path = bowl_assets_dir / 'outer.xml'
        bowl_inner_path = bowl_assets_dir / 'inner.xml'
        if not bowl_outer_path.exists() or not bowl_inner_path.exists():
            raise FileNotFoundError(
                f"Missing bowl asset XML files under {bowl_assets_dir}. "
                f"Run object_generation.py first to create inner.xml/outer.xml."
            )

        bowl_outer_mjcf = mjcf.from_path(bowl_outer_path.as_posix())
        arena_mjcf.include_copy(bowl_outer_mjcf)

        distance = 0.05
        for i in range(self.num_cubes):
            bowl_mjcf = mjcf.from_path(bowl_inner_path.as_posix())
            pos = -distance * (self.num_cubes - 1) + 2 * distance * i
            bowl_body = bowl_mjcf.find('body', 'object_0')
            bowl_target_body = bowl_mjcf.find('body', 'object_target_0')
            if bowl_body is None or bowl_target_body is None:
                raise ValueError(
                    "Bowl inner XML must contain bodies named 'object_0' and 'object_target_0'."
                )
            bowl_body.pos[1] = pos
            bowl_target_body.pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in bowl_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None:
                        item.name = self._rename_local_object_item(item.name, i)
            arena_mjcf.include_copy(bowl_mjcf)

        for i in range(self.num_cubes):
            self._object_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        for i in range(self.num_cubes):
            self._object_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))

    def _parse_tableware_split_from_mode(self, mode):
        """Parse tableware naming rule from mode, e.g., data_collection_tableware2+0.

        Returns:
            (pickplace_count, stack_count)
        """
        match = re.search(r'tableware(\d+)\+(\d+)', mode)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def _cycle_tableware_kinds(self, preferred_order, count):
        if count <= 0:
            return []
        return [preferred_order[i % len(preferred_order)] for i in range(count)]

    def _build_tableware_object_kinds(self):
        """Build per-object semantic labels for generated tableware assets.

        Object ids are role ordered: pick-place objects first, then stack objects
        from bottom to top. Thus tableware0+2 is plate -> bowl, tableware1+2
        is mug -> plate -> bowl, and tableware3+2 is mug -> fork -> knife
        followed by plate -> bowl.
        """
        if self.num_objects <= 0:
            self._tableware_ppp_block_indices = []
            self._tableware_stack_block_indices = []
            return []

        tableware_split = self._parse_tableware_split_from_mode(self._mode)
        if tableware_split is None:
            pickplace_count, stack_count = self.num_objects, 0
        else:
            pickplace_count, stack_count = tableware_split

        if stack_count > 0:
            if pickplace_count >= 3:
                pickplace_order = ['mug', 'fork', 'knife', 'bowl', 'plate']
            else:
                pickplace_order = ['mug', 'bowl', 'plate']
        else:
            pickplace_order = ['bowl', 'plate', 'mug', 'fork', 'knife']
        stack_order = ['plate', 'bowl', 'mug']

        pickplace_kinds = self._cycle_tableware_kinds(pickplace_order, pickplace_count)
        stack_kinds = self._cycle_tableware_kinds(stack_order, stack_count)
        object_kinds = pickplace_kinds + stack_kinds

        if len(object_kinds) < self.num_objects:
            object_kinds.extend(self._cycle_tableware_kinds(
                ['bowl', 'plate', 'mug', 'fork', 'knife'],
                self.num_objects - len(object_kinds),
            ))
        object_kinds = object_kinds[: self.num_objects]

        self._tableware_ppp_block_indices = list(range(min(pickplace_count, self.num_objects)))
        stack_start = len(self._tableware_ppp_block_indices)
        stack_end = min(stack_start + stack_count, self.num_objects)
        self._tableware_stack_block_indices = list(range(stack_start, stack_end))
        return object_kinds

    def _tableware_dual_asset_paths(self, kind):
        assets_dir = self._local_object_pack_root / 'assets' / kind
        outer_path = assets_dir / 'outer.xml'
        inner_path = assets_dir / 'inner.xml'
        if outer_path.exists() and inner_path.exists():
            return outer_path, inner_path
        return None

    def _tableware_asset_xml_path(self, kind):
        asset_xml_candidates = [
            self._local_object_pack_root / 'assets_main' / kind / '00' / 'object.xml',
            self._local_object_pack_root / kind / '00' / 'object.xml',
            self._project_root / 'extra_assets' / 'glb_style' / kind / '00' / 'object.xml',
        ]
        for asset_xml_path in asset_xml_candidates:
            if asset_xml_path.exists():
                return asset_xml_path

        searched_paths = ', '.join(str(path) for path in asset_xml_candidates)
        raise FileNotFoundError(
            f"Missing tableware asset XML for '{kind}'. "
            f"Expected either {self._local_object_pack_root / 'assets' / kind / 'outer.xml'} + "
            f"{self._local_object_pack_root / 'assets' / kind / 'inner.xml'}, "
            f"or one of: {searched_paths}."
        )

    def _default_tableware_profile(self, kind):
        profiles = {
            'bowl': {'bottom_z': -0.03, 'top_z': 0.03, 'horizontal_radius': 0.06},
            'plate': {'bottom_z': -0.01, 'top_z': 0.01, 'horizontal_radius': 0.11},
            'mug': {'bottom_z': -0.05, 'top_z': 0.07, 'horizontal_radius': 0.05},
            'fork': {'bottom_z': -0.01, 'top_z': 0.01, 'horizontal_radius': 0.08},
            'knife': {'bottom_z': -0.01, 'top_z': 0.01, 'horizontal_radius': 0.11},
        }
        return dict(profiles.get(kind, {'bottom_z': -0.02, 'top_z': 0.02, 'horizontal_radius': 0.04}))

    def _extract_tableware_profile(self, item_mjcf, kind):
        profile = self._default_tableware_profile(kind)
        site_to_key = {
            'bottom_site': 'bottom_z',
            'top_site': 'top_z',
            'horizontal_radius_site': 'horizontal_radius',
        }
        for site in item_mjcf.find_all('site'):
            site_name = site.name or ''
            key = site_to_key.get(site_name)
            if key is None:
                for prefix, candidate_key in site_to_key.items():
                    if site_name.startswith(prefix + '_'):
                        key = candidate_key
                        break
            if key is None or site.pos is None:
                continue
            if key == 'horizontal_radius':
                profile[key] = float(site.pos[0])
            else:
                profile[key] = float(site.pos[2])
        return profile

    def _tableware_resting_z_from_profile(self, profile):
        return max(0.0, -float(profile.get('bottom_z', -0.02)))

    def _tableware_resting_z(self, object_idx):
        profiles = getattr(self, '_tableware_object_profiles', None)
        if profiles is None or not (0 <= object_idx < len(profiles)):
            return 0.02
        return self._tableware_resting_z_from_profile(profiles[object_idx])

    def _tableware_horizontal_radius(self, object_idx, default=0.04):
        profiles = getattr(self, '_tableware_object_profiles', None)
        if profiles is None or not (0 <= object_idx < len(profiles)):
            return float(default)
        return float(profiles[object_idx].get('horizontal_radius', default))

    def _tableware_max_horizontal_radius(self, object_indices=None, default=0.04):
        profiles = getattr(self, '_tableware_object_profiles', None)
        if profiles is None or len(profiles) == 0:
            return float(default)
        if object_indices is None:
            object_indices = range(len(profiles))
        radii = [
            self._tableware_horizontal_radius(int(idx), default=default)
            for idx in object_indices
            if 0 <= int(idx) < len(profiles)
        ]
        return max(radii) if radii else float(default)

    def _with_object_resting_z(self, object_idx, pos):
        pos = np.asarray(pos, dtype=np.float32).copy()
        if hasattr(self, '_tableware_object_profiles'):
            pos[2] = self._tableware_resting_z(object_idx)
        return pos

    def _get_or_sample_tableware_ppp_target(self, ppp_block, block_xyzs):
        ppp_block = int(ppp_block)
        if not hasattr(self, '_tableware_ppp_target_positions'):
            self._tableware_ppp_target_positions = {}
        if not hasattr(self, '_tableware_ppp_target_oris'):
            self._tableware_ppp_target_oris = {}

        if ppp_block in self._tableware_ppp_target_positions and ppp_block in self._tableware_ppp_target_oris:
            return (
                np.asarray(self._tableware_ppp_target_positions[ppp_block], dtype=np.float32).copy(),
                list(self._tableware_ppp_target_oris[ppp_block]),
            )

        avoid_blocks = [i for i in range(self.num_objects) if i != ppp_block]
        avoid_xyzs = block_xyzs[avoid_blocks] if len(avoid_blocks) > 0 else np.empty((0, 3), dtype=np.float32)
        current_xy = np.asarray(block_xyzs[ppp_block], dtype=np.float32)[:2]
        ppp_tar_pos = self.sample_obj_pos(
            np.array(avoid_xyzs),
            object_idx=ppp_block,
            sampled_object_indices=avoid_blocks,
            reference_xy=current_xy,
            min_reference_dist=self._tableware_ppp_min_offset,
            max_reference_dist=self._tableware_ppp_max_offset,
        )
        ppp_tar_pos = self._with_object_resting_z(ppp_block, ppp_tar_pos)
        ppp_tar_ori = lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
        self._tableware_ppp_target_positions[ppp_block] = ppp_tar_pos.copy()
        self._tableware_ppp_target_oris[ppp_block] = list(ppp_tar_ori)
        return ppp_tar_pos.copy(), list(ppp_tar_ori)

    def _build_tableware_stack_target_positions(self, stack_targets, base_xy):
        stack_positions = []
        support_top_z = 0.0
        profiles = getattr(self, '_tableware_object_profiles', [])
        for block_id in stack_targets:
            if 0 <= block_id < len(profiles):
                profile = profiles[block_id]
                bottom_z = float(profile.get('bottom_z', -0.02))
                top_z = float(profile.get('top_z', 0.02))
            else:
                bottom_z, top_z = -0.02, 0.02
            object_z = support_top_z - bottom_z
            stack_positions.append(np.array([base_xy[0], base_xy[1], object_z], dtype=np.float32))
            support_top_z = object_z + top_z
        return stack_positions

    def _tableware_stack_targets_complete(self, stack_targets, threshold=0.04):
        if len(stack_targets) == 0:
            return True

        stack_targets = [int(block_id) for block_id in stack_targets]
        stored_targets = getattr(self, '_stack_targets', [])
        stored_positions = getattr(self, '_stack_target_positions', [])
        use_stored_positions = (
            len(stored_targets) >= len(stack_targets)
            and len(stored_positions) >= len(stack_targets)
            and [int(i) for i in stored_targets[: len(stack_targets)]] == stack_targets
        )

        for idx, block_id in enumerate(stack_targets):
            if use_stored_positions:
                tar_pos = np.asarray(stored_positions[idx], dtype=np.float32)
            else:
                mocap_id = self._object_target_mocap_ids[block_id]
                tar_pos = np.asarray(self._data.mocap_pos[mocap_id], dtype=np.float32)
            if (not np.all(np.isfinite(tar_pos))) or float(tar_pos[2]) < -0.1:
                return False
            block_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
            if np.linalg.norm(block_pos - tar_pos) > float(threshold):
                return False
        return True

    def _clip_tableware_stack_base_xy(self, base_xy, stack_targets, clearance=None):
        base_xy = np.asarray(base_xy, dtype=np.float32).copy()
        clearance = self._tableware_stack_base_clearance if clearance is None else float(clearance)
        bounds = np.asarray(self._object_sampling_bounds, dtype=np.float32).copy()
        candidate_radius = self._tableware_max_horizontal_radius(stack_targets)
        side_margin = float(candidate_radius) + clearance + 0.03
        bounds[0, 1] = max(bounds[0, 1], self._workspace_bounds[0, 1] + side_margin)
        bounds[1, 1] = min(bounds[1, 1], self._workspace_bounds[1, 1] - side_margin)
        return np.clip(base_xy, bounds[0, :2], bounds[1, :2])

    def _sample_tableware_stack_base_near_current(
        self,
        base_block,
        stack_targets,
        block_xyzs,
        avoid_blocks=None,
        min_offset=None,
        max_offset=None,
        clearance=None,
    ):
        base_xy = np.asarray(block_xyzs[int(base_block)][:2], dtype=np.float32)
        avoid_blocks = [] if avoid_blocks is None else [int(i) for i in avoid_blocks]
        min_offset = self._tableware_stack_base_min_offset if min_offset is None else float(min_offset)
        max_offset = self._tableware_stack_base_max_offset if max_offset is None else float(max_offset)
        max_offset = max(min_offset, max_offset)
        clearance = self._tableware_stack_base_clearance if clearance is None else float(clearance)
        candidate_radius = self._tableware_max_horizontal_radius(stack_targets)

        best_xy = base_xy.copy()
        best_offset = 0.0
        for attempt in range(300):
            widen = 1.0 + 0.25 * (attempt // 75)
            radius = float(self.np_random.uniform(min_offset, max_offset * widen))
            theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            offset = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * radius
            candidate_xy = self._clip_tableware_stack_base_xy(
                base_xy + offset,
                stack_targets,
                clearance=clearance,
            )
            actual_offset = float(np.linalg.norm(candidate_xy - base_xy))
            if actual_offset > best_offset:
                best_xy = candidate_xy.copy()
                best_offset = actual_offset
            if actual_offset < min_offset * 0.85:
                continue

            valid = True
            for avoid_block in avoid_blocks:
                avoid_xy = np.asarray(block_xyzs[avoid_block][:2], dtype=np.float32)
                avoid_radius = self._tableware_horizontal_radius(avoid_block)
                min_dist = candidate_radius + avoid_radius + clearance
                if float(np.linalg.norm(candidate_xy - avoid_xy)) <= min_dist:
                    valid = False
                    break
            if valid:
                return np.array([candidate_xy[0], candidate_xy[1], block_xyzs[int(base_block)][2]], dtype=np.float32)

        return np.array([best_xy[0], best_xy[1], block_xyzs[int(base_block)][2]], dtype=np.float32)

    def _tableware_stack_base_safe_distance(self, base_block, other_block, clearance=None):
        clearance = (
            self._tableware_stack_base_safe_clearance
            if clearance is None
            else float(clearance)
        )
        return (
            self._tableware_horizontal_radius(int(base_block))
            + self._tableware_horizontal_radius(int(other_block))
            + max(0.0, clearance)
        )

    def _find_tableware_stack_base_blocker(
        self,
        base_block,
        base_xy,
        block_xyzs,
        candidate_blocks=None,
        clearance=None,
    ):
        clearance = (
            self._tableware_stack_base_safe_clearance
            if clearance is None
            else float(clearance)
        )
        if clearance <= 0.0:
            return None

        base_block = int(base_block)
        base_xy = np.asarray(base_xy, dtype=np.float32)[:2]
        if candidate_blocks is None:
            candidate_blocks = range(self.num_objects)

        best_block = None
        best_violation = 0.0
        for block_id in candidate_blocks:
            block_id = int(block_id)
            if block_id == base_block:
                continue
            block_xy = np.asarray(block_xyzs[block_id], dtype=np.float32)[:2]
            safe_dist = self._tableware_stack_base_safe_distance(
                base_block,
                block_id,
                clearance=clearance,
            )
            violation = safe_dist - float(np.linalg.norm(block_xy - base_xy))
            if violation > best_violation:
                best_violation = violation
                best_block = block_id

        return best_block

    def _sample_tableware_preclear_target(
        self,
        blocker_block,
        base_block,
        base_xy,
        block_xyzs,
        clearance=None,
        attempts=None,
    ):
        blocker_block = int(blocker_block)
        base_block = int(base_block)
        base_xy = np.asarray(base_xy, dtype=np.float32)[:2]
        block_xyzs = np.asarray(block_xyzs, dtype=np.float32)
        safe_clearance = (
            self._tableware_stack_base_safe_clearance
            if clearance is None
            else max(0.0, float(clearance))
        )
        attempts = self._tableware_preclear_target_attempts if attempts is None else int(attempts)
        attempts = max(1, attempts)

        blocker_radius = self._tableware_horizontal_radius(blocker_block)
        safe_dist = self._tableware_stack_base_safe_distance(
            base_block,
            blocker_block,
            clearance=safe_clearance,
        )
        object_clearance = self._tableware_stack_base_clearance
        other_blocks = [i for i in range(self.num_objects) if i != blocker_block]

        sampling_bounds = np.asarray(self._object_sampling_bounds, dtype=np.float32).copy()
        side_margin = float(blocker_radius) + float(object_clearance) + 0.03
        sampling_bounds[0, 1] = max(sampling_bounds[0, 1], self._workspace_bounds[0, 1] + side_margin)
        sampling_bounds[1, 1] = min(sampling_bounds[1, 1], self._workspace_bounds[1, 1] - side_margin)
        if np.any(sampling_bounds[0] > sampling_bounds[1]):
            sampling_bounds = np.asarray(self._object_sampling_bounds, dtype=np.float32).copy()

        best_xy = None
        best_score = -np.inf
        found_valid = False
        for _ in range(attempts):
            xy_pos = self.np_random.uniform(*sampling_bounds).astype(np.float32)
            base_margin = float(np.linalg.norm(xy_pos - base_xy)) - safe_dist
            other_margin = np.inf
            for other_block in other_blocks:
                other_xy = np.asarray(block_xyzs[other_block], dtype=np.float32)[:2]
                required_dist = (
                    blocker_radius
                    + self._tableware_horizontal_radius(other_block)
                    + object_clearance
                )
                margin = float(np.linalg.norm(xy_pos - other_xy)) - required_dist
                other_margin = min(other_margin, margin)
            score = min(base_margin, other_margin)
            if score > best_score:
                best_score = score
                best_xy = xy_pos.copy()
            if base_margin >= 0.0 and other_margin >= 0.0:
                best_xy = xy_pos.copy()
                found_valid = True
                break

        if best_xy is None:
            current_xy = np.asarray(block_xyzs[blocker_block], dtype=np.float32)[:2]
            away_dir = current_xy - base_xy
            norm = float(np.linalg.norm(away_dir))
            if norm < 1e-6:
                theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                away_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            else:
                away_dir = away_dir / norm
            best_xy = base_xy + away_dir * (safe_dist + object_clearance)
            best_xy = np.clip(best_xy, sampling_bounds[0, :2], sampling_bounds[1, :2])
        elif not found_valid:
            corners = np.array(
                [
                    [sampling_bounds[0, 0], sampling_bounds[0, 1]],
                    [sampling_bounds[0, 0], sampling_bounds[1, 1]],
                    [sampling_bounds[1, 0], sampling_bounds[0, 1]],
                    [sampling_bounds[1, 0], sampling_bounds[1, 1]],
                ],
                dtype=np.float32,
            )
            farthest_corner = corners[np.argmax(np.linalg.norm(corners - base_xy, axis=1))]
            if np.linalg.norm(farthest_corner - base_xy) > np.linalg.norm(best_xy - base_xy):
                best_xy = farthest_corner.copy()

        target_pos = np.array([best_xy[0], best_xy[1], block_xyzs[blocker_block][2]], dtype=np.float32)
        return self._with_object_resting_z(blocker_block, target_pos)

    def _make_tableware_dual_asset_instance(self, kind, object_idx, y_pos):
        dual_paths = self._tableware_dual_asset_paths(kind)
        if dual_paths is None:
            return None, None
        _, inner_path = dual_paths
        item_mjcf = mjcf.from_path(inner_path.as_posix())

        object_body = item_mjcf.find('body', 'object_0')
        target_body = item_mjcf.find('body', 'object_target_0')
        if object_body is None or target_body is None:
            raise ValueError(
                f"HECRL tableware inner XML for '{kind}' must contain bodies "
                "'object_0' and 'object_target_0'."
            )

        profile = self._extract_tableware_profile(item_mjcf, kind)
        resting_z = self._tableware_resting_z_from_profile(profile)
        object_body.pos[1] = y_pos
        object_body.pos[2] = resting_z
        target_body.pos[1] = y_pos
        target_body.pos[2] = resting_z

        for tag in ['body', 'joint', 'geom', 'site']:
            for item in item_mjcf.find_all(tag):
                if hasattr(item, 'name') and item.name is not None:
                    item.name = self._rename_local_object_item(item.name, object_idx)

        return item_mjcf, profile

    def _make_tableware_asset_instance(self, kind, object_idx, y_pos):
        dual_item_mjcf, dual_profile = self._make_tableware_dual_asset_instance(kind, object_idx, y_pos)
        if dual_item_mjcf is not None:
            return dual_item_mjcf, dual_profile

        item_mjcf = mjcf.from_path(self._tableware_asset_xml_path(kind).as_posix())
        item_mjcf.model = f'{kind}_{object_idx}'

        bodies = list(item_mjcf.find_all('body'))
        if len(bodies) == 0:
            raise ValueError(f"Tableware asset '{kind}' must contain at least one body.")
        root_body = bodies[0]
        profile = self._extract_tableware_profile(item_mjcf, kind)
        resting_z = self._tableware_resting_z_from_profile(profile)

        root_body.name = f'object_{object_idx}'
        root_body.pos = (0.3, y_pos, resting_z)

        visual_geoms = [geom for geom in item_mjcf.find_all('geom') if geom.name == 'g_visual']
        if len(visual_geoms) == 0:
            raise ValueError(f"Tableware asset '{kind}' must contain a visual geom named 'g_visual'.")
        visual_geom = visual_geoms[0]
        self._hide_tableware_collision_geoms(item_mjcf, visual_geom)

        for tag in ['body', 'joint', 'geom', 'site', 'mesh', 'texture', 'material']:
            for item in item_mjcf.find_all(tag):
                if item is root_body:
                    continue
                if hasattr(item, 'name') and item.name is not None:
                    item.name = f'{kind}_{object_idx}_{item.name}'

        root_body.add('freejoint', name=f'object_joint_{object_idx}')

        target_body = item_mjcf.worldbody.add(
            'body',
            name=f'object_target_{object_idx}',
            pos=(0.45, y_pos, resting_z),
            mocap=True,
        )
        target_body.add(
            'geom',
            name=f'target_object_{object_idx}',
            type='mesh',
            pos=visual_geom.pos,
            quat=visual_geom.quat,
            mesh=visual_geom.mesh,
            material=visual_geom.material,
            rgba=(0.5, 0.5, 0.5, 0.2),
            contype=0,
            conaffinity=0,
            group=1,
        )
        return item_mjcf, profile

    def _hide_tableware_collision_geoms(self, item_mjcf, visual_geom):
        """Render only the textured visual mesh while keeping collision geoms physical."""
        for geom in item_mjcf.find_all('geom'):
            if geom is visual_geom:
                geom.group = 1
                continue
            if getattr(geom, 'contype', 0) or getattr(geom, 'conaffinity', 0):
                geom.group = self._hidden_collision_geom_group
                geom.rgba = (0.0, 0.0, 0.0, 0.0)

    def add_tableware_objects(self, arena_mjcf):
        """Add mixed tableware objects from generated local assets."""
        object_kinds = self._build_tableware_object_kinds()
        self._tableware_object_kinds = object_kinds
        self._tableware_object_profiles = []

        included_dual_outers = set()
        for kind in object_kinds:
            dual_paths = self._tableware_dual_asset_paths(kind)
            if dual_paths is None or kind in included_dual_outers:
                continue
            outer_path, _ = dual_paths
            arena_mjcf.include_copy(mjcf.from_path(outer_path.as_posix()))
            included_dual_outers.add(kind)

        distance = 0.05
        for i, kind in enumerate(object_kinds):
            y_pos = -distance * (self.num_cubes - 1) + 2 * distance * i
            item_mjcf, profile = self._make_tableware_asset_instance(kind, i, y_pos)
            self._tableware_object_profiles.append(profile)
            arena_mjcf.include_copy(item_mjcf)

        for i in range(self.num_cubes):
            self._object_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        for i in range(self.num_cubes):
            self._object_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))

    @staticmethod
    def _rename_local_object_item(name, object_idx):
        replacements = [
            ('object_target_0', f'object_target_{object_idx}'),
            ('target_object_0', f'target_object_{object_idx}'),
            ('object_joint_0', f'object_joint_{object_idx}'),
            ('object_0', f'object_{object_idx}'),
            ('com_0', f'com_{object_idx}'),
            ('bottom_site_0', f'bottom_site_{object_idx}'),
            ('top_site_0', f'top_site_{object_idx}'),
            ('horizontal_radius_site_0', f'horizontal_radius_site_{object_idx}'),
        ]
        for old_prefix, new_prefix in replacements:
            if name == old_prefix:
                return new_prefix
            if name.startswith(old_prefix + '_'):
                return new_prefix + name[len(old_prefix) :]
        return name

    def post_compilation(self):  # 将 MJCF XML 转换为 MuJoCo 模型。此时方块已经存在于模型中
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -200, -200, -50])
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        if self._object_type != 'cube':
            # Tableware/grasp-helper assets are harder to lift than cubes.
            # Use moderate gripper force to avoid contact explosions.
            self._model.actuator_forcerange[self._gripper_actuator_ids] = np.asarray([[-16.0, 16.0]])
            for pad_name, sliding_friction in (
                ('ur5e/robotiq/right_pad1', 0.85),
                ('ur5e/robotiq/right_pad2', 0.75),
                ('ur5e/robotiq/left_pad1', 0.85),
                ('ur5e/robotiq/left_pad2', 0.75),
            ):
                pad_id = self._model.geom(pad_name).id
                self._model.geom_friction[pad_id, 0] = max(
                    self._model.geom_friction[pad_id, 0],
                    sliding_friction,
                )

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

        self.post_compilation_objects()

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._object_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_geoms] for object_geoms in self._object_geoms_list
        ]
        self._object_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self.num_objects)
        ]
        self._object_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_target_geoms]
            for object_target_geoms in self._object_target_geoms_list
        ]

        if self._object_type != 'cube':
            # Stabilize free-body dynamics for tableware-like objects.
            lin_damping = 4.0
            ang_damping = 0.4
            lin_armature = 0.005
            ang_armature = 0.001
            for i in range(self.num_objects):
                joint_id = self._model.joint(f'object_joint_{i}').id
                dof_adr = self._model.jnt_dofadr[joint_id]
                self._model.dof_damping[dof_adr : dof_adr + 3] = lin_damping
                self._model.dof_damping[dof_adr + 3 : dof_adr + 6] = ang_damping
                self._model.dof_armature[dof_adr : dof_adr + 3] = lin_armature
                self._model.dof_armature[dof_adr + 3 : dof_adr + 6] = ang_armature

    def reset(self, options=None, *args, **kwargs):
        if 'task' in self._mode:
            if options is None:
                options = {}
            # Whether to provide a rendering of the goal.
            self._render_goal = False
            if 'render_goal' in options:
                self._render_goal = options['render_goal']
        return super().reset(*args, **kwargs)#过ogbench的reset

    def initialize_arm(self):
        # Sample initial effector position and orientation.
        eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        #import pdb; pdb.set_trace()#看一下self._data是啥
        # Set object colors.
        if self._mode == 'task_sort':
            color_idx = np.array([i % self._num_colors for i in range(self.num_objects)])
        elif self._random_colors:
            color_idx = np.random.choice(len(self._object_colors), size=self.num_objects, replace=False)
        else:
            color_idx = range(self.num_objects)
        for i in range(self.num_objects):
            if self._object_type == 'cube':
                for gid in self._object_geom_ids_list[i]:
                    self._model.geom(gid).rgba = self._object_colors[color_idx[i]]
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._object_colors[color_idx[i], :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if "data_collection" in self._mode:
            tableware_reset = ('tableware' in self._mode) or self._object_type != 'cube'
            init_attempts = self._tableware_init_resample_attempts if tableware_reset else 1
            for init_attempt in range(init_attempts):
                try:
                    # Randomize the scene.
                    self.initialize_arm()

                    # Randomize object positions and orientations.
                    state_xyzs = []
                    for i in range(self.num_objects):
                        #print("过一次物品")
                        obj_pos = self.sample_obj_pos(np.array(state_xyzs), object_idx=i)#第一次随便摆，后面的避免与之前的(state_xyzs)碰撞
                        obj_pos = self._with_object_resting_z(i, obj_pos)
                        state_xyzs.append(obj_pos)
                        yaw = self.np_random.uniform(0, 2 * np.pi)
                        obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                        self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos #state_xyzs[i]
                        self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

                    # Set a new target.
                    self._stacking = False
                    self._stack_targets = []
                    self._stack_target_positions = []
                    self._stack_target_oris = []
                    self._tableware_ppp_target_positions = {}
                    self._tableware_ppp_target_oris = {}
                    self.set_new_target(return_info=False)
                    break
                except AssertionError as exc:
                    if (not tableware_reset) or ('Sampling' not in str(exc)) or (init_attempt + 1 >= init_attempts):
                        raise
                    self._tableware_ppp_target_positions = {}
                    self._tableware_ppp_target_oris = {}
                    continue
        else:
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            
            # First, force set the current scene to a randomized goal state to obtain the goal observation.
            self.initialize_arm()
            goal_xyzs = [None for _ in range(self.num_objects)]
            goal_oris = [lie.SO3.identity().wxyz.tolist() for _ in range(self.num_objects)]

            if self._is_hybrid_mode and 'task' in self._mode:
                # Hybrid goal: first Y blocks are stacked, remaining X blocks go to fixed PPP points.
                for idx, block_id in enumerate(self._stack_block_indices):
                    goal_xyzs[block_id] = self._hybrid_stack_target_positions[idx]
                for idx, block_id in enumerate(self._ppp_block_indices):
                    goal_xyzs[block_id] = self._hybrid_ppp_target_positions[idx]
            else:
                sampled_goal_xyzs = []
                sampled_goal_oris = []
                for i in range(self.num_objects):
                    obj_pos = self.sample_obj_pos(np.array(sampled_goal_xyzs), goal=True, object_idx=i)
                    obj_pos = self._with_object_resting_z(i, obj_pos)
                    sampled_goal_xyzs.append(obj_pos)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    sampled_goal_oris.append(lie.SO3.from_z_radians(yaw).wxyz.tolist())
                for i in range(self.num_objects):
                    goal_xyzs[i] = sampled_goal_xyzs[i]
                    goal_oris[i] = sampled_goal_oris[i]

            for i in range(self.num_objects):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = goal_oris[i]
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = goal_oris[i]
            mujoco.mj_forward(self._model, self._data)
            
            # Do a few random steps to make the scene stable.
            for _ in range(2):
                self.step(self.action_space.sample())

            # Save the goal observation.
            import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print(f"\033[95m 初始化episode，记录当前goal============\033[0m")
            self._cur_goal_ob = self.compute_observation()
            self._cur_goal_ob_info = self.compute_ob_info(description="self._cur_goal_ob_info")
            if self._render_goal:
                self._cur_goal_rendered = self.get_pixel_observation()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            # Randomize object positions and orientations.
            state_xyzs = []
            for i in range(self.num_objects):
                obj_pos = self.sample_obj_pos(np.array(state_xyzs), object_idx=i)
                obj_pos = self._with_object_resting_z(i, obj_pos)
                state_xyzs.append(obj_pos)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = goal_oris[i]

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False
        self._success_frac = 0
        import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print(" ")

    def sample_obj_pos(
        self,
        sampled_positions,
        goal=False,
        object_idx=None,
        candidate_radius=None,
        sampled_object_indices=None,
        clearance=0.015,
        reference_xy=None,
        min_reference_dist=0.0,
        max_reference_dist=None,
    ):
        num_positions_sampled = sampled_positions.shape[0]
        sampling_bounds = np.asarray(self._object_sampling_bounds, dtype=np.float32).copy()
        reference_xy = None if reference_xy is None else np.asarray(reference_xy, dtype=np.float32)[:2]
        min_reference_dist = max(0.0, float(min_reference_dist))
        max_reference_dist = (
            None
            if max_reference_dist is None
            else max(min_reference_dist, float(max_reference_dist))
        )
        has_reference_constraint = (
            reference_xy is not None
            and (min_reference_dist > 0.0 or max_reference_dist is not None)
        )
        tableware_sampling = self._object_type != 'cube' and hasattr(self, '_tableware_object_profiles')
        if tableware_sampling:
            if candidate_radius is None:
                if object_idx is None:
                    candidate_radius = self._tableware_max_horizontal_radius()
                else:
                    candidate_radius = self._tableware_horizontal_radius(object_idx)
            # Keep large tableware far enough from the y workspace boundary that the gripper
            # can still approach the object's outer edge for push/contact actions.
            side_margin = float(candidate_radius) + float(clearance) + 0.03
            sampling_bounds[0, 1] = max(sampling_bounds[0, 1], self._workspace_bounds[0, 1] + side_margin)
            sampling_bounds[1, 1] = min(sampling_bounds[1, 1], self._workspace_bounds[1, 1] - side_margin)
        
        if goal and self._mode == 'task_sort':
            if num_positions_sampled < self._num_colors:
                xy_pos = self._sort_goal_positions[num_positions_sampled]
                return (*xy_pos, 0.02)
            else:
                return (2, 2, 0.02)  # place the additional objects out of view
        
        else:
            if num_positions_sampled == 0 and not has_reference_constraint:
                # First position is guaranteed to be collision-free
                xy_pos = self.np_random.uniform(*sampling_bounds)
                return (*xy_pos, 0.02)

            if goal and self._mode == 'task_stack' and num_positions_sampled % 2 == 1:
                xy_pos = sampled_positions[-1, :2]  # stack on top of the last odd object
                return (*xy_pos, 0.06)

            elif goal and self._mode == 'task_stack_all' and num_positions_sampled > 0:
                xy_pos = sampled_positions[0, :2]  # stack on top of the last object
                return (*xy_pos, 0.02 + 0.04 * num_positions_sampled)

        # Define minimum distance between objects.
        if tableware_sampling:
            if sampled_object_indices is None:
                sampled_object_indices = range(num_positions_sampled)
            sampled_radii = np.array(
                [self._tableware_horizontal_radius(int(i)) for i in sampled_object_indices],
                dtype=np.float32,
            )
            min_dist = sampled_radii + float(candidate_radius) + float(clearance)
        else:
            cube_size = 0.04  # NOTE: maybe get from xml
            min_dist = 2 * cube_size * np.sqrt(2) / 2.0
                    
        # Sample collision-free positions
        valid = False
        for i in range(10000):
            # Sample xy positions
            xy_pos = self.np_random.uniform(*sampling_bounds)
            if has_reference_constraint:
                reference_dist = np.linalg.norm(xy_pos - reference_xy)
                if reference_dist < min_reference_dist:
                    continue
                if max_reference_dist is not None and reference_dist > max_reference_dist:
                    continue
            # Check if sampled values are valid
            xy_pos_copy = np.tile(xy_pos, (len(sampled_positions), 1))
            cube_dist = np.linalg.norm(xy_pos_copy - sampled_positions[..., :2], axis=-1)
            if np.all(cube_dist > min_dist, axis=-1):
                valid = True
                break
        
        # Make sure we succeeded at sampling
        assert valid, "Sampling cube locations was unsuccessful! ):"

        return (*xy_pos, 0.02)

    def _parse_cube_split_from_mode(self, mode):
        """Parse hybrid naming rule from mode, e.g., task_cube2+3 / data_collection_cube2+3."""
        match = re.search(r'cube(\d+)\+(\d+)', mode)
        if match is None:
            return None
        ppp_count = int(match.group(1))
        stack_count = int(match.group(2))
        return ppp_count, stack_count

    def _default_ppp_target_xys(self, ppp_count):
        """Return default fixed PPP target XY coordinates."""
        if ppp_count <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        anchors = np.array(
            [
                [0.32, -0.22],
                [0.53, 0.22],
                [0.32, 0.22],
                [0.53, -0.22],
                [0.425, 0.22],
                [0.425, -0.22],
            ],
            dtype=np.float32,
        )
        return anchors[:ppp_count].copy()

    def _build_hybrid_target_positions(self):
        """Build target positions for the hybrid 3-stack + 2-PPP task."""
        stack_target_positions = []
        # Convert 2D table coordinate (x, y) into 3D world coordinates by adding table height (z=0.02).
        for level in range(len(self._stack_block_indices)):
            stack_target_positions.append(
                np.array(
                    [
                        self._hybrid_stack_base_xy[0],
                        self._hybrid_stack_base_xy[1],
                        0.02 + self._hybrid_stack_height_step * level,
                    ],
                    dtype=np.float32,
                )
            )

        ppp_target_positions = []
        for xy in self._hybrid_ppp_target_xys:
            # PPP target points are fixed table-top coordinates with z fixed to table height.
            ppp_target_positions.append(np.array([xy[0], xy[1], 0.02], dtype=np.float32))

        return stack_target_positions, ppp_target_positions

    def _apply_hybrid_targets_to_mocap(self):
        """Apply all hybrid targets to mocap markers for visualization and policy conditioning."""
        quat_identity = lie.SO3.identity().wxyz.tolist()
        for i in range(self.num_objects):
            if i in self._stack_block_indices:
                idx = self._stack_block_indices.index(i)
                tar_pos = self._hybrid_stack_target_positions[idx]
            elif i in self._ppp_block_indices:
                idx = self._ppp_block_indices.index(i)
                tar_pos = self._hybrid_ppp_target_positions[idx]
            else:
                tar_pos = np.array([0.0, 0.0, -0.3], dtype=np.float32)
            self._data.mocap_pos[self._object_target_mocap_ids[i]] = tar_pos
            self._data.mocap_quat[self._object_target_mocap_ids[i]] = quat_identity

    def _compute_hybrid_successes(self):
        """Compute per-block success for the hybrid 3-stack + 2-PPP task."""
        successes = [False for _ in range(self.num_objects)]

        # Stack: require both xy alignment and height alignment.
        for idx, block_id in enumerate(self._stack_block_indices):
            obj_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
            tar_pos = self._hybrid_stack_target_positions[idx]
            xy_ok = np.linalg.norm(obj_pos[:2] - tar_pos[:2]) <= self._hybrid_stack_xy_threshold
            z_ok = np.abs(obj_pos[2] - tar_pos[2]) <= self._hybrid_stack_z_threshold
            successes[block_id] = bool(xy_ok and z_ok)

        # PPP: require full 3D position error below threshold.
        for idx, block_id in enumerate(self._ppp_block_indices):
            obj_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
            tar_pos = self._hybrid_ppp_target_positions[idx]
            successes[block_id] = bool(np.linalg.norm(obj_pos - tar_pos) <= self._hybrid_ppp_pos_threshold)

        return successes

    def _update_hybrid_stack_target(self, successes):
        """Keep stack subtasks corrected online, similar to stack-only mode behavior."""
        if self._target_task_group != 'stack':
            return

        unsatisfied_stack = [i for i in self._stack_block_indices if not successes[i]]
        if len(unsatisfied_stack) == 0:
            self._stacking = False
            return

        self._target_block = unsatisfied_stack[0]
        self._target_task = 'stack_cube'
        self._target_task_group = 'stack'
        self._stacking = True

    def _set_new_target_hybrid(self, return_info=True, successes=None):
        """Select next subtask target for the hybrid data-collection mode."""
        if successes is None:
            successes = self._compute_hybrid_successes()

        # Prioritize stack subtasks first, then PPP subtasks.
        unsatisfied_stack = [i for i in self._stack_block_indices if not successes[i]]
        unsatisfied_ppp = [i for i in self._ppp_block_indices if not successes[i]]

        if len(unsatisfied_stack) > 0:
            self._target_block = unsatisfied_stack[0]
            self._target_task = 'stack_cube'
            self._target_task_group = 'stack'
        elif len(unsatisfied_ppp) > 0:
            self._target_block = unsatisfied_ppp[0]
            self._target_task = 'pickplace_cube'
            self._target_task_group = 'pickplace'
        else:
            # All targets reached; keep emitting valid subtasks to continue collecting transitions.
            if len(self._ppp_block_indices) > 0:
                self._target_block = int(self.np_random.choice(self._ppp_block_indices))
                self._target_task = 'pickplace_cube'
                self._target_task_group = 'pickplace'
            elif len(self._stack_block_indices) > 0:
                self._target_block = int(self.np_random.choice(self._stack_block_indices))
                self._target_task = 'stack_cube'
                self._target_task_group = 'stack'
            else:
                raise RuntimeError(f"Mode '{self._mode}' has no hybrid targets to select from.")

        self._stacking = self._target_task_group == 'stack'
        self._apply_hybrid_targets_to_mocap()

        # Highlight only the active target block when visualization is enabled.
        for i in range(self.num_objects):
            alpha = 0.2 if (self._visualize_info and i == self._target_block) else 0.0
            for gid in self._object_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[3] = alpha

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks.
        """
        assert 'data_collection' in self._mode
        if self._is_hybrid_mode:
            return self._set_new_target_hybrid(return_info=return_info)

        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self.num_objects)])
        aux_target_positions = {}
        aux_target_oris = {}

        # Compute the top blocks.
        top_blocks = []
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i == j:
                    continue
                if block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)

        if self._stacking:
            # Find the first block in the stack that is not in place
            next_target_index = None
            stack_in_place_threshold = 0.04 if (('tableware' in self._mode) or (self._object_type != 'cube')) else 0.02
            for idx, block_id in enumerate(self._stack_targets):
                block_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
                tar_pos = self._stack_target_positions[idx]
                in_place = np.linalg.norm(block_pos - tar_pos) <= stack_in_place_threshold
                if not in_place:
                    next_target_index = idx
                    break
            assert next_target_index is not None
            self._stack_target_index = next_target_index
            self._target_block = self._stack_targets[self._stack_target_index]
            
        else:
            is_tableware_mode = ('tableware' in self._mode) or (self._object_type != 'cube')
            if self._mode == "data_collection_stack":
                if len(top_blocks) >= 2 and self.np_random.uniform() < p_stack:
                    # Multi-block stacking
                    self._stacking = True
                    stack_size = self.num_objects
                    # stack_size = self.np_random.integers(2, self.num_objects + 1)  # NOTE: uncomment to randomize stack size
                    self._stack_targets = self.np_random.choice(self.num_objects, size=stack_size, replace=False).tolist()
                    # Choose a random base position for the stack
                    base_pos = self.sample_obj_pos(
                        np.array(block_xyzs),
                        candidate_radius=self._tableware_max_horizontal_radius(self._stack_targets),
                    )
                    self._stack_target_positions = [np.array([base_pos[0], base_pos[1], 0.02 + 0.04 * i]) for i in range(stack_size)]
                    self._stack_target_index = 0
                    self._stack_step_counter = 0
                    self._target_task = 'pickplace_cube'
                    self._target_task_group = 'stack'
                    self._target_block = self._stack_targets[self._stack_target_index]
                    tar_pos = self._stack_target_positions[self._stack_target_index]
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._stack_target_oris = [lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist() for _ in range(stack_size)]
                    tar_ori = self._stack_target_oris[self._stack_target_index]
                else:
                    # Single block pickplace
                    self._stacking = False
                    self._target_block = self.np_random.choice(top_blocks)
                    self._target_task = 'pickplace_cube'
                    self._target_task_group = 'pickplace'
                    tar_pos = self.sample_obj_pos(np.array(block_xyzs), object_idx=self._target_block)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            elif is_tableware_mode:
                # Tableware naming rule: tablewareX+Y where X is pick-place count and Y is stack count.
                # Object ids are role ordered by _build_tableware_object_kinds().
                tableware_split = self._parse_tableware_split_from_mode(self._mode)
                pickplace_count, stack_count = (0, 0) if tableware_split is None else tableware_split
                candidate_top_blocks = [int(b) for b in top_blocks]
                if len(candidate_top_blocks) == 0:
                    candidate_top_blocks = [int(self.np_random.choice(np.arange(self.num_objects)))]
                ppp_block_indices = getattr(
                    self,
                    '_tableware_ppp_block_indices',
                    list(range(min(pickplace_count, self.num_objects))),
                )
                stack_block_indices = getattr(
                    self,
                    '_tableware_stack_block_indices',
                    list(
                        range(
                            min(pickplace_count, self.num_objects),
                            min(pickplace_count + stack_count, self.num_objects),
                        )
                    ),
                )
                ordered_top_blocks = [i for i in range(self.num_objects) if i in candidate_top_blocks]
                candidate_stack_blocks = [i for i in stack_block_indices if i in candidate_top_blocks]
                stack_targets_complete = self._tableware_stack_targets_complete(stack_block_indices)

                should_stack = False
                if stack_count > 0 and len(candidate_stack_blocks) >= 2:
                    if pickplace_count <= 0:
                        # 0+Y: pure stacking curriculum.
                        should_stack = True
                    else:
                        # X+Y: finish the stack first, then dispatch PPP objects such as the mug.
                        should_stack = not stack_targets_complete

                if should_stack:
                    self._stacking = True
                    stack_size = int(np.clip(stack_count, 2, len(candidate_stack_blocks)))
                    planned_stack_targets = candidate_stack_blocks[:stack_size]
                    stored_stack_targets = [int(i) for i in getattr(self, '_stack_targets', [])]
                    stored_stack_positions = getattr(self, '_stack_target_positions', [])
                    stored_stack_oris = getattr(self, '_stack_target_oris', [])
                    reuse_stack_plan = (
                        stored_stack_targets[:stack_size] == [int(i) for i in planned_stack_targets]
                        and len(stored_stack_positions) >= stack_size
                        and len(stored_stack_oris) >= stack_size
                    )
                    if reuse_stack_plan:
                        for idx in range(stack_size):
                            stored_pos = np.asarray(stored_stack_positions[idx], dtype=np.float32)
                            if (not np.all(np.isfinite(stored_pos))) or float(stored_pos[2]) < -0.1:
                                reuse_stack_plan = False
                                break

                    self._stack_targets = planned_stack_targets
                    # Move the plate a controlled local distance, then use it as the stack base.
                    base_block = int(self._stack_targets[0])
                    avoid_blocks = [i for i in range(self.num_objects) if i != base_block]
                    if reuse_stack_plan:
                        self._stack_target_positions = [
                            np.asarray(stored_stack_positions[idx], dtype=np.float32).copy()
                            for idx in range(stack_size)
                        ]
                        self._stack_target_oris = [
                            list(stored_stack_oris[idx])
                            for idx in range(stack_size)
                        ]
                        base_pos = self._stack_target_positions[0]
                    else:
                        base_pos = self._sample_tableware_stack_base_near_current(
                            base_block,
                            self._stack_targets,
                            block_xyzs,
                            avoid_blocks=avoid_blocks,
                        )
                        self._stack_target_positions = self._build_tableware_stack_target_positions(
                            self._stack_targets,
                            base_pos[:2],
                        )
                        self._stack_target_oris = [
                            lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
                            for _ in range(stack_size)
                        ]
                    self._stack_target_index = 0
                    self._stack_step_counter = 0
                    blocker_block = self._find_tableware_stack_base_blocker(
                        base_block,
                        base_pos[:2],
                        block_xyzs,
                        candidate_blocks=avoid_blocks,
                    )
                    if blocker_block is None:
                        self._target_task = 'pickplace_cube'
                        self._target_task_group = 'stack'
                        self._target_block = self._stack_targets[self._stack_target_index]
                        tar_pos = self._stack_target_positions[self._stack_target_index]
                        tar_ori = self._stack_target_oris[self._stack_target_index]
                    else:
                        self._stacking = False
                        self._target_task = 'pickplace_cube'
                        self._target_task_group = 'pickplace'
                        self._target_block = int(blocker_block)
                        tar_pos = self._sample_tableware_preclear_target(
                            self._target_block,
                            base_block,
                            base_pos[:2],
                            block_xyzs,
                        )
                        tar_ori = lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
                        for aux_block, aux_pos, aux_ori in zip(
                            self._stack_targets,
                            self._stack_target_positions,
                            self._stack_target_oris,
                        ):
                            aux_block = int(aux_block)
                            if aux_block == int(self._target_block):
                                continue
                            aux_target_positions[aux_block] = np.asarray(aux_pos, dtype=np.float32).copy()
                            aux_target_oris[aux_block] = list(aux_ori)
                    for ppp_block in ppp_block_indices:
                        ppp_block = int(ppp_block)
                        if ppp_block == int(self._target_block):
                            continue
                        ppp_tar_pos, ppp_tar_ori = self._get_or_sample_tableware_ppp_target(ppp_block, block_xyzs)
                        aux_target_positions[ppp_block] = ppp_tar_pos
                        aux_target_oris[ppp_block] = ppp_tar_ori
                else:
                    self._stacking = False
                    if pickplace_count > 0:
                        candidate_ppp_blocks = [b for b in ppp_block_indices if b in candidate_top_blocks]
                        if len(candidate_ppp_blocks) == 0:
                            candidate_ppp_blocks = ordered_top_blocks
                        self._target_block = int(candidate_ppp_blocks[0])
                    else:
                        self._target_block = int(ordered_top_blocks[0])
                    self._target_task = 'pickplace_cube'
                    self._target_task_group = 'pickplace'
                    if int(self._target_block) in [int(block_id) for block_id in ppp_block_indices]:
                        tar_pos, tar_ori = self._get_or_sample_tableware_ppp_target(self._target_block, block_xyzs)
                    else:
                        avoid_blocks = [i for i in range(self.num_objects) if i != self._target_block]
                        avoid_xyzs = block_xyzs[avoid_blocks] if len(avoid_blocks) > 0 else np.empty((0, 3), dtype=np.float32)
                        tar_pos = self.sample_obj_pos(
                            np.array(avoid_xyzs),
                            object_idx=self._target_block,
                            sampled_object_indices=avoid_blocks,
                        )
                        tar_pos = self._with_object_resting_z(self._target_block, tar_pos)
                        yaw = self.np_random.uniform(0, 2 * np.pi)
                        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    if stack_count > 0 and len(stack_block_indices) >= 2:
                        aux_stack_targets = list(stack_block_indices[:stack_count])
                        stored_stack_targets = [int(i) for i in getattr(self, '_stack_targets', [])]
                        has_completed_stack_targets = (
                            self._tableware_stack_targets_complete(aux_stack_targets)
                            and stored_stack_targets[: len(aux_stack_targets)] == [int(i) for i in aux_stack_targets]
                            and len(getattr(self, '_stack_target_positions', [])) >= len(aux_stack_targets)
                            and len(getattr(self, '_stack_target_oris', [])) >= len(aux_stack_targets)
                        )
                        if has_completed_stack_targets:
                            aux_stack_positions = [
                                np.asarray(self._stack_target_positions[idx], dtype=np.float32).copy()
                                for idx in range(len(aux_stack_targets))
                            ]
                            aux_stack_oris = [
                                self._stack_target_oris[idx]
                                for idx in range(len(aux_stack_targets))
                            ]
                        else:
                            # Visualize the stack goal at a controlled local offset from the current plate.
                            stack_base_block = int(aux_stack_targets[0])
                            avoid_blocks = [i for i in range(self.num_objects) if i != stack_base_block]
                            stack_base_pos = self._sample_tableware_stack_base_near_current(
                                stack_base_block,
                                aux_stack_targets,
                                block_xyzs,
                                avoid_blocks=avoid_blocks,
                            )
                            aux_stack_positions = self._build_tableware_stack_target_positions(
                                aux_stack_targets,
                                stack_base_pos[:2],
                            )
                            aux_stack_oris = [
                                lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
                                for _ in aux_stack_targets
                            ]
                        for aux_block, aux_pos, aux_ori in zip(aux_stack_targets, aux_stack_positions, aux_stack_oris):
                            aux_target_positions[aux_block] = aux_pos
                            aux_target_oris[aux_block] = aux_ori
            else:
                # Original logic for other data_collection modes
                self._target_block = self.np_random.choice(top_blocks)
                if self._target_block < self.num_cubes:
                    self._target_task = self.np_random.choice(['pickplace_cube', 'push_cube'], p=[0.3, 0.7])
                else:
                    self._target_task = self.np_random.choice(['pickplace_t', 'push_t'], p=[0.7, 0.3])
                self.stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
                self._target_task_group = 'stack' if self.stack else 'pickplace'
                if self.stack:
                    self.stack_block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
                    block_pos = self._data.joint(f'object_joint_{self.stack_block_idx}').qpos[:3]
                    tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
                else:
                    tar_pos = self.sample_obj_pos(np.array(block_xyzs))
                yaw = self.np_random.uniform(0, 2 * np.pi)
                tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        
        # Set mocap targets for all blocks
        for i in range(self.num_objects):
            if self._stacking and i in self._stack_targets:
                idx = self._stack_targets.index(i)
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = self._stack_target_positions[idx]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = self._stack_target_oris[idx]
            elif not self._stacking and i == self._target_block:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = tar_ori
            elif i in aux_target_positions:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = aux_target_positions[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = aux_target_oris[i]
            else:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Set the target colors.
        for i in range(self.num_objects):
            if self._visualize_info and ((self._stacking) or (not self._stacking and i == self._target_block)):
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def update_stack_target(self):
        # Only applies if stacking is active
        if not getattr(self, "_stacking", False):
            return

        # Find the first block in the stack that is not in place
        next_target_index = None
        stack_in_place_threshold = 0.04 if (('tableware' in self._mode) or (self._object_type != 'cube')) else 0.02
        for idx, block_id in enumerate(self._stack_targets):
            block_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
            tar_pos = self._stack_target_positions[idx]
            in_place = np.linalg.norm(block_pos - tar_pos) <= stack_in_place_threshold
            if not in_place:
                next_target_index = idx
                break
        
        # If a previous block is not in place, make it your target. Otherwise, let set_new_target() handle the next block logic
        if next_target_index is not None:
            next_target_index = min(self._stack_target_index, next_target_index)

        self._stack_step_counter += 1

        # If all blocks are in place, or timeout, reset stack targets.
        # Large tableware objects may need longer edge-push corrections than cubes.
        stack_timeout = 900 if (('tableware' in self._mode) or (self._object_type != 'cube')) else 500
        if next_target_index is None or self._stack_step_counter > stack_timeout:
            self._stacking = False
            return

        # Set the new target block
        self._stack_target_index = next_target_index
        self._target_block = self._stack_targets[self._stack_target_index]

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        # Compute target effector pose based on the relative action.
        effector_pos = self._data.site_xpos[self._pinch_site_id].copy()
        effector_yaw = lie.SO3.from_matrix(
            self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()
        gripper_opening = np.array(np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1))
        target_effector_translation = effector_pos + a_pos
        target_effector_orientation = (
            lie.SO3.from_z_radians(a_ori)
            @ lie.SO3.from_z_radians(effector_yaw)
            @ self._effector_down_rotation.inverse()
        )
        target_gripper_opening = gripper_opening + a_gripper

        # Make sure the target pose respects the action limits.
        np.clip(
            target_effector_translation,
            *self._workspace_bounds,
            out=target_effector_translation,
        )
        yaw = np.clip(
            target_effector_orientation.compute_yaw_radians(),
            -np.pi,
            +np.pi,
        )
        target_effector_orientation = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation
        target_gripper_opening = np.clip(target_gripper_opening, 0.0, 1.0)

        # Pinch pose in the world frame -> attach pose in the world frame.
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = self._target_effector_pose @ self._T_pa

        # Solve for the desired joint positions.
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        # Set the desired joint positions for the underlying PD controller.
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = 255.0 * target_gripper_opening

    def pre_step(self):
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info(description="self._prev_ob_info状态对比")
        #self._prev_ob_info 用于后续 step 时对比状态变化

    def post_step(self):
        if 'data_collection' in self._mode:
            if self._is_hybrid_mode:
                object_successes = self._compute_hybrid_successes()
                self._update_hybrid_stack_target(object_successes)
            else:
                object_successes = self._compute_successes()
                self.update_stack_target()
            self._success = object_successes[self._target_block]
        else:
            object_successes = self._compute_successes()
            self._success = all(object_successes)
            self._success_frac = np.mean(object_successes)

        # Adjust the colors of the target cubes.
        for i in range(self.num_objects):
            if self._visualize_info and ('task' in self._mode or self._stacking or i == self._target_block):
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

    def compute_ob_info(self,description=False):#info起始之地
        '''
        ob_info                                  # 当前时间帧的结构化状态信息（机器人 + 物体 + 系统状态）
        │
        │
        ├── proprio/joint_pos   (6,)             # 机械臂6个关节的位置（角度/位移），来自 qpos；每个关节一个值
        ├── proprio/joint_vel   (6,)             # 机械臂6个关节的速度，来自 qvel；每个关节一个速度值
        │
        ├── proprio/effector_pos (3,)            # 末端执行器在世界坐标系的位置 [x,y,z]，来自 site_xpos
        ├── proprio/effector_yaw (1,)            # 末端绕竖直轴的旋转角 yaw；桌面操作通常只关心平面朝向
        │
        ├── proprio/gripper_opening (1,)         # 夹爪开合程度（0≈闭合，1≈张开），由夹爪关节位置归一化得到
        ├── proprio/gripper_vel (1,)             # 夹爪开合速度，来自夹爪关节 qvel
        ├── proprio/gripper_contact (1,)         # 夹爪接触强度信号，由夹爪pad外力(cfrc_ext)计算，表示是否接触物体
        │
        ├── privileged/block_0_pos (3,)          # 第0个方块的位置 [x,y,z]
        ├── privileged/block_0_quat (4,)         # 第0个方块姿态四元数 [w,x,y,z]
        ├── privileged/block_0_yaw (1,)          # 第0个方块绕竖直轴的朝向角
        │
        ├── privileged/block_1_pos (3,)          # 第1个方块的位置
        ├── privileged/block_1_quat (4,)         # 第1个方块姿态四元数
        ├── privileged/block_1_yaw (1,)          # 第1个方块朝向角
        │
        ├── privileged/block_2_pos (3,)          # 第2个方块的位置
        ├── privileged/block_2_quat (4,)         # 第2个方块姿态四元数
        ├── privileged/block_2_yaw (1,)          # 第2个方块朝向角
        │                                                # “privileged”通常表示训练时可用的环境真值信息
        │
        ├── prev_qpos (35,)                      # 上一帧系统完整位置状态（robot + objects）；长度=模型 nq
        ├── prev_qvel (32,)                      # 上一帧系统完整速度状态；长度=模型 nv
        │
        ├── qpos (35,)                           # 当前帧完整位置状态（机器人关节 + 物体位置 + 物体姿态等）
        ├── qvel (32,)                           # 当前帧完整速度状态（机器人关节速度 + 物体线/角速度）
        │
        ├── control (7,)                         # 当前控制输入；7个执行器（6个关节 + 1个夹爪）
        │
        └── time (1,)                            # 当前仿真时间（秒）
        '''
        ob_info = {}
        import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print(f"\033[92m info起始之地==============={description}\033[0m")
        # Proprioceptive observations
        ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
        ob_info['proprio/effector_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
        )
        ob_info['proprio/gripper_opening'] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )
        ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.array(
            [np.clip(np.linalg.norm(self._data.body('ur5e/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
        )

        self.add_object_info(ob_info)

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = self._data.qpos.copy()
        ob_info['qvel'] = self._data.qvel.copy()
        ob_info['control'] = self._data.ctrl.copy()
        ob_info['time'] = np.array([self._data.time])
        #import pdb; pdb.set_trace()
        return ob_info

    def add_object_info(self, ob_info):
        # Object positions and orientations.
        for i in range(self.num_objects):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array([lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()])

        if 'data_collection' in self._mode:
            # Target cube info.
            ob_info['privileged/target_task'] = self._target_task
            ob_info['privileged/target_task_group'] = self._target_task_group

            target_mocap_id = self._object_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()])

    def get_pixel_observation(self):
        if self._mode in ['task_stack', 'task_stack_all', 'data_collection_stack'] or self._is_hybrid_mode:
            camera = 'front_pixels' if not self._multiview else ['front_pixels', 'side_pixels_stack']
        else:
            camera = 'front_pixels' if not self._multiview else ['front_pixels', 'side_pixels']
        frame = self.render(camera=camera)
        return frame

    def compute_observation(self): #由ogbench的reset调用 ogbench/ogbench/manipspace/envs/env.py
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            ob_info = self.compute_ob_info(description="应该是当前obs，用于制作21维度state")#数据基本取自self._data(MJData)
            #{'proprio/joint_pos': (6,), 'proprio/joint_vel': (6,), 'proprio/effector_pos': (3,), 'proprio/effector_yaw': (1,), 'proprio/gripper_opening': (1,), 'proprio/gripper_vel': (1,), 'proprio/gripper_contact': (1,), 
            #'privileged/block_0_pos': (3,), 'privileged/block_0_quat': (4,), 'privileged/block_0_yaw': (1,), 'privileged/block_1_pos': (3,), 'privileged/block_1_quat': (4,), 'privileged/block_1_yaw': (1,), 'privileged/block_2_pos': (3,), 'privileged/block_2_quat': (4,), 'privileged/block_2_yaw': (1,), 
            #'prev_qpos': (35,), 'prev_qvel': (32,), 'qpos': (35,), 'qvel': (32,), 'control': (7,), 'time': (1,)}
            if self._ob_type == 'states_clean':#baseline走这个
                return self.get_state_clean_observation(ob_info)#只是提取部分，摊开，去中心ob_info. 结果为21维
            else: # self._ob_type == 'states'
                return self.get_state_observation(ob_info)

    def get_state_clean_observation(self, ob_info):
        '''
        state_obs (21,)
        │
        ├── effector position
        │   ├── (3,)  (effector_pos - xyz_center(np.array([0.425, 0.0, 0.0]))) 数值* 10
        │
        ├── effector orientation
        │   ├── cos(yaw) (1,)
        │   └── sin(yaw) (1,)
        │
        ├── gripper
        │   └── gripper_opening 数值* 3 (1,)
        │
        ├── block_0
        │   ├── pos (3,)
        │   ├── cos(yaw) (1,)
        │   └── sin(yaw) (1,)
        │
        ├── block_1
        │   ├── pos (3,)
        │   ├── cos(yaw) (1,)
        │   └── sin(yaw) (1,)
        │
        └── block_2
            ├── pos (3,)
            ├── cos(yaw) (1,)
            └── sin(yaw) (1,)
        '''
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob = [
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
            ]
        for i in range(self.num_objects):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                ]
            )

        return np.concatenate(ob)
    
    def get_state_observation(self, ob_info):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
        for i in range(self.num_objects):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/block_{i}_quat'],
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                ]
            )
        return np.concatenate(ob)
    
    def _compute_successes(self):
        """Compute object successes."""
        if self._is_hybrid_mode:
            return self._compute_hybrid_successes()

        successes = []

        if self._mode == 'task_sort':
            for i in range(self.num_cubes):
                obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i % self._num_colors]]
                successes.append(np.linalg.norm(obj_pos - tar_pos) <= 4 * 0.04)

        else:
            for i in range(self.num_cubes):
                obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i]]
                
                # Cube: only position matters
                pos_success = np.linalg.norm(obj_pos - tar_pos) <= 0.04
                successes.append(pos_success)

        return successes
    
    def compute_reward(self):
        import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print(f"\033[94m compute_reward，其中self._reward_mode是{self._reward_mode}===============\033[0m")
        if self._reward_mode == 'sparse':#初始过这个,之后训练也是
            # Success if all cubes are in place
            successes = self._compute_successes()
            reward = 0.0 if all(successes) else -1.0

        elif self._reward_mode == 'step':
            # Penalize each object not in place
            successes = self._compute_successes()
            reward = -float(np.mean([not s for s in successes]))

        elif self._reward_mode == 'dense':
            # Penalize mean position error
            obj_pos = np.zeros((self.num_cubes, 3), dtype=np.float32)
            tar_pos = np.zeros((self.num_cubes, 3), dtype=np.float32)
            for i in range(self.num_cubes):
                obj_pos[i] = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos[i] = self._data.mocap_pos[self._object_target_mocap_ids[i]]
            dists = np.linalg.norm(obj_pos - tar_pos, axis=-1)
            reward = -np.mean(dists) * self._reward_scale
        
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")
        return reward

    def compute_reward_batch(self, qpos, goal_qpos):
        """Compute the reward for a batch of observations and goals from buffer."""
        import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print(f"\033[94m compute_reward_batch，其中self._reward_mode是{self._reward_mode}===============\033[0m")
        qpos_obj_start_idx = 14
        qpos_obj_length = 7  # 3 pos + 4 quat

        obj_xyzs_list = []
        goal_xyzs_list = []
        obj_quats_list = []
        goal_quats_list = []
        
        for i in range(self.num_cubes):
            obj_xyzs_list.append(qpos[:, qpos_obj_start_idx + i * qpos_obj_length : qpos_obj_start_idx + i * qpos_obj_length + 3])
            goal_xyzs_list.append(goal_qpos[:, qpos_obj_start_idx + i * qpos_obj_length : qpos_obj_start_idx + i * qpos_obj_length + 3])
            obj_quats_list.append(qpos[:, qpos_obj_start_idx + i * qpos_obj_length + 3 : qpos_obj_start_idx + (i+1) * qpos_obj_length])
            goal_quats_list.append(goal_qpos[:, qpos_obj_start_idx + i * qpos_obj_length + 3 : qpos_obj_start_idx + (i+1) * qpos_obj_length])
        obj_xyzs = torch.stack(obj_xyzs_list, dim=1)
        goal_xyzs = torch.stack(goal_xyzs_list, dim=1)
        obj_quats = torch.stack(obj_quats_list, dim=1)
        goal_quats = torch.stack(goal_quats_list, dim=1)

        dists = torch.linalg.norm(goal_xyzs - obj_xyzs, dim=-1)  # (batch, horizon, num_objects)

        # Success: all cubes are in place
        all_success = dists <= 0.04
        
        if self._reward_mode == 'sparse':
            reward = torch.where(torch.all(all_success, dim=-1, keepdim=True), 0.0, -1.0)
        elif self._reward_mode == 'step':
            reward = -1.0 * torch.mean((~all_success).float(), dim=-1, keepdim=True)
        elif self._reward_mode == 'dense':
            reward = -torch.mean(dists, dim=-1, keepdim=True) * self.cfg.env_kwargs["reward_scale"]
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")

        return reward
    
    def get_reset_info(self):
        '''
        接着compute_ob_info
        │
        ├── state_obs (21,)                        # 当前状态的“clean observation”（policy输入）
        │
        ├── goal (21,)                             # 任务目标状态向量
        │
        ├── state_goal (21,)                       # 目标状态的clean observation
        │
        └── goal_rendered (None)                   # 目标图像（当前未启用）
        '''
        reset_info = self.compute_ob_info(description="当前info，加上了goal的那几个key")
        reset_info['state_obs'] = self.get_state_clean_observation(reset_info)
        if 'task' in self._mode:
            reset_info['goal'] = self._cur_goal_ob
            reset_info['state_goal'] = self.get_state_clean_observation(self._cur_goal_ob_info)
            if self._render_goal is not None:
                reset_info['goal_rendered'] = self._cur_goal_rendered
        if 'data_collection' in self._mode:
            reset_info['prev_state_clean_observation'] = self.get_state_clean_observation(self._prev_ob_info)
        return reset_info

    def get_step_info(self):#由ogbench的reset调用
        ob_info = self.compute_ob_info(description="get_step_info可能用于之后评估")
        ob_info['state_obs'] = self.get_state_clean_observation(ob_info)
        ob_info['success'] = self._success
        ob_info['success_frac'] = self._success_frac
        ob_info['was_agent_object_interaction'] = self._was_agent_object_interaction(ob_info)
        if 'data_collection' in self._mode:
            ob_info['prev_state_clean_observation'] = self.get_state_clean_observation(self._prev_ob_info)
        return ob_info

    def terminate_episode(self):
        if self._terminate_at_goal:
            return self._success
        else:
            return False

    def render(
        self,
        camera='front_pixels',
        *args,
        **kwargs,
    ):
        if isinstance(camera, (list, tuple)):
            imgs = []
            for cam in camera:
                img = super().render(camera=cam, *args, **kwargs)
                imgs.append(img)
            return np.stack(imgs, axis=0)
        else:
            return super().render(camera=camera, *args, **kwargs)

    def _was_agent_object_interaction(self, ob_info):
        qpos_obj_start_idx = 14
        qpos_object_length = 7

        prev_qpos = ob_info['prev_qpos']
        qpos = ob_info['qpos']

        prev_object_xyzs_list = []
        object_xyzs_list = []
        for i in range(self.num_objects):
            prev_object_xyzs_list.append(prev_qpos[..., qpos_obj_start_idx + i * qpos_object_length : qpos_obj_start_idx + i * qpos_object_length + 3])
            object_xyzs_list.append(qpos[..., qpos_obj_start_idx + i * qpos_object_length : qpos_obj_start_idx + i * qpos_object_length + 3])
        prev_object_xyzs = np.stack(prev_object_xyzs_list, axis=0)
        object_xyzs = np.stack(object_xyzs_list, axis=0)

        dists = np.linalg.norm(prev_object_xyzs - object_xyzs, axis=-1)
        return np.any(dists >= 1e-4, axis=-1)
    
