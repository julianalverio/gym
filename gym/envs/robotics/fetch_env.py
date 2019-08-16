from gym.envs.robotics import robot_env


class FetchEnv(robot_env.RobotEnv):
    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(FetchEnv, self).__init__(gripper_extra_height=gripper_extra_height, block_gripper=block_gripper,
            has_object=has_object, target_in_the_air=target_in_the_air, target_offset=target_offset,
            obj_range=obj_range, target_range=target_range, distance_threshold=distance_threshold,
            reward_type=reward_type, model_path=model_path, n_substeps=n_substeps, n_actions=4, initial_qpos=initial_qpos)
