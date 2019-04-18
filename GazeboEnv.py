import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import actionlib
import math
import rospy
import moveit_commander
import cv2
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import *
from moveit_python import MoveGroupInterface
from tf.transformations import quaternion_from_euler
from moveit_msgs.msg import MoveItErrorCodes
from control_msgs.msg import PointHeadAction, PointHeadGoal
from geometry_msgs.msg import PoseStamped, Quaternion, Point, Pose


class RGB(object):

    def __init__(self):

        topic_name_k = 'camera/image_raw'

        self._bridge = CvBridge()
        self._input_kinect_image = None
        self._sub_kincet_image = rospy.Subscriber(topic_name_k, Image, self._kinect_image_cb)

    def _kinect_image_cb(self, data):
        try:
            self._input_kinect_image = self._bridge.imgmsg_to_cv2(data, "bgr16")  # todo
            self.color_time_stamped = data.header.stamp
            self.is_updated = True
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_kinect_data(self):
        rgb = np.array(self._input_kinect_image)
        return cv2.resize(rgb, (224, 224))

    def read_kinect_data_raw(self):
        return self._input_kinect_image


class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class Robot(object):
    """
    fetch max x = 0.96 (wrist roll joint)
    """

    def __init__(self):
        rospy.loginfo("Waiting for Fetch ...")
        self.tuck_the_arm_joints_value = [0, 1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0]
        self.stow_joints_value = [0.0, 1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        self.pose_group = moveit_commander.MoveGroupCommander("arm_with_torso")
        self.joint_names = ["torso_lift_joint", "shoulder_pan_joint",
                            "shoulder_lift_joint", "upperarm_roll_joint",
                            "elbow_flex_joint", "forearm_roll_joint",
                            "wrist_flex_joint", "wrist_roll_joint"]

    def move_to_pose(self, postion, quaternion, frame='base_link'):
        target_frame = 'wrist_roll_link'
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = Pose(
            Point(postion[0], postion[1], postion[2]),
            Quaternion(quaternion[0], quaternion[1],
                       quaternion[2], quaternion[3])
        )
        if not rospy.is_shutdown():
            self.move_group.moveToPose(pose_stamped, target_frame)
            result = self.move_group.get_move_action().get_result()
            if result:
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    rospy.loginfo("Move success!")
                    return True
                else:
                    rospy.logerr("in state: %s",
                                 self.move_group.get_move_action().get_state())
                    return False

            else:
                rospy.logerr("MoveIt! failure no result returned.")
                return False

    def move_to_joints(self, joints_value):
        # Plans the joints in joint_names to angles in pose
        self.move_group.moveToJointPosition(self.joint_names, joints_value, wait=False)
        # Since we passed in wait=False above we need to wait here
        self.move_group.get_move_action().wait_for_result()
        result = self.move_group.get_move_action().get_result()
        if result:
            # Checking the MoveItErrorCode
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                rospy.loginfo("move to joint Success!")
            else:
                # If you get to this point please search for:
                # moveit_msgs/MoveItErrorCodes.msg
                rospy.logerr("Arm goal in state: %s",
                             self.move_group.get_move_action().get_state())
        else:
            rospy.logerr("MoveIt! failure no result returned.")

    def tuck_the_arm(self):
        self.move_to_joints(self.tuck_the_arm_joints_value)

    def stow(self):
        self.move_to_joints(self.stow_joints_value)

    def stop(self):
        """This stops all arm movement goals
        It should be called when a program is exiting so movement stops
        """
        self.move_group.get_move_action().cancel_all_goals()


class GazeboEnv(object):

    def __init__(self):
        """initialize move_group API"""
        rospy.init_node('gazebo_world')
        """init parm"""
        self.Robot_model = {'fetch': {'init': [0, 0, 0]}}
        self.Obstacle_model = {'obstacle': {'init': [0.7, 0, 0.75]}}
        self.Move_Obstacle_model = {'obstacle': {'init': [1.7, 0, 0.75]}}
        self.Fetch_init_pose_joints = [0.0, 1.5, 1.1, 0.0, -2.2, -1.8, 2, 0.4]

        self.camera = RGB()
        self.fetch = Robot()
        self.robot_pose = ModelState()
        self.pose_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.pose_get = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

        self.dis = 100

    def reset_robot(self):
        # rospy.loginfo("Reset robot")
        for k, v in self.Robot_model.items():
            self.set_model_pose(k, v['init'])

    def get_state(self):
        end_position = self.get_wrist_end_pose()
        img = self.camera.read_kinect_data()
        return end_position, img

    def step(self, action):
        done = False
        reward = 0
        end_position_old = self.get_wrist_end_pose()
        target_position_old = self.get_target_pose()

        result_fetch = self.fetch.move_to_pose((end_position_old[0]+action[0], end_position_old[1]+action[1], end_position_old[2]+action[2]), (0, 0, 0, 0))
        print "result_fetch = {}".format(result_fetch)

        target_position_now = self.get_target_pose()
        end_position_now = self.get_wrist_end_pose()

        """err : if collision"""
        err = math.sqrt(math.pow(target_position_old[0] - target_position_now[0], 2)
                        + math.pow(target_position_old[1] - target_position_now[1], 2)
                        + math.pow(target_position_old[2] - target_position_now[2], 2))
        print "err = {}".format(err)

        """result : if moving succeed"""
        if result_fetch is False or err > 0.05:
            done = True
            reward -= 40

        """after move, dis"""
        dis = math.sqrt(math.pow(target_position_old[0] - end_position_now[0], 2)
                        + math.pow(target_position_old[1] - end_position_now[1], 2)
                        + math.pow(target_position_old[2] - end_position_now[2], 2))

        if dis < 0.05:
            done = True
            reward += 100
        elif dis <= self.dis:
            reward += 30
        elif dis > self.dis:
            reward -= 30

        self.dis = dis

        return end_position_now, reward, done

    def init_fetch_pose(self):
        # rospy.loginfo("Init fetch robot pose")
        self.fetch.move_to_joints(self.Fetch_init_pose_joints)
        time.sleep(1)

    def reset_model(self):
        # rospy.loginfo("Reset model")
        for k, v in self.Obstacle_model.items():
            self.set_model_pose(k, v['init'])

    def move_away_model(self):
        # rospy.loginfo("move away model")
        for k, v in self.Move_Obstacle_model.items():
            self.set_model_pose(k, v['init'])

    def reset_world(self):
        # rospy.loginfo("Reset Env")
        self.move_away_model()
        self.init_fetch_pose()
        self.reset_robot()
        self.reset_model()

    def set_model_pose(self, name, pose, orient=None):
        """
        :param name: cube name, a string
        :param pose: cube position, a list of three float, [x, y, z]
        :param orient: cube orientation, a list of three float, [ix, iy, iz]
        :return:
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        self.robot_pose.model_name = name
        p = self.robot_pose.pose
        # cube_init = self.CubeMap[name]["init"]
        p.position.x = pose[0]
        p.position.y = pose[1]
        p.position.z = pose[2]
        if orient is None:
            orient = [0, 0, 0]
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        p.orientation = Quaternion(*q)
        self.pose_pub.publish(self.robot_pose)

    def get_wrist_end_pose(self):
        # rospy.loginfo("Get wrist position")
        s = self.pose_get("wrist_roll_link", "base_link")
        p = s.link_state.pose.position
        return [p.x, p.y, p.z]

    def get_target_pose(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        # rospy.loginfo("Get target position")
        s = self.pose_get("link_8", "base_link")
        p = s.link_state.pose.position
        return [p.x, p.y, p.z]


if __name__ == '__main__':
    g = GazeboEnv()

    # g.init_fetch_pose()
    # g.reset_robot()
    # g.reset_model()
    # a = [0.1, 0.1, 0.1]
    # print g.step(a)
    # # g.fetch.stow()
    # p = g.get_target_pose()
    # print p
    # pp = g.get_wrist_end_pose()
    # print pp
    # get_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    #
    # p = get_state("link_8", "base_link")
    # print p
    #
    # pp = get_state("wrist_roll_link", "base_link")
    # print pp
    #
    # ppp = g.fetch.pose_group.get_current_pose("wrist_roll_link").pose
    # print ppp
