import os

# Panda hostname/IP and Desk login information of your robot.
# Provide credentials via env vars: FRANKA_HOST, FRANKA_USER, FRANKA_PASS.
hostname = os.environ['FRANKA_HOST']
username = os.environ['FRANKA_USER']
password = os.environ['FRANKA_PASS']

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)


import panda_py

desk = panda_py.Desk(hostname, username, password)
#desk.unlock()
#desk.activate_fci()


panda = panda_py.Panda(hostname)
panda.move_to_start()
pose = panda.get_pose()
pose[2,3] -= .1
q = panda_py.ik(pose)
panda.move_to_joint_position(q)
