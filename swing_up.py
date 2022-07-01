from ROAController import RoAController
import process_data, motor_control_loop
from utilities import get_params
from pathlib import Path
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from plots import plotFunnel3d_fromCsv
from utilities import prepare_trajectory

# set motor parameters
motor_id = 0x08
can_port = 'can0'

WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[0])
print("Workspace is set to:", WORK_DIR)

"""
    RoA verification
"""
name = "RoA verification"
folder_name = "meaningless" #"0initNoisySucc" 
attribute = "motorfft"


# get parameters
params_file = "sp_parameters_roa.yaml"
params_path = os.path.join(WORK_DIR, params_file)
params = get_params(params_path)
data_dict = process_data.prepare_empty(params)

# initial condition and trajectory from funnelComputation
x0 = [0.,  0.  ]
csv_path = "log_data/direct_collocation/trajectory.csv"
traj_dict = process_data.prepare_trajectory(csv_path)

## Going to the initial position and then activating the tvlqr
# TODO: now x_i has velocity zero but it should be random
control_method = RoAController(traj_dict, params, x_i= x0, disturbance=True)
                    
data_dict = process_data.prepare_empty(params)

# start control loop for ak80_6
start, end, meas_dt, data_dict = motor_control_loop.ak80_6(control_method,
                                                           name, attribute,
                                                           params, data_dict,
                                                           motor_id, can_port)

# save measurements
TIMESTAMP = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
output_folder = str(WORK_DIR) + '/results/'+ folder_name + f'{TIMESTAMP}_'
process_data.save(output_folder, data_dict)

# plot data
print("Making data plots.")
des_time = data_dict["des_time_list"]
des_pos = data_dict["des_pos_list"]
des_vel = data_dict["des_vel_list"]
des_tau = data_dict["des_tau_list"]
meas_time = data_dict["meas_time_list"]
meas_pos = data_dict["meas_pos_list"]
meas_vel = data_dict["meas_vel_list"]
meas_tau = data_dict["meas_tau_list"]

plt.figure()
plt.plot(meas_time, meas_pos)
plt.plot(des_time, des_pos)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title("Position (rad) vs Time (s)")
plt.legend(['position_measured', 'position_desired'])
plt.draw()
plt.savefig(output_folder + '/swingup_pos.pdf')

plt.figure()
plt.plot(meas_time, meas_vel)
plt.plot(des_time, des_vel)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['velocity_measured', 'velocity_desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.draw()
plt.savefig(output_folder + '/swingup_vel.pdf')

plt.figure()
plt.plot(meas_time, meas_tau)
plt.plot(des_time, des_tau)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Measured Torque', 'Desired Torque'])
plt.draw()
plt.savefig(output_folder + '/swingup_tau.pdf')
plt.show()

# load results of simulation and compare with real behaviour
csv_path = "log_data/direct_collocation/trajectory.csv"
funnels_csv_path = "log_data/funnel/funnel.csv"
data_dict = prepare_trajectory(csv_path)
trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
time = trajectory.T[0].T
x0_traj = [trajectory.T[1].T, trajectory.T[2].T]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(time, x0_traj[0], x0_traj[1])
indeces = np.where(meas_time >= control_method.tvlqrTime)[0]
meas_time = meas_time[indeces] - control_method.tvlqrTime*np.ones(len(meas_time[indeces]))
meas_pos = meas_pos[indeces]
meas_vel = meas_vel[indeces]
ax.plot(meas_time, meas_pos, meas_vel) 
plotFunnel3d_fromCsv(funnels_csv_path,x0_traj,time, ax)
plt.savefig(output_folder + '/swingup_funnel.pdf')
plt.show()