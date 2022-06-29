from ROAController import RoAController
import process_data, motor_control_loop
from utilities import get_params
from pathlib import Path
from datetime import datetime
import os
import matplotlib.pyplot as plt

# set motor parameters
motor_id = 0x08
can_port = 'can0'

WORK_DIR = ""
print("Workspace is set to:", WORK_DIR)


"""
    RoA verification
"""
name = "RoA verification"
folder_name = "roaVerification"
attribute = "motorfft"


# get parameters
params_file = "sp_parameters_roa.yaml"
params_path = os.path.join(WORK_DIR, params_file)
params = get_params(params_path)
data_dict = process_data.prepare_empty(params)

# initial condition and trajectory from funnelComputation
x0 = [0.5,  0.  ]
csv_path = "log_data/direct_collocation/trajectory.csv"
traj_dict = process_data.prepare_trajectory(csv_path)

## Going to the initial position and then activating the tvlqr
# TODO: now x_i has velocity zero but it should be random
control_method = RoAController(traj_dict, mass=params['mass'], length=params['length'],
                            damping=params['damping'], gravity=params['gravity'],
                            torque_limit=params['torque_limit_control'], x_i= x0)
#control_method.set_goal([np.pi, 0.0])  # final point must be stable point
                    
data_dict = process_data.prepare_empty(params)

# start control loop for ak80_6
start, end, meas_dt, data_dict = motor_control_loop.ak80_6(control_method,
                                                           name, attribute,
                                                           params, data_dict,
                                                           motor_id, can_port)

# save measurements
# TIMESTAMP = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
# output_folder = str(WORK_DIR) + f'/results/{TIMESTAMP}_' + folder_name
# process_data.save(output_folder, data_dict)

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
# plt.savefig(output_folder + '/swingup_pos.pdf')
plt.show()

plt.figure()
plt.plot(meas_time, meas_vel)
# plt.plot(meas_time, meas_vel_filt)
plt.plot(des_time, des_vel)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend(['velocity_measured', 'velocity_desired'])
plt.title("Velocity (rad/s) vs Time (s)")
plt.draw()
# plt.savefig(output_folder + '/swingup_vel.pdf')
plt.show()

plt.figure()
plt.plot(meas_time, meas_tau)
plt.plot(des_time, des_tau)
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque (Nm) vs Time (s)")
plt.legend(['Measured Torque', 'Desired Torque'])
plt.draw()
# plt.savefig(output_folder + '/swingup_tau.pdf')
plt.show()