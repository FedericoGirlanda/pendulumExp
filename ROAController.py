from utilities import AbstractController, LQRController, TVLQRController, PendulumPlant_
import numpy as np

class RoAController(AbstractController):

    def __init__(self,data_dict, mass=1.0, length=0.5, damping=0.1,
    gravity=9.81, torque_limit=np.inf, x_i = [0.0, 0.1]):

        ## LQR controller initialization for reaching the initial state
        self.lqr = LQRController(mass=mass,
        length=length,
        damping=damping,
        gravity=gravity)
        self.lqr.set_goal(x_i) 

        # seconds to wait after reaching the initial condition
        self.wait_time = 1
        self.change_time = 0

        ## TVLQR controller initialization for swing-up
        self.tvlqr = TVLQRController(data_dict=data_dict, mass=mass, length=length,
        damping=damping, gravity=gravity,
        torque_limit=torque_limit)
        self.tvlqr.set_goal([np.pi, 0.0])

        self.active_controller = "lqr"

    def get_control_output(self, meas_pos, meas_vel,
        meas_tau=0, meas_time=0):               

        if self.active_controller == "lqr":
            des_pos, des_vel, u = self.lqr.get_control_output(meas_pos, meas_vel, meas_tau, meas_time)
            print(f"lqr action at time {meas_time} with input u = {meas_tau} and state x = [{meas_pos, meas_vel}]")

            cond1 = (np.round(meas_pos,1) == np.round(self.x_i[0],1)) and (np.round(meas_vel,1) == np.round(self.x_i[1],1)) 
            if cond1:
                self.active_controller = "wait"
                self.change_time = meas_time
        if self.active_controller == "wait":
            print("waiting...")
            des_pos, des_vel, u = self.lqr.get_control_output(meas_pos, meas_vel, meas_tau, meas_time)
            self.wait_time = self.wait_time - (meas_time-self.change_time)
            if self.wait_time < 0:
                self.active_controller = "tvlqr"
        if self.active_controller == "tvlqr":
            des_pos, des_vel, u = self.tvlqr.get_control_output(meas_pos, meas_vel, meas_tau, meas_time)
            print(f"tvlqr action at time {meas_time} with input u = {meas_tau} and state x = [{meas_pos, meas_vel}]")
        else:
            des_pos, des_vel, u = None,None,0

        return des_pos, des_vel, u