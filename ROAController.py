from utilities import AbstractController, LQRController, TVLQRController, PendulumPlant_
import numpy as np

class RoAController(AbstractController):

    def __init__(self,data_dict, mass=1.0, length=0.5, damping=0.1,
    gravity=9.81, torque_limit=np.inf, x_i = [0.0, 0.1]):

        ## LQR controller initialization for reaching the initial state
        self.x_i = x_i

        self.lqr = LQRController(mass=mass,
        length=length,
        damping=damping,
        gravity=gravity)
        self.lqr.set_goal(x_i)

        # boolean to take care of the controller switch
        self.xi_reached = False

        # seconds to wait after reaching the initial condition
        self.wait_time = 1
        self.change_time = 0

        ## TVLQR controller initialization for swing-up
        pendulum = PendulumPlant_(mass=mass,
                         length=length,
                         damping=damping,
                         gravity=gravity,
                         coulomb_fric=0,
                         inertia=None,
                         torque_limit=torque_limit)

        self.tvlqr = TVLQRController(data_dict=data_dict, mass=mass, length=length,
        damping=damping, gravity=gravity,
        torque_limit=torque_limit)

        # Taking the finals values of S and rho from the invariant case, SOS method has been chosen
        Sf = LQRController(mass=mass,
                            length=length,
                            damping=damping,
                            gravity=gravity,
                            torque_limit=torque_limit).S
        self.tvlqr.set_Qf(Sf)
        self.tvlqr.set_goal([np.pi, 0.0])

    def get_control_output(self, meas_pos, meas_vel,
        meas_tau=0, meas_time=0):
        des_pos, des_vel, u = self.lqr.get_control_output(meas_pos, meas_vel, meas_tau, meas_time)
        if (not self.xi_reached):
            print(f"lqr action at time {meas_time}")

        cond1 = (np.round(meas_pos,1) == np.round(self.x_i[0],1)) and (np.round(meas_vel,1) == np.round(self.x_i[1],1)) 
        if( cond1 or self.xi_reached):
            if self.wait_time > 0:
                print(f"tvlqr action at time {meas_time}")
                des_pos, des_vel, u = self.tvlqr.get_control_output(meas_pos, meas_vel, meas_tau, meas_time)
                
                if not self.xi_reached:
                    self.xi_reached = True
                    self.change_time = meas_time
                else:
                    self.wait_time = self.wait_time - (meas_time-self.change_time)                 
            

        return des_pos, des_vel, u