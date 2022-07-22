"""
Simulator
=========
"""


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad
from pydrake.all import Variable, TaylorExpand, sin


class Simulator:
    def __init__(self, plant):
        """
        Simulator class, can simulate and animate the pendulum

        Parameters
        ----------
        plant: plant object
            (e.g. PendulumPlant from simple_pendulum.models.pendulum_plant.py)
        """

        self.plant = plant

        self.x = np.zeros(2*self.plant.dof)  # position, velocity
        self.t = 0.0  # time

        self.reset_data_recorder()

    def set_state(self, time, x):
        """
        set the state of the pendulum plant

        Parameters
        ----------
        time: float
            time, unit: s
        x: type as self.plant expects a state,
            state of the pendulum plant
        """

        self.x = np.copy(x)
        self.t = np.copy(float(time))

    def get_state(self):
        """
        Get current state of the plant

        Returns
        -------
        self.t : float,
            time, unit: s
        self.x : type as self.plant expects a state
            plant state
        """

        return self.t, self.x

    def reset_data_recorder(self):
        """
        Reset the internal data recorder of the simulator
        """

        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def record_data(self, time, x, tau):
        """
        Records data in the internal data recorder

        Parameters
        ----------
        time : float
            time to be recorded, unit: s
        x : type as self.plant expects a state
            state to be recorded, units: rad, rad/s
        tau : type as self.plant expects an actuation
            torque to be recorded, unit: Nm
        """

        self.t_values.append(np.copy(time))
        self.x_values.append(np.copy(x))
        self.tau_values.append(np.copy(tau))

    def euler_integrator(self, t, y, tau):
        """
        Euler integrator for the simulated plant

        Parameters
        ----------
        t : float
            time, unit: s
        y: type as self.plant expects a state
            state of the pendulum
        tau:  type as self.plant expects an actuation
            torque input

        Returns
        -------
        array-like : the Euler integrand
        """

        return self.plant.rhs(t, y, tau)

    def runge_integrator(self, t, y, dt, tau):
        """
        Runge-Kutta integrator for the simulated plant

        Parameters
        ----------
        t : float
            time, unit: s
        y: type as self.plant expects a state
            state of the pendulum
        dt: float
            time step, unit: s
        tau: type as self.plant expects an actuation
            torque input

        Returns
        -------
        array-like : the Runge-Kutta integrand
        """

        k1 = self.plant.rhs(t, y, tau)
        k2 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, tau)
        k3 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, tau)
        k4 = self.plant.rhs(t + dt, y + dt * k3, tau)
        return (k1 + 2 * (k2 + k3) + k4) / 6.0

    def step(self, tau, dt, integrator="runge_kutta"):
        """
        Performs a single step of the plant.

        Parameters
        ----------
        tau: type as self.plant expects an actuation
            torque input
        dt: float
            time step, unit: s
        integrator: string
            "euler" for euler integrator
            "runge_kutta" for Runge-Kutta integrator
        """

        if integrator == "runge_kutta":
            self.x += dt * self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x += dt * self.euler_integrator(self.t, self.x, tau)
        else:
            raise NotImplementedError(
                   f'Sorry, the integrator {integrator} is not implemented.')
        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)

    def simulate(self, t0, x0, tf, dt, controller=None,
                 integrator="runge_kutta"):
        """
        Simulates the plant over a period of time.

        Parameters
        ----------
        t0: float
            start time, unit s
        x0: type as self.plant expects a state
            start state
        tf: float
            final time, unit: s
        controller: A controller object of the type of the
                    AbstractController in
                    simple_pendulum.controllers.abstract_controller.py
                    If None, a free pendulum is simulated.
        integrator: string
            "euler" for euler integrator,
            "runge_kutta" for Runge-Kutta integrator

        Returns
        -------
        self.t_values : list
            a list of time values
        self.x_values : list
            a list of states
        self.tau_values : list
            a list of torques
        """

        self.set_state(t0, x0)
        self.reset_data_recorder()

        while (self.t <= tf):
            if controller is not None:
                _, _, tau = controller.get_control_output(
                                        meas_pos=self.x[:self.plant.dof],
                                        meas_vel=self.x[self.plant.dof:],
                                        meas_tau=np.zeros(self.plant.dof),
                                        meas_time=self.t)
            else:
                tau = np.zeros(self.plant.n_actuators)
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values

    def _animation_init(self):
        """
        init of the animation plot
        """

        self.animation_ax.set_xlim(self.plant.workspace_range[0][0],
                                   self.plant.workspace_range[0][1])
        self.animation_ax.set_ylim(self.plant.workspace_range[1][0],
                                   self.plant.workspace_range[1][1])
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots[:-1]:
            ap.set_data([], [])
        self.animation_plots[-1].set_text("t = 0.000")

        self.tau_arrowarcs = []
        self.tau_arrowheads = []
        for link in range(self.plant.n_links):
            arc, head = get_arrow(radius=0.001,
                                  centX=0,
                                  centY=0,
                                  angle_=110,
                                  theta2_=320,
                                  color_="red")
            self.tau_arrowarcs.append(arc)
            self.tau_arrowheads.append(head)
            self.animation_ax.add_patch(arc)
            self.animation_ax.add_patch(head)

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """

        t0 = time.time()
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        if controller is not None:
            _, _, tau = controller.get_control_output(
                meas_pos=self.x[:self.plant.dof],
                meas_vel=self.x[self.plant.dof:],
                meas_tau=np.zeros(self.plant.dof),
                meas_time=self.t)
        else:
            tau = np.zeros(self.plant.n_actuators)
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.plant.forward_kinematics(self.x[:self.plant.dof])
        ee_pos.insert(0, self.plant.base)
        ani_plot_counter = 0
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(ee_pos[link+1][0],
                                                            ee_pos[link+1][1])
            ani_plot_counter += 1
            self.animation_plots[ani_plot_counter].set_data(
                            [ee_pos[link][0], ee_pos[link+1][0]],
                            [ee_pos[link][1], ee_pos[link+1][1]])
            ani_plot_counter += 1

            set_arrow_properties(self.tau_arrowarcs[link],
                                 self.tau_arrowheads[link],
                                 float(np.squeeze(tau)),
                                 ee_pos[link][0],
                                 ee_pos[link][1])
        t = float(self.animation_plots[ani_plot_counter].get_text()[4:])
        t = round(t+dt, 3)
        self.animation_plots[ani_plot_counter].set_text(f"t = {t}")

        # if the animation runs slower than real time
        # the time display will be red
        if time.time() - t0 > dt:
            self.animation_plots[ani_plot_counter].set_color("red")
        else:
            self.animation_plots[ani_plot_counter].set_color("black")
        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _ps_init(self):
        """
        init of the phase space animation plot
        """

        self.ps_ax.set_xlim(-np.pi, np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.ps_plots:
            ap.set_data([], [])
        return self.ps_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """

        for d in range(self.plant.dof):
            self.ps_plots[d].set_data(
                            np.asarray(self.x_values).T[d],
                            np.asarray(self.x_values).T[self.plant.dof+d])
        return self.ps_plots

    def simulate_and_animate(self, t0, x0, tf, dt, controller=None,
                             integrator="runge_kutta", phase_plot=False,
                             save_video=False, video_name="video"):

        """
        Simulation and animation of the plant motion.
        The animation is only implemented for 2d serial chains.
        input:
        Simulates the plant over a period of time.

        Parameters
        ----------
        t0: float
            start time, unit s
        x0: type as self.plant expects a state
            start state
        tf: float
            final time, unit: s
        controller: A controller object of the type of the
                    AbstractController in
                    simple_pendulum.controllers.abstract_controller.py
                    If None, a free pendulum is simulated.
        integrator: string
            "euler" for euler integrator,
            "runge_kutta" for Runge-Kutta integrator
        phase_plot: bool
            whether to show a plot of the phase space together with
            the animation
        save_video: bool
            whether to save the animation as mp4 video
        video_name: string
            if save_video, the name of the file where the video will be stored

        Returns
        -------
        self.t_values : list
            a list of time values
        self.x_values : list
            a list of states
        self.tau_values : list
            a list of torques
        """

        self.set_state(t0, x0)
        self.reset_data_recorder()

        fig = plt.figure(figsize=(20, 20))
        self.animation_ax = plt.axes()
        self.animation_plots = []

        for link in range(self.plant.n_links):
            ee_plot, = self.animation_ax.plot([], [], "ro",
                                              markersize=25.0, color="blue")
            self.animation_plots.append(ee_plot)
            bar_plot, = self.animation_ax.plot([], [], "-",
                                               lw=5, color="black")
            self.animation_plots.append(bar_plot)

        text_plot = self.animation_ax.text(0.15, 0.85, [],
                                           fontsize=40,
                                           transform=fig.transFigure)

        self.animation_plots.append(text_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        animation = FuncAnimation(fig, self._animation_step, frames=frames,
                                  init_func=self._animation_init, blit=True,
                                  repeat=False, interval=dt*1000)

        if phase_plot:
            ps_fig = plt.figure(figsize=(10, 10))
            self.ps_ax = plt.axes()
            self.ps_plots = []
            for d in range(self.plant.dof):
                ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
                self.ps_plots.append(ps_plot)

            animation2 = FuncAnimation(ps_fig, self._ps_update,
                                       init_func=self._ps_init, blit=True,
                                       repeat=False, interval=dt*1000)

        if save_video:
            print(f"Saving video to {video_name}.mp4")
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save(video_name+'.mp4', writer=writer)
            print("Saving video done.")
        plt.show()

        return self.t_values, self.x_values, self.tau_values


def get_arrow(radius, centX, centY, angle_, theta2_, color_='black'):
    arc = Arc([centX, centY],
              radius,
              radius,
              angle=angle_,
              theta1=0,
              theta2=theta2_,
              capstyle='round',
              linestyle='-',
              lw=2,
              color=color_)

    endX = centX+(radius/2)*np.cos(rad(theta2_+angle_))
    endY = centY+(radius/2)*np.sin(rad(theta2_+angle_))

    head = RegularPolygon((endX, endY),            # (x,y)
                          3,                       # number of vertices
                          radius/20,               # radius
                          rad(angle_+theta2_),     # orientation
                          color=color_)
    return arc, head


def set_arrow_properties(arc, head, tau, x, y):
    tau_rad = np.clip(0.1*np.abs(tau) + 0.1, -1, 1)
    if tau > 0:
        theta2 = -40
        arrow_angle = 110
        endX = x+(tau_rad/2)*np.cos(rad(theta2+arrow_angle))
        endY = y+(tau_rad/2)*np.sin(rad(theta2+arrow_angle))
        orientation = rad(arrow_angle + theta2)
    else:
        theta2 = 320
        arrow_angle = 110
        endX = x+(tau_rad/2)*np.cos(rad(arrow_angle))
        endY = y+(tau_rad/2)*np.sin(rad(arrow_angle))
        orientation = rad(-arrow_angle-theta2)
    arc.center = [x, y]
    arc.width = tau_rad
    arc.height = tau_rad
    arc.angle = arrow_angle
    arc.theta2 = theta2

    head.xy = [endX, endY]
    head.radius = tau_rad/20
    head.orientation = orientation

    if np.abs(tau) <= 0.01:
        arc.set_visible(False)
        head.set_visible(False)
    else:
        arc.set_visible(True)
        head.set_visible(True)

class PendulumPlantApprox:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81,
                 coulomb_fric=0.0, inertia=None, torque_limit=np.inf, taylorApprox_order = 1, 
                 nominal_traj = None):

        """
        The PendulumPlantApprox class contains the taylor-approximated dynamics
        of the simple pendulum.

        The state of the pendulum in this class is described by
            state = [angle, angular velocity]
            (array like with len(state)=2)
            in units: rad and rad/s
        The zero state of the angle corresponds to the pendulum hanging down.
        The plant expects an actuation input (tau) either as float or
        array like in units Nm.
        (in which case the first entry is used (which should be a float))

        Parameters
        ----------
        mass : float, default=1.0
            pendulum mass, unit: kg
        length : float, default=0.5
            pendulum length, unit: m
        damping : float, default=0.1
            damping factor (proportional to velocity), unit: kg*m/s
        gravity : float, default=9.81
            gravity (positive direction points down), unit: m/s^2
        coulomb_fric : float, default=0.0
            friction term, (independent of magnitude of velocity), unit: Nm
        inertia : float, default=None
            inertia of the pendulum (defaults to point mass inertia)
            unit: kg*m^2
        torque_limit: float, default=np.inf
            maximum torque that the motor can apply, unit: Nm
        taylorApprox_order: int, default=1
            order of the taylor approximation of the sine term
        """

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        if inertia is None:
            self.inertia = mass*length*length
        else:
            self.inertia = inertia

        self.torque_limit = torque_limit

        self.nominal_t = nominal_traj

        self.dof = 1
        self.n_actuators = 1
        self.base = [0, 0]
        self.n_links = 1
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]
        self.order = taylorApprox_order

    def forward_dynamics(self, state, tau, time):

        """
        Computes forward dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float
            motor torque, unit: Nm

        Returns
        -------
            - float, angular acceleration, unit: rad/s^2
        """

        torque = np.clip(tau, -np.asarray(self.torque_limit),
                         np.asarray(self.torque_limit))

        # Taylor approximation of the sine term
        x0 = Variable("theta")
        Tsin_exp = TaylorExpand(sin(x0), {x0: self.nominal_t.value(time)[0] },self.order)
        Tsin = Tsin_exp.Evaluate({x0 : state[0]})

        accn = (torque - self.m * self.g * self.l * Tsin -
                self.b * state[1] - np.sign(state[1]) * self.coulomb_fric) / self.inertia
        return accn

    def rhs(self, t, state, tau):

        """
        Computes the integrand of the equations of motion.

        Parameters
        ----------
        t : float
            time, not used (the dynamics of the pendulum are time independent)
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float or array like
            motor torque, unit: Nm

        Returns
        -------
        res : array like
              the integrand, contains [angular velocity, angular acceleration]
        """

        if isinstance(tau, (list, tuple, np.ndarray)):
            torque = tau[0]
        else:
            torque = tau

        accn = self.forward_dynamics(state, torque, t)

        res = np.zeros(2*self.dof)
        res[0] = state[1]
        res[1] = accn
        return res
