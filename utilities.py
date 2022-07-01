import numpy as np
import yaml
import matplotlib.pyplot as plt
import scipy.linalg
from abc import ABC, abstractmethod


from pydrake.solvers.mathematicalprogram import Solve
from pydrake.systems.trajectory_optimization import DirectCollocation

from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, \
                        FiniteHorizonLinearQuadraticRegulator, \
                        PiecewisePolynomial, \
                        Linearize, \
                        LinearQuadraticRegulator, Variables, \
                        Jacobian, MathematicalProgram
from pydrake.examples.pendulum import PendulumPlant, PendulumState


class PendulumPlant_:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81,
                 coulomb_fric=0.0, inertia=None, torque_limit=np.inf):

        """
        The PendulumPlant class contains the kinematics and dynamics
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

        self.dof = 1
        self.n_actuators = 1
        self.base = [0, 0]
        self.n_links = 1
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def load_params_from_file(self, filepath):
        """
        Load the pendulum parameters from a yaml file.

        Parameters
        ----------
        filepath : string
            path to yaml file
        """

        with open(filepath, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
        self.m = params["mass"]
        self.l = params["length"]
        self.b = params["damping"]
        self.g = params["gravity"]
        self.coulomb_fric = params["coulomb_fric"]
        self.inertia = params["inertia"]
        self.torque_limit = params["torque_limit"]
        self.dof = params["dof"]
        self.n_actuators = params["n_actuators"]
        self.base = params["base"]
        self.n_links = params["n_links"]
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def forward_kinematics(self, pos):

        """
        Computes the forward kinematics.

        Parameters
        ----------
        pos : float, angle of the pendulum

        Returns
        -------
        list : A list containing one list (for one end-effector)
              The inner list contains the x and y coordinates
              for the end-effector of the pendulum
        """

        ee_pos_x = self.l * np.sin(pos)
        ee_pos_y = -self.l*np.cos(pos)
        return [[ee_pos_x, ee_pos_y]]

    def inverse_kinematics(self, ee_pos):

        """
        Comutes inverse kinematics

        Parameters
        ----------
        ee_pos : array like,
            len(state)=2
            contains the x and y position of the end_effector
            floats, units: m

        Returns
        -------
        pos : float
            angle of the pendulum, unit: rad
        """

        pos = np.arctan2(ee_pos[0]/self.l, ee_pos[1]/(-1.0*self.l))
        return pos

    def forward_dynamics(self, state, tau):

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

        accn = (torque - self.m * self.g * self.l * np.sin(state[0]) -
                self.b * state[1] -
                np.sign(state[1]) * self.coulomb_fric) / self.inertia
        return accn

    def inverse_dynamics(self, state, accn):

        """
        Computes inverse dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        accn : float
            angular acceleration, unit: rad/s^2

        Returns
        -------
        tau : float
            motor torque, unit: Nm
        """

        tau = accn * self.inertia + \
            self.m * self.g * self.l * np.sin(state[0]) + \
            self.b*state[1] + np.sign(state[1]) * self.coulomb_fric
        return tau

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

        accn = self.forward_dynamics(state, torque)

        res = np.zeros(2*self.dof)
        res[0] = state[1]
        res[1] = accn
        return res

    def potential_energy(self, state):
        Epot = self.m*self.g*self.l*(1-np.cos(state[0]))
        return Epot

    def kinetic_energy(self, state):
        Ekin = 0.5*self.m*(self.l*state[1])**2.0
        return Ekin

    def total_energy(self, state):
        E = self.potential_energy(state) + self.kinetic_energy(state)
        return E



class DirectCollocationCalculator():
    """
    Class to calculate a control trajectory with direct collocation.
    """
    def __init__(self):
        """
        Class to calculate a control trajectory with direct collocation.
        """
        self.pendulum_plant = PendulumPlant()
        self.pendulum_context = self.pendulum_plant.CreateDefaultContext()

    def init_pendulum(self, mass=0.57288, length=0.5, damping=0.15,
                      gravity=9.81, torque_limit=2.0):
        """
        Initialize the pendulum parameters.

        Parameters
        ----------
        mass : float, default=0.57288
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.15
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=2.0
            the torque_limit of the pendulum actuator
        """
        self.pendulum_params = self.pendulum_plant.get_mutable_parameters(
                                                        self.pendulum_context)
        self.pendulum_params[0] = mass
        self.pendulum_params[1] = length
        self.pendulum_params[2] = damping
        self.pendulum_params[3] = gravity
        self.torque_limit = torque_limit

    def compute_trajectory(self, N=21, max_dt=0.5, start_state=[0.0, 0.0],
                           goal_state=[np.pi, 0.0], initial_x_trajectory=None):
        """
        Compute a trajectory from a start state to a goal state
        for the pendulum.

        Parameters
        ----------
        N : int, default=21
            number of collocation points
        max_dt : float, default=0.5
            maximum allowed timestep between collocation points
        start_state : array_like, default=[0.0, 0.0]
            the start state of the pendulum [position, velocity]
        goal_state : array_like, default=[np.pi, 0.0]
            the goal state for the trajectory
        initial_x_trajectory : array-like, default=None
            initial guess for the state space trajectory
            ignored if None

        Returns
        -------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            trajectory in state space
        dircol : pydrake.systems.trajectory_optimization.DirectCollocation
            DirectCollocation pydrake object
        result : pydrake.solvers.mathematicalprogram.MathematicalProgramResult
            MathematicalProgramResult pydrake object
        """
        dircol = DirectCollocation(self.pendulum_plant,
                                   self.pendulum_context,
                                   num_time_samples=N,
                                   minimum_timestep=0.001,
                                   maximum_timestep=max_dt)

        dircol.AddEqualTimeIntervalsConstraints()

        u = dircol.input()
        dircol.AddConstraintToAllKnotPoints(-self.torque_limit <= u[0])
        dircol.AddConstraintToAllKnotPoints(u[0] <= self.torque_limit)

        initial_state = PendulumState()
        initial_state.set_theta(start_state[0])
        initial_state.set_thetadot(start_state[1])
        dircol.prog().AddBoundingBoxConstraint(initial_state.get_value(),
                                               initial_state.get_value(),
                                               dircol.initial_state())

        final_state = PendulumState()
        final_state.set_theta(goal_state[0])
        final_state.set_thetadot(goal_state[1])
        dircol.prog().AddBoundingBoxConstraint(final_state.get_value(),
                                               final_state.get_value(),
                                               dircol.final_state())

        R = 10.0  # Cost on input "effort".
        dircol.AddRunningCost(R * u[0]**2)

        if initial_x_trajectory is not None:
            dircol.SetInitialTrajectory(PiecewisePolynomial(),
                                        initial_x_trajectory)

        result = Solve(dircol.prog())
        assert result.is_success()

        x_trajectory = dircol.ReconstructStateTrajectory(result)
        return x_trajectory, dircol, result

    def plot_phase_space_trajectory(self, x_trajectory, save_to=None):
        """
        Plot the computed trajectory in phase space.

        Parameters
        ----------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            the trajectory returned from the compute_trajectory function.
        save_to : string, default=None
            string pointing to the location where the figure is supposed
            to be stored. If save_to==None, the figure is not stored but shown
            in a window instead.
        """
        fig, ax = plt.subplots()

        time = np.linspace(x_trajectory.start_time(),
                           x_trajectory.end_time(),
                           100)

        x_knots = np.hstack([x_trajectory.value(t) for t in time])

        ax.plot(x_knots[0, :], x_knots[1, :])
        if save_to is None:
            plt.show()
        else:
            plt.xlim(-np.pi, np.pi)
            plt.ylim(-10, 10)
            plt.savefig(save_to)
        plt.close()

    def extract_trajectory(self, x_trajectory, dircol, result, N=801):
        """
        Extract time, position, velocity and control trajectories from
        the outputs of the compute_trajectory function.

        Parameters
        ----------
        x_trajectory : pydrake.trajectories.PiecewisePolynomial
            trajectory in state space
        dircol : pydrake.systems.trajectory_optimization.DirectCollocation
            DirectCollocation pydrake object
        result : pydrake.solvers.mathematicalprogram.MathematicalProgramResult
            MathematicalProgramResult pydrake object
        N : int, default=801
            The number of sampling points of the returned trajectories

        Returns
        -------
        time_traj : array_like
            the time trajectory
        theta : array_like
            the position trajectory
        theta_dot : array_like
            the velocity trajectory
        torque_traj : array_like
            the control (torque) trajectory
        """
        # Extract Time
        time = np.linspace(x_trajectory.start_time(),
                           x_trajectory.end_time(),
                           N)
        time_traj = time.reshape(N, 1).T[0]

        # Extract State
        theta_theta_dot = np.hstack([x_trajectory.value(t) for t in time])

        theta = theta_theta_dot[0, :].reshape(N, 1).T[0]
        theta_dot = theta_theta_dot[1, :].reshape(N, 1).T[0]

        # Extract Control Inputs
        u_trajectory = dircol.ReconstructInputTrajectory(result)
        torque_traj = np.hstack([u_trajectory.value(t) for t in time])[0]

        return time_traj, theta, theta_dot, torque_traj

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    ref: Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the lqr gain
    K = np.array(scipy.linalg.inv(R).dot(B.T.dot(X)))
    eigVals, eigVecs = scipy.linalg.eig(A-B.dot(K))
    return K, X, eigVals


class AbstractController(ABC):
    """
    Abstract controller class. All controller should inherit from
    this abstract class.
    """
    @abstractmethod
    def get_control_output(self, meas_pos, meas_vel, meas_tau, meas_time):
        """
        The function to compute the control input for the pendulum actuator.
        Supposed to be overwritten by actual controllers. The API of this
        method should be adapted. Unused inputs/outputs can be set to None.

        **Parameters**

        ``meas_pos``: ``float``
            The position of the pendulum [rad]
        ``meas_vel``: ``float``
            The velocity of the pendulum [rad/s]
        ``meas_tau``: ``float``
            The meastured torque of the pendulum [Nm]
        ``meas_time``: ``float``
            The collapsed time [s]

        Returns
        -------
        ``des_pos``: ``float``
            The desired position of the pendulum [rad]
        ``des_vel``: ``float``
            The desired velocity of the pendulum [rad/s]
        ``des_tau``: ``float``
            The torque supposed to be applied by the actuator [Nm]
        """

        des_pos = None
        des_vel = None
        des_tau = None
        return des_pos, des_vel, des_tau

    def init(self, x0):
        """
        Initialize the controller. May not be necessary.

        Parameters
        ----------
        ``x0``: ``array like``
            The start state of the pendulum
        """
        self.x0 = x0

    def set_goal(self, x):
        """
        Set the desired state for the controller. May not be necessary.

        Parameters
        ----------
        ``x``: ``array like``
            The desired goal state of the controller
        """

        self.goal = x

class LQRController(AbstractController):
    """
    Controller which stabilizes the pendulum at its instable fixpoint.
    """
    def __init__(self, mass=1.0, length=0.5, damping=0.1,
                 gravity=9.81, torque_limit=np.inf):
        """
        Controller which stabilizes the pendulum at its instable fixpoint.
        Parameters
        ----------
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """

        self.goal = [np.pi, 0.0]

        self.m = mass
        self.len = length
        self.b = damping
        self.g = gravity
        self.torque_limit = torque_limit

        self.A = np.array([[0, 1],
                           [self.g/self.len, -self.b/(self.m*self.len**2.0)]])
        self.B = np.array([[0, 1./(self.m*self.len**2.0)]]).T
        self.Q = np.diag((10, 1))
        self.R = np.array([[1]])

        self.K, self.S, _ = lqr(self.A, self.B, self.Q, self.R)

        self.clip_out = False

    def set_goal(self, x):
        self.goal = x

    def set_clip(self):
        self.clip_out = True

    def get_control_output(self, meas_pos, meas_vel,
                           meas_tau=0, meas_time=0):
        """
        The function to compute the control input for the pendulum actuator
        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float, default=0
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float, default=0
            the collapsed time [s]
            (not used)
        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        #th = pos + np.pi
        #th = (th + np.pi) % (2*np.pi) - np.pi
        th = pos - self.goal[0]

        y = np.asarray([th, vel])

        u = np.asarray(-self.K.dot(y))[0]

        if not self.clip_out:
            if np.abs(u) > self.torque_limit:
                u = None

        else:
            u = np.clip(u, -self.torque_limit, self.torque_limit)

        

        # since this is a pure torque controller,
        # set des_pos and des_pos to None
        des_pos = self.goal[0]
        des_vel = self.goal[1]

        return des_pos, des_vel, u


class TVLQRController(AbstractController):
    """
    Controller acts on a predefined trajectory.
    """
    def __init__(self,
                 data_dict,
                 mass=1.0,
                 length=0.5,
                 damping=0.1,
                 gravity=9.81,
                 torque_limit=np.inf):
        """
        Controller acts on a predefined trajectory.

        Parameters
        ----------
        data_dict : dictionary
            a dictionary containing the trajectory to follow
            should have the entries:
            data_dict["des_time_list"] : desired timesteps
            data_dict["des_pos_list"] : desired positions
            data_dict["des_vel_list"] : desired velocities
            data_dict["des_tau_list"] : desired torques
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """

        # load the trajectory
        self.traj_time = data_dict["des_time_list"]
        self.traj_pos = data_dict["des_pos_list"]
        self.traj_vel = data_dict["des_vel_list"]
        self.traj_tau = data_dict["des_tau_list"]

        self.max_time = self.traj_time[-1]

        self.traj_time = np.reshape(self.traj_time,
                                    (self.traj_time.shape[0], -1))
        self.traj_tau = np.reshape(self.traj_tau,
                                   (self.traj_tau.shape[0], -1)).T

        x0_desc = np.vstack((self.traj_pos, self.traj_vel))
        # u0_desc = self.traj_tau

        u0 = PiecewisePolynomial.FirstOrderHold(self.traj_time, self.traj_tau)
        x0 = PiecewisePolynomial.CubicShapePreserving(
                                              self.traj_time,
                                              x0_desc,
                                              zero_end_point_derivatives=True)

        # create drake pendulum plant
        self.plant = PendulumPlant()
        self.context = self.plant.CreateDefaultContext()
        params = self.plant.get_mutable_parameters(self.context)
        params[0] = mass
        params[1] = length
        params[2] = damping
        params[3] = gravity

        self.torque_limit = torque_limit

        # create lqr context
        self.tilqr_context = self.plant.CreateDefaultContext()
        self.plant.get_input_port(0).FixValue(self.tilqr_context, [0])
        self.Q_tilqr = np.diag((50., 1.))
        self.R_tilqr = [1]

        # Setup Options and Create TVLQR
        self.options = FiniteHorizonLinearQuadraticRegulatorOptions()
        self.Q = np.diag([200., 0.1])
        self.options.x0 = x0
        self.options.u0 = u0

        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def init(self, x0):
        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def set_goal(self, x):
        pos = x[0] + np.pi
        pos = (pos + np.pi) % (2*np.pi) - np.pi
        self.tilqr_context.SetContinuousState([pos, x[1]])
        linearized_pendulum = Linearize(self.plant, self.tilqr_context)
        (K, S) = LinearQuadraticRegulator(linearized_pendulum.A(),
                                          linearized_pendulum.B(),
                                          self.Q_tilqr,
                                          self.R_tilqr)

        self.options.Qf = S  
        
        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(
                        self.plant,
                        self.context,
                        t0=self.options.u0.start_time(),
                        tf=self.options.u0.end_time(),
                        Q=self.Q,
                        R=np.eye(1)*2,
                        options=self.options)

    def get_control_output(self, meas_pos=None, meas_vel=None, meas_tau=None,
                           meas_time=None):
        """
        The function to read and send the entries of the loaded trajectory
        as control input to the simulator/real pendulum.

        Parameters
        ----------
        meas_pos : float, deault=None
            the position of the pendulum [rad]
        meas_vel : float, deault=None
            the velocity of the pendulum [rad/s]
        meas_tau : float, deault=None
            the meastured torque of the pendulum [Nm]
        meas_time : float, deault=None
            the collapsed time [s]

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
        des_vel : float
            the desired velocity of the pendulum [rad/s]
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))
        x = np.array([[pos], [vel]])

        des_pos = self.last_pos
        des_vel = self.last_vel
        # des_tau = 0

        if self.counter < len(self.traj_pos):
            des_pos = self.traj_pos[self.counter]
            des_vel = self.traj_vel[self.counter]
            # des_tau = self.traj_tau[self.counter]
            self.last_pos = des_pos
            self.last_vel = des_vel

        self.counter += 1

        time = min(meas_time, self.max_time)

        uu = self.tvlqr.u0.value(time)
        xx = self.tvlqr.x0.value(time)
        KK = self.tvlqr.K.value(time)
        kk = self.tvlqr.k0.value(time)

        xdiff = x - xx
        pos_diff = (xdiff[0] + np.pi) % (2*np.pi) - np.pi
        xdiff[0] = pos_diff

        des_tau = (uu - KK.dot(xdiff) - kk)[0][0]
        des_tau = np.clip(des_tau, -self.torque_limit, self.torque_limit)

        return des_pos, des_vel, des_tau

def prepare_trajectory(csv_path):
    """
    inputs:
        csv_path: string
            path to a csv file containing a trajectory in the
            below specified format

    The csv file should have 4 columns with values for
    [time, position, velocity, torque] respectively.
    The values shopuld be separated with a comma.
    Each row in the file is one timestep. The number of rows can vary.
    The values are assumed to be in SI units, i.e. time in s, position in rad,
    velocity in rad/s, torque in Nm.
    The first line in the csv file is reserved for comments
    and will be skipped during read out.

    Example:

        # time, position, velocity, torque
        0.00, 0.00, 0.00, 0.10
        0.01, 0.01, 0.01, -0.20
        0.02, ....

    """
    # load trajectories from csv file
    trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
    des_time_list = trajectory.T[0].T                       # desired time in s
    des_pos_list = trajectory.T[1].T               # desired position in radian
    des_vel_list = trajectory.T[2].T             # desired velocity in radian/s
    des_tau_list = trajectory.T[3].T                     # desired torque in Nm

    n = len(des_time_list)
    t = des_time_list[n - 1]
    dt = round((des_time_list[n - 1] - des_time_list[0]) / n, 3)

    # create 4 empty numpy array, where measured data can be stored
    meas_time_list = np.zeros(n)
    meas_pos_list = np.zeros(n)
    meas_vel_list = np.zeros(n)
    meas_tau_list = np.zeros(n)
    vel_filt_list = np.zeros(n)

    data_dict = {"des_time_list": des_time_list,
                 "des_pos_list": des_pos_list,
                 "des_vel_list": des_vel_list,
                 "des_tau_list": des_tau_list,
                 "meas_time_list": meas_time_list,
                 "meas_pos_list": meas_pos_list,
                 "meas_vel_list": meas_vel_list,
                 "meas_tau_list": meas_tau_list,
                 "vel_filt_list": vel_filt_list,
                 "n": n,
                 "dt": dt,
                 "t": t}
    return data_dict

def rhoVerification(rho, pendulum, controller):
    """SOS Verification of the Lyapunov conditions for a given rho in order to obtain an estimate  of the RoA for the closed loop dynamics.
       This method is described by Russ Tedrake in "Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation", 
       Course Notes for MIT 6.832, 2022, "http://underactuated.mit.edu", sec. 9.3.2: "Basic region of attraction formulation".

    Parameters
    ----------
    rho: float
        value of rho to be verified
    pendulum : simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller : simple_pendulum.controllers.lqr.lqr_controller
        configured lqr controller object

    Returns
    -------
    result : boolean
        result of the verification
    """

    #K and S matrices from LQR control
    K = controller.K
    S = controller.S

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # non-linear dyamics
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "x")
    xg = [np.pi, 0]  # reference
    x = xbar + xg
    ubar = -K.dot(xbar)[0] # control input with reference
    Tsin = -(x[0]-xg[0]) + (x[0]-xg[0])**3/6 - (x[0]-xg[0])**5/120 + (x[0]-xg[0])**7/5040
    fn = [x[1], (ubar-b*x[1]-Tsin*m*g*l)/(m*l*l)]

    # cost-to-go of LQR as Lyapunov candidate
    V = (xbar).dot(S.dot(xbar))
    Vdot = Jacobian([V], xbar).dot(fn)[0]

    # Saturation for fn and Vdot
    u_minus = - torque_limit
    u_plus = torque_limit
    fn_minus = [x[1], (u_minus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_minus = V.Jacobian(xbar).dot(fn_minus)
    fn_plus = [x[1], (u_plus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_plus = V.Jacobian(xbar).dot(fn_plus)

    # Define the Lagrange multipliers.
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_5 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_6 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    lambda_7 = prog.NewSosPolynomial(Variables(xbar), 6)[0].ToExpression()
    epsilon=10e-10

    # Optimization constraints
    prog.AddSosConstraint(-Vdot_minus + lambda_1*(V-rho) + lambda_2*(-u_minus+ubar) - epsilon*xbar.dot(xbar))
    prog.AddSosConstraint(-Vdot + lambda_3*(V-rho) + lambda_4*(u_minus-ubar) + lambda_5*(-u_plus+ubar) - epsilon*xbar.dot(xbar))
    prog.AddSosConstraint(-Vdot_plus + lambda_6*(V-rho) + lambda_7*(u_plus-ubar) - epsilon*xbar.dot(xbar))

    # Solve the problem
    result = Solve(prog).is_success()
    return result

def SOSequalityConstrained(pendulum, controller):
    """Estimate the RoA for the closed loop dynamics using the method described by Russ Tedrake in "Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation", 
       Course Notes for MIT 6.832, 2022, "http://underactuated.mit.edu", sec. 9.3.2: "The equality-constrained formulation".
       This is discussed a bit more by Shen Shen and Russ Tedrake in "Sampling Quotient-Ring Sum-of-Squares Programs for Scalable Verification of Nonlinear Systems", 
       Proceedings of the 2020 59th IEEE Conference on Decision and Control (CDC) , 2020., http://groups.csail.mit.edu/robotics-center/public_papers/Shen20.pdf, pg. 2-3. 

    Parameters
    ----------
    pendulum : simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller : simple_pendulum.controllers.lqr.lqr_controller
        configured lqr controller object

    Returns
    -------
    rho : float
        estimated value of rho
    S : np.array
        S matrix from the lqr controller
    """

    # K and S matrices from LQR control
    K = controller.K
    S = controller.S

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # Opt. problem definition
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "x") # shifted system state
    rho = prog.NewContinuousVariables(1)[0]

    # Symbolic and polynomial dynamic linearized using Taylor approximation
    xg = [np.pi, 0]  # goal state to stabilize
    x = xbar + xg
    ubar = -K.dot(xbar)[0]
    Tsin = -(x[0]-xg[0]) + (x[0]-xg[0])**3/6 - (x[0]-xg[0])**5/120 + (x[0]-xg[0])**7/5040 
    fn = [x[1], (ubar-b*x[1]-Tsin*m*g*l)/(m*l*l)]

    # Optimal cost-to-go from LQR as Lyapunov candidate
    V = (xbar).dot(S.dot(xbar))
    Vdot = Jacobian([V], xbar).dot(fn)[0]

    # Free multipliers
    lambda_a = prog.NewFreePolynomial(Variables(xbar), 4).ToExpression()
    lambda_b = prog.NewFreePolynomial(Variables(xbar), 4).ToExpression()
    lambda_c = prog.NewFreePolynomial(Variables(xbar), 4).ToExpression()

    # Boundaries due to the saturation 
    u_minus = - torque_limit
    u_plus = torque_limit
    fn_minus = [x[1], (u_minus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_minus = V.Jacobian(xbar).dot(fn_minus)
    fn_plus = [x[1], (u_plus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_plus = V.Jacobian(xbar).dot(fn_plus)

    # Define the Lagrange multipliers.
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()

    # Optimization constraints and cost
    prog.AddSosConstraint(((xbar.T).dot(xbar)**2)*(V - rho) + lambda_a*(Vdot_minus) + lambda_1*(-u_minus+ubar))
    prog.AddSosConstraint(((xbar.T).dot(xbar)**2)*(V - rho) + lambda_b*(Vdot) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar))
    prog.AddSosConstraint(((xbar.T).dot(xbar)**2)*(V - rho) + lambda_c*(Vdot_plus) + lambda_4*(u_plus-ubar))
    prog.AddConstraint(rho >= 0)
    prog.AddCost(-rho)

    # Solve the problem
    result = Solve(prog)
    return [result.GetSolution(rho), S]

def direct_sphere(d,r_i=0,r_o=1):
    """Direct Sampling from the d Ball based on Krauth, Werner. Statistical Mechanics: Algorithms and Computations. Oxford Master Series in Physics 13. Oxford: Oxford University Press, 2006. page 42

    Parameters
    ----------
    d : int
        dimension of the ball
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector directly sampled from the solid d Ball
    """
    # vector of univariate gaussians:
    rand=np.random.normal(size=d)
    # get its euclidean distance:
    dist=np.linalg.norm(rand,ord=2)
    # divide by norm
    normed=rand/dist
    
    # sample the radius uniformly from 0 to 1 
    rad=np.random.uniform(r_i,r_o**d)**(1/d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed*rad

def sample_from_ellipsoid(M,rho,r_i=0,r_o=1):
    """sample directly from the ellipsoid defined by xT M x.

    Parameters
    ----------
    M : np.array
        Matrix M such that xT M x leq rho defines the hyperellipsoid to sample from
    rho : float
        rho such that xT M x leq rho defines the hyperellipsoid to sample from
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector from within the hyperellipsoid
    """
    lamb,eigV=np.linalg.eigh(M/rho) 
    d=len(M)
    xy=direct_sphere(d,r_i=r_i,r_o=r_o) #sample from outer shells
    T=np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)),eigV.T)) #transform sphere to ellipsoid (refer to e.g. boyd lectures on linear algebra)
    return np.dot(T,xy.T).T

def quad_form(M,x):
    """
    Helper function to compute quadratic forms such as x^TMx
    """
    return np.dot(x,np.dot(M,x))

def projectedEllipseFromCostToGo(s0Idx,s1Idx,rho,M):
    """
    Returns ellipses in the plane defined by the states matching the indices s0Idx and s1Idx for funnel plotting.
    """
    ellipse_widths=[]
    ellipse_heights=[]
    ellipse_angles=[]
    
    #loop over all values of rho
    for idx, rho in enumerate(rho):
        #extract 2x2 matrix from S
        S=M[idx]
        ellipse_mat=np.array([[S[s0Idx][s0Idx],S[s0Idx][s1Idx]],
                              [S[s1Idx][s0Idx],S[s1Idx][s1Idx]]])*(1/rho)
        
        #eigenvalue decomposition to get the axes
        w,v=np.linalg.eigh(ellipse_mat) 

        try:
            #let the smaller eigenvalue define the width (major axis*2!)
            ellipse_widths.append(2/float(np.sqrt(w[0])))
            ellipse_heights.append(2/float(np.sqrt(w[1])))

            #the angle of the ellipse is defined by the eigenvector assigned to the smallest eigenvalue (because this defines the major axis (width of the ellipse))
            ellipse_angles.append(np.rad2deg(np.arctan2(v[:,0][1],v[:,0][0])))
        except:
            continue
    return ellipse_widths,ellipse_heights,ellipse_angles

def getEllipseContour(S,rho,xg):
    """
    Returns a certain number(nSamples) of sampled states from the contour of a given ellipse.

    Parameters
    ----------
    S : np.array
        Matrix S that define one ellipse
    rho : np.array
        rho value that define one ellipse
    xg : np.array
        center of the ellipse

    Returns
    -------
    c : np.array
        random vector of states from the contour of the ellipse
    """
    nSamples = 1000 
    c = sample_from_ellipsoid(S,rho,r_i = 0.99) +xg
    for i in range(nSamples-1):
        xBar = sample_from_ellipsoid(S,rho,r_i = 0.99)
        c = np.vstack((c,xBar+xg))
    return c

def get_params(params_path):
    with open(params_path, 'r') as fle:
        params = yaml.safe_load(fle)
    return params