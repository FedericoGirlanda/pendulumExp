import numpy as np
from pydrake.all import Variables, MonomialBasis, Solve, MathematicalProgram, Jacobian, PiecewisePolynomial
from pydrake.symbolic import Polynomial as simb_poly
from pydrake.symbolic import sin, TaylorExpand

def TVrhoSearch(pendulum, controller, knot, time, h_map, rho_t):

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # Opt. problem definition
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "x") # shifted system state
    rho_i = prog.NewContinuousVariables(1)[0]
    rho_dot_i = (rho_t[knot] - rho_i)/dt
    prog.AddCost(-rho_i)
    prog.AddConstraint(rho_i >= 0)

    # Dynamics definition
    u0 = controller.tvlqr.u0.value(t_i)[0][0]
    K_i = controller.tvlqr.K.value(t_i)[0]
    ubar = - K_i.dot(xbar)
    u = (u0 + ubar) #input

    x0 = controller.tvlqr.x0.value(t_i)
    x = (xbar + x0)[0]
    Tsin_x = TaylorExpand(sin(x[0]),{xbar[0]: 0}, 5)
    fn = [xbar[1], (ubar -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)] # shifted state dynamics

    # Lyapunov function and its derivative
    S_t = controller.tvlqr.S
    S_i = S_t.value(t_i)
    S_iplus1 = S_t.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt
    V_i = (xbar).dot(S_i.dot(xbar))
    Vdot_i_x = V_i.Jacobian(xbar).dot(fn)
    Vdot_i_t = xbar.dot(Sdot_i.dot(xbar))
    Vdot_i = Vdot_i_x + Vdot_i_t

    # Boundaries due to the saturation 
    u_minus = - torque_limit -u0
    u_plus = torque_limit -u0
    fn_minus = [xbar[1], (u_minus -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)]
    Vdot_minus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_minus)
    fn_plus = [xbar[1], (u_plus -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)]
    Vdot_plus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_plus)

    # Multipliers definition
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()

    # Retriving the mu result 
    h = prog.NewFreePolynomial(Variables(xbar), 4)
    ordered_basis = list(h.monomial_to_coefficient_map().keys())
    zip_iterator = zip(ordered_basis, list(h_map.values()))
    h_dict = dict(zip_iterator)
    h = simb_poly(h_dict)
    h.RemoveTermsWithSmallCoefficients(4)
    mu_ij = h.ToExpression()

    # Optimization constraints 
    constr_minus = - (Vdot_minus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_1*(-u_minus+ubar) 
    constr = - (Vdot_i) + rho_dot_i + mu_ij*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar) 
    constr_plus = - (Vdot_plus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_4*(u_plus-ubar) 

    for c in [constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    # Solve the problem
    result = Solve(prog)
    rho_opt = result.GetSolution(rho_i)

    # failing checker
    fail = not result.is_success()
    if fail:
        print("rho step Error")

    return fail, rho_opt

def TVmultSearch(pendulum, controller, knot, time, rho_t):

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # Opt. problem definition
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "x") # shifted system state
    gamma = prog.NewContinuousVariables(1)[0]
    prog.AddCost(gamma)
    prog.AddConstraint(gamma <= 0)

    # Dynamics definition
    u0 = controller.tvlqr.u0.value(t_i)[0][0]
    K_i = controller.tvlqr.K.value(t_i)[0]
    ubar = - K_i.dot(xbar)
    u = (u0 + ubar) #input

    x0 = controller.tvlqr.x0.value(t_i)
    x = (xbar + x0)[0]
    Tsin_x = TaylorExpand(sin(x[0]),{xbar[0]: 0}, 5)
    fn = [xbar[1], (ubar -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)] # shifted state dynamics

    # Lyapunov function and its derivative
    S_t = controller.tvlqr.S
    S_i = S_t.value(t_i)
    S_iplus1 = S_t.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt
    V_i = (xbar).dot(S_i.dot(xbar))
    Vdot_i_x = V_i.Jacobian(xbar).dot(fn)
    Vdot_i_t = xbar.dot(Sdot_i.dot(xbar))
    Vdot_i = Vdot_i_x + Vdot_i_t

    # Boundaries due to the saturation 
    u_minus = - torque_limit -u0
    u_plus = torque_limit -u0
    fn_minus = [xbar[1], (u_minus -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)]
    Vdot_minus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_minus)
    fn_plus = [xbar[1], (u_plus -b*xbar[1]-(Tsin_x-np.sin(x0[0]))*m*g*l)/(m*l*l)]
    Vdot_plus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_plus)

    # Multipliers definition
    h = prog.NewSosPolynomial(Variables(xbar), 4)[0]
    mu_ij = h.ToExpression()
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()

    # rho dot definition
    rho_i = rho_t[knot-1]
    rho_iplus1 = rho_t[knot]
    rho_dot_i = (rho_iplus1 - rho_i)/dt

    # Optimization constraints 
    constr_minus = - (Vdot_minus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_1*(-u_minus+ubar) + gamma
    constr = - (Vdot_i) + rho_dot_i + mu_ij*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar) + gamma
    constr_plus = - (Vdot_plus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_4*(u_plus-ubar) + gamma

    for c in [constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    # Solve the problem and store the polynomials
    result_mult = Solve(prog)  

    h_map = result_mult.GetSolution(h).monomial_to_coefficient_map()
    eps = result_mult.get_optimal_cost()

    # failing checker
    fail = not result_mult.is_success()
    if fail:
        print(f"mult step Error, decreasing rho to make it feasible...")  

    # go ahead if right enough, step back?
    if eps < np.inf:
        if (round(eps*10e-4) == 0):
            fail = False

    return fail, h_map