# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:52:17 2015

@author: Jesse VanderWees

"""

import time

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)


def initialize():
    """
    Geometry is a wide duct with velocity defined on the inlet and outlet,
    and an initial velocity of zero everywhere else. To reflect no-slip
    conditions, the velocity along the top and bottom of the pipe will
    always be zero. The grid is set up so that the pressure does
    not need to be defined in the initial conditions.
    """

    # Duct Dimensions
    length = 0.02  # m, horizontal dimension, cooresponds to u
    width = 0.0125  # m, vertical dimension, cooresponds to v

    # Assume srid spacing equal in horizontal and vertical
    res = 120  # Grid Resolution, points per width
    v_in = 0.0055765  # m/s, initial velocity u distributed on inlet and outlet

    # Pseudo-speed of Sound (v_max / a < 1)
    a = 100  # m/s

    # Fluid Properties (at 20 deg C)
    rho = 998.23  # Density, kg/m^3 (honey = 1420 kg/m^3, water = 998.23 kg/m^3)
    nu = 0.001002  # dynamic viscosity, Pa*s (honey = 2-10, water = 0.001002)

    return length, width, res, v_in, a, rho, nu


def Re(p, rho, len_scale, v_in, nu):
    """
    The Reynolds number is the ratio of momentum forces to viscous forces. It
    is calculated using the equation:
        Re = rho * V * len_scale / nu

    rho --> density
    V --> maximum velocity
    len_scale --> parameter specific to geometry, 2x width for parallel plates
    nu --> dynamic viscosity

    take_step() was written to accept a np.array with the same shape as p, so
    Re() must
    """

    Re_vec = np.zeros_like(p)
    Re = rho * v_in * len_scale / nu
    Re_vec = Re_vec + Re

    """
    If Re were calculated at every point P, it could look like this:

        V = np.zeros_like(p)
        u_p_x = np.zeros_like(p)
        v_p_y = np.zeros_like(p)

        u_p_x = (u[:-1,1:-1] + u[1:,1:-1]) * 0.5
        v_p_y = (v[1:-1,1:] + v[2:,0:-1]) * 0.5

        V = np.sqrt(u_p_x*u_p_x + v_p_y*v_p_y) * v_in

        Re = rho * V * len_scale / nu # Reynold's Number

    It would take to long for this implemantation to converge on my
    computer, if it converged at all.
    """
    return Re_vec


def delta_t_max(delta_x, delta_y, a, num_dimensions=2):
    """
    This function is an equation defined in the CFD textbook to estimate
    the largest stable value of delta_t for this numerical scheme.
    """
    return (
        2
        * np.min([delta_x, delta_y])
        / (np.sqrt(num_dimensions) * a * (1 + np.sqrt(5)))
    )


def take_step(p, u, v, delta_t, Re, delta_x, delta_y, a):
    """
    This function updates the values of p, u, and v to one time step ahead

    Implements Reynolds-Reduced Navier-Stokes in a 2-d stagered grid
    """

    # Extrapolate v and u on boundaries by reversing the averaging equation
    v[0, :] = 2 * v[1, :] - v[2, :]
    v[-1, :] = 2 * v[-2, :] - v[-3, :]
    u[:, 0] = -u[:, 1]
    u[:, -1] = -u[:, -2]

    # Evaluate some intermediate terms
    u2 = 0.25 * (u[1:, 1:-1] + u[:-1, 1:-1]) ** 2  # u^2; has extra dimension in u
    v2 = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1]) ** 2  # v^2; has extra dimension in v

    # These are sets of u*v values. uv1 has extra dimension in v
    uv1 = 0.25 * (u[1:, 1:-1] + u[1:, 2:]) * (v[1:-1, 1:] + v[2:, 1:])
    uv2 = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2]) * (v[1:-2, :-1] + v[2:-1, :-1])
    uv3 = 0.25 * (u[:-1, 1:-2] + u[:-1, 2:-1]) * (v[1:-1, 1:-1] + v[:-2, 1:-1])

    # Evaluate velocity terms
    u[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        - delta_t
        * (
            (u2[1:, :] - u2[:-1, :]) / delta_x
            + (p[1:, :] - p[:-1, :]) / delta_x
            + (uv1[:-1, :] - uv2) / delta_y
        )
        + (
            delta_t
            / ((Re[1:, :] + Re[:-1, :]) / 2)
            * (
                (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / delta_x ** 2
                + (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / delta_y ** 2
            )
        )
    )

    v[1:-1, 1:-1] = (
        v[1:-1, 1:-1]
        - delta_t
        * (
            (v2[:, 1:] - v2[:, :-1]) / delta_y
            + (p[:, 1:] - p[:, :-1]) / delta_y
            + (uv1[:, :-1] - uv3) / delta_x
        )
        + (
            delta_t
            / ((Re[:, 1:] + Re[:, :-1]) / 2)
            * (
                (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / delta_x ** 2
                + (v[1:-1, :-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / delta_y ** 2
            )
        )
    )

    # Evaluate Pressure terms. Presure terms use newly updated u and v values
    p = p - delta_t * a ** 2 * (
        (u[1:, 1:-1] - u[:-1, 1:-1]) / delta_x + (v[1:-1, 1:] - v[1:-1, :-1]) / delta_y
    )
    return p, u, v


def u_theoretical(npts, v_avg, fullwidth=True):
    """
    This equation comes from:
    http://www.calpoly.edu/~kshollen/ME343/Examples/Example_14.pdf

    It describes the velocity profile of fluid flow between two parallel
    plates under the following conditions:

    1. steady flow
    2. constant properties
    3. Newtonian fluid
    4. negligible radiation
    5. negligible gravity effects
    6. laminar flow
    7. negligible viscous dissipation
    8. fully developed
    9. negligible end effects
    10. conduction in y-direction much greater than conduction in x-direction

    Therefore, the velocity profile approaches this shape as Re approaches 1.
    """

    # This makes a theoretical u curve of npts points

    u_theor = np.empty(npts)

    if not fullwidth:
        for i in range(npts):
            u_theor[i] = 3.0 / 2.0 * v_avg * (1 - ((npts - i) / npts) ** 2)
        return u_theor

    for i in range(npts // 2 + 1):
        u_theor[i] = 3.0 / 2.0 * v_avg * (1 - ((npts - i * 2) / npts) ** 2)
        u_theor[-i - 1] = 3.0 / 2.0 * v_avg * (1 - ((npts - i * 2) / npts) ** 2)
    return u_theor


def compare_u_profile(u, delta_y, width, v_in, Re):
    """
    Makes a plot comparing the velocity profile at the center of u to the
    theoretically derived profile shape. This function takes a dimensional
    u and outputs results in denormalized form.
    """

    # This sets up y and u
    u = u[len(u[0, :]) // 2, :]
    y = np.zeros_like(u)
    for i in range(len(y)):
        y[i] = (i - 0.5) * delta_y

    # This makes a theoretical u and y curve of npts points
    npts = 50
    y_theor = np.linspace(0, width, npts)
    u_theor = u_theoretical(npts, v_in)

    # This plots the calculated and theoretical u velocity profiles
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_ylim([0, width])
    ax.set_xlim([0, np.max(u_theor) * 1.1])
    ax.plot(u, y, label="Simulated: Re = %2.1f" % Re)
    ax.plot(u_theor, y_theor, label="Theoretical (Re < 100)")
    ax.set_title("Comparison of Simulated Velocity Profile to Theoretical")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Vertical Displacement [m]")
    ax.legend(loc="best")
    plt.show()


def make_planes(length, width, res):
    # Normalize distance terms by deviding by width,
    # Normalize velocity terms by deviding by v_in

    delta_x = (length / width) / res
    delta_y = 1.0 / res

    # Force Conformity to Grid
    n_x = int((length / width) * res)
    n_y = int(res)

    p = np.zeros([n_x, n_y])
    u = np.zeros([n_x + 1, n_y + 2])
    v = np.zeros([n_x + 2, n_y + 1])

    return p, u, v, delta_x, delta_y


def infinite_planes(tol=0.01):
    """
    This function takes one attribute, tol, the criteria for convergence. When
    the maximum change in p between consecutive timesteps is less than tol,
    the continuity equation is considered met and the system has converged.
    """

    start_time = time.time()
    # Set up variables
    length, width, res, v_in, a, rho, nu = initialize()
    p, u, v, delta_x, delta_y = make_planes(length, width, res)
    delta_t = delta_t_max(delta_x, delta_y, a)
    Re_vec = Re(p, rho, width * 2, v_in, nu)

    print("Re =", np.max(Re_vec), flush=True)

    # Initial Boundary Conditions
    u[0, 1:-1] = 1.0
    u[-1, 1:-1] = 1.0

    # Initialize While loop Variables
    conv = False
    i = 0
    infinitize = []

    # Loop runs until the system converges (continuity equation is satisfied)
    while not conv:
        i += 1
        p_old = p
        p, u, v = take_step(p, u, v, delta_t, Re_vec, delta_y, delta_x, a)
        continuity_factor = np.max(np.abs((p_old - p) / (delta_t * a ** 2)))
        if i % 10000 == 0:
            print("continuity_factor = %1.4f" % continuity_factor, flush=True)
        if continuity_factor < tol:
            """
            When system converges, this updates boundary conditions with
            velocity profile at the center of the pipe and the loop runs
            until the pipe converges again. The tol can be decreased or
            extra infinitizations added for increased precision. I refer
            to the updating of the boundary conditions as 'infinitizing'.
            """
            if len(infinitize) >= 1:
                conv = True
                continue
            u[0, :] = u[len(u[:, 1]) // 2, :]
            u[-1, :] = u[0, :]
            duration = time.time() - start_time
            print(
                "Infinitized at %2.0f seconds," % duration,
                i,
                "iterations.",
                flush=True,
            )
            infinitize.append(i)

    # Report Calculation Statistics
    duration = time.time() - start_time
    print(
        "Scheme took %2.0f seconds and" % duration,
        i,
        "iterations to converge. tol:",
        tol,
    )

    u, v, p, delta_x, delta_y = denormalize(u, v, p, delta_x, delta_y, width, rho, v_in)

    return u, v, p, delta_x, delta_y, width, length, v_in, np.max(Re_vec)


def denormalize(u, v, p, delta_x, delta_y, width, rho, v_in):
    """
    denormalize output
    """
    u = u * v_in
    v = v * v_in
    p = p * rho * v_in ** 2
    delta_x = delta_x * width
    delta_y = delta_y * width
    return u, v, p, delta_x, delta_y


def widening_of_planes(tol=0.001):
    """
    This function takes one attribute, tol, the criteria for convergence. When
    the maximum change in p between consecutive timesteps is less than tol,
    the continuity equation is considered met and the system has converged.
    """

    start_time = time.time()
    # Set up variables
    length, width, res, v_in, a, rho, nu = initialize()
    p, u, v, delta_x, delta_y = make_planes(length, width, res)
    delta_t = delta_t_max(delta_x, delta_y, a)
    Re_vec = Re(p, rho, width * 2, v_in, nu)

    print("Re =", np.max(Re_vec), flush=True)
    print("delta_t =", delta_t, flush=True)

    # Initial Boundary Conditions (Assume flow in and out has stable shape)
    exp = 1.3889  # Expansion Factor
    exp_index = int(len(u[0, :]) * (1 - 1 / exp))
    u[0, exp_index:] = u_theoretical(len(u[0, exp_index:]), v_in)

    u[-1, 1:-1] = u_theoretical(len(u[-1, 1:-1]), v_in / exp)

    # Initialize While loop Variables
    conv = False
    i = 0

    # Loop runs until the system converges (continuity equation is satisfied)
    while not conv:
        i += 1
        p_old = p
        p, u, v = take_step(p, u, v, delta_t, Re_vec, delta_y, delta_x, a)
        continuity_factor = np.max(np.abs((p_old - p) / (delta_t * a ** 2)))
        if i % 1000 == 0:
            print(
                "continuity_factor = %1.4f" % continuity_factor,
                ", i =",
                i,
                flush=True,
            )
        if continuity_factor < tol:
            conv = True

    # Report Calculation Statistics
    duration = time.time() - start_time
    print(
        "\nScheme took %4.0f seconds and" % duration,
        i,
        "iterations to converge. tol:",
        tol,
    )

    u, v, p, delta_x, delta_y = denormalize(u, v, p, delta_x, delta_y, width, rho, v_in)

    return u, v, p, delta_x, delta_y, width, length, np.max(Re_vec)


def plot_results(u, v, p, delta_x, delta_y, width, length):
    """
    Plots u, v, p
    """

    def X(midpoints=False):
        if midpoints:
            n = 0
            offset = delta_x / 2
        else:
            n = 1
            offset = 0
        return np.linspace(offset, length - offset, len(p[:, 0]) + n)

    def Y(midpoints=False):
        if midpoints:
            n = 0
            offset = delta_y / 2
        else:
            n = 1
            offset = 0
        return np.linspace(offset, width - offset, len(p[0, :]) + n)

    # Plot Results
    plt.figure(figsize=(11, 14))
    plt.subplot(3, 1, 1)
    plt.contourf(X(False), Y(True), np.transpose(u[:, 1:-1]), cmap="coolwarm")
    plt.colorbar()
    plt.title("Horizontal Velocity Contours [m/s]")
    plt.ylabel("Location in Vertical [m]")
    plt.subplot(3, 1, 2)
    plt.contourf(X(True), Y(False), np.transpose(v[1:-1, :]), cmap="coolwarm")
    plt.colorbar()
    plt.title("Vertical Velocity Contours [m/s]")
    plt.ylabel("Location in Vertical [m]")
    plt.subplot(3, 1, 3)
    plt.contourf(X(True), Y(True), np.transpose(p) / 1000, cmap="coolwarm")
    plt.colorbar()
    plt.title("Pressure Contours [kPa]")
    plt.ylabel("Location in Vertical [m]")
    plt.xlabel("Location in Horizontal [m]")
    plt.show()


def plot_streamplot(u, v, delta_x, delta_y, width, length, Re):

    u_bar = np.transpose((u[:-1, 1:-1] + u[1:, 1:-1]) * 0.5)
    v_bar = np.transpose((v[1:-1, 1:] + v[1:-1, :-1]) * 0.5)

    offset_x = delta_x / 2
    offset_y = delta_y / 2
    X = np.linspace(offset_x, length - offset_x, len(u_bar[0, :]))
    Y = np.linspace(offset_y, width - offset_y, len(u_bar[:, 0]))
    speed = np.sqrt(u_bar * u_bar + v_bar * v_bar)
    lw = 2 * speed / speed.max()

    # Plot
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    ax.streamplot(X, Y, u_bar, v_bar, density=0.6, color="k", linewidth=lw)
    ax.set_title(
        "Streamplot of Fluid (Line Weight Proportional to Velocity)," " Re = %2.1f" % Re
    )
    ax.set_xlim([0, length])
    ax.set_ylim([0, width])
    ax.set_ylabel("Location in Vertical [m]")
    ax.set_xlabel("Location in Horizontal [m]")
    plt.show()


if __name__ == "__main__":

    case = 2

    print("Case", case)
    if case == 1:
        print("Laminar Flow Between Two Infinate Planes")
        u, v, p, delta_x, delta_y, width, length, v_in, ReN = infinite_planes()
        compare_u_profile(u, delta_y, width, v_in, ReN)
    elif case == 2:
        print("Laminar Flow at Instantaneous Widening of Planes")
        u, v, p, delta_x, delta_y, width, length, ReN = widening_of_planes()
        plot_streamplot(u, v, delta_x, delta_y, width, length, ReN)
    # plot_results(u, v, p, delta_x, delta_y, width, length)
