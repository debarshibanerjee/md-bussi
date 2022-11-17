import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def force_harmonic(k, q):
    ## U = 0.5 k*q^2
    ## F = -dU/dq
    return -(k * q)


@njit
def force_lj(eps, sigma, q):
    ## U = 4*eps*(((sigma/q)**12) - ((sigma/q)**6))
    ## F = -dU/dq
    return -4 * eps * ((-12.0 * ((sigma / q) ** 12) / q) + (6.0 * ((sigma / q) ** 6) / q))


@njit
def verlet_harmonic(del_t, nsteps, k, q_0=2.0, q_prev=2.0, mass=1.0):
    ## limiting value prop to 1/sqrt(2) if we multiply kappa by 2
    ## limiting value prop to sqrt(2) if we multiply mass by 2
    ## this is because frequency of HO = sqrt(mass/kappa)
    ## So the frequency of explosion scales w.r.t. factor of these args
    ## system explodes at dt=2.0 if q0=2.0, m=1.0, k=1.0
    ## system explodes at dt=sqrt(2.0)+0.01 if q0=2.0, m=1.0, k=2.0 (we increase kappa by factor of 2)
    ## kappa = second derivative of pot w.r.t. pos = d2U/dx2
    dt = del_t
    m = mass
    q0 = q_0
    q = np.zeros((nsteps))
    t = np.zeros_like(q)
    traj = np.zeros((nsteps, 2))
    q[0] = q0
    q[-1] = q_prev
    t[0] = 0.0
    for istep in range(nsteps - 1):
        q[istep + 1] = 2 * q[istep] - q[istep - 1] + (((dt ** 2) / m) * force_harmonic(k, q[istep]))
        t[istep + 1] = dt * istep
    traj[:, 0] = t
    traj[:, 1] = q
    return traj


@njit
def verlet_lj(del_t, nsteps, eps=1.0, sigma=1.0, q_0=2.0, q_prev=2.0, mass=1.0):
    ## stable for dt=0.01 (stable upto 0.05)
    ## divergence at 0.06 and beyond...maybe even lesser for very long simulations)
    ## m=1.0, eps=1.0, sigma=1.0, q0=2.0
    ## sort of analytically we are comparing to max timestep we obtained in HO
    ## there max dt scales w.r.t. 1/sqrt(kappa) == 1/sqrt(d2U/dx2)
    ## so, here we can also obtain 1/sqrt(d2U/dx2) at x=x0 as maximum timestep
    ## this is sort of intuitive understanding, NOT really purely deterministic
    dt = del_t
    m = mass
    q0 = q_0
    q = np.zeros((nsteps))
    t = np.zeros_like(q)
    traj = np.zeros((nsteps, 2))
    q[0] = q0
    q[-1] = q_prev
    t[0] = 0.0
    for istep in range(nsteps - 1):
        q[istep + 1] = 2 * q[istep] - q[istep - 1] + (((dt ** 2) / m) * force_lj(eps, sigma, q[istep]))
        t[istep + 1] = dt * istep
    traj[:, 0] = t
    traj[:, 1] = q
    return traj


def plot_traj(data, system):
    # plt.figure(figsize=(11, 5))
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title(f"Trajectory for system: {system}")
    plt.plot(data[:, 0], data[:, 1], "*-", label=f"{system}")
    plt.legend()
    plt.savefig(f"traj_{system}.png")


def main():
    n_args = len(sys.argv)

    if n_args != 4:
        print("Needs 3 args: dt, nsteps, k")
        sys.exit("usage: 1. python3 ex01.py 0.001 10000 1.0")

    del_t = float(sys.argv[1])
    nsteps = int(sys.argv[2])
    k = float(sys.argv[3])

    print("Total time of simulation:", del_t * nsteps)

    q_0 = 2.0
    q_prev = 2.0
    mass = 1.0
    eps = 1.0
    sigma = 1.0

    res_ho_fwd = verlet_harmonic(del_t, nsteps, k, q_0, q_prev, mass)
    output_file_ho_fwd = "verlet_ho_fwd.dat"
    np.savetxt(output_file_ho_fwd, res_ho_fwd)
    plot_traj(res_ho_fwd, "HO_fwd")

    ## check time-reversibility for HO
    res_ho_bwd = verlet_harmonic(del_t, nsteps, k, res_ho_fwd[-1, 1], res_ho_fwd[-2, 1], mass)
    ## reverse the time axis of the trajectory
    res_ho_bwd[:, 0] = np.flip(res_ho_bwd[:, 0], axis=0)
    output_file_ho_bwd = "verlet_ho_bwd.dat"
    np.savetxt(output_file_ho_bwd, res_ho_bwd)
    plot_traj(res_ho_bwd, "HO_bwd")

    res_lj_fwd = verlet_lj(del_t, nsteps, eps, sigma, q_0, q_prev, mass)
    output_file_lj_fwd = "verlet_lj_fwd.dat"
    np.savetxt(output_file_lj_fwd, res_lj_fwd)
    plot_traj(res_lj_fwd, "LJ_fwd")

    ## check time-reversibility for LJ
    res_lj_bwd = verlet_lj(del_t, nsteps, eps, sigma, res_lj_fwd[-1, 1], res_lj_fwd[-2, 1], mass)
    ## reverse the time axis of the trajectory
    res_lj_bwd[:, 0] = np.flip(res_lj_bwd[:, 0], axis=0)
    output_file_lj_bwd = "verlet_lj_bwd.dat"
    np.savetxt(output_file_lj_bwd, res_lj_bwd)
    plot_traj(res_lj_bwd, "LJ_bwd")


## main() ##
if __name__ == "__main__":
    main()
