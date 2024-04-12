# ***************************************
# *                                     *
# *         IDEAL GAS SIMULATION        *
# *                                     *
# ***************************************

# TO DO :
#   -> Calculate pressure with the correct microscopical estimator (virial)

# Ideal gas model :
#   -> Non interacting particles (a = 0)
#   -> No heat exchange with outside (adiabatic system)
#   -> Isotropic velocity distribution
#   -> Elastic collisions with walls (the perpendicular velocity changes sign)
#   -> 2D elastic collisions between the particles

# Units :
#   -> mass = 1
#   -> Boltzmann constant = 1

# Aim :
#   -> Show Maxwell Boltzmann distribution
#   -> Show the validity of PV = NkT from the microscopical point of view (remember k = 1 in our units)

# Importing modules
from utils.functions import *
import numpy as np
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# VARIABLES

# Particles
N_part = 400                                    # Number of particles
x = np.zeros((N_part,2))                        # Positions
v = np.zeros((N_part,2))                        # Velocities
a = np.zeros((N_part,2))                        # Accelerations

# Box
density = 1                                     # Density
V = N_part / density                            # Volume
L = np.sqrt(V)                                  # Side of the square box
R_part = L/N_part                               # Radius of the particles

# Simulation
dt = 0.002                                      # Integration time-step
t_equil = 20000                                 # Number of steps for equilibration
t_meas_init  = 20000                            # Number of steps for initial measure
t_expansion  = 20000                            # Number of steps for expansion
t_meas_final = 20000                            # Number of steps for final measure
t_sample = 15                                   # Time every which I sample
traj = []                                       # Trajectory
bool_wall = []

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# INITIAL CONDITIONS

# Placing particles randomly in the box (no need to place them carefully, no interactions are present)
x = np.random.rand(N_part,2) * (L - 2 * R_part)

# Extract the velocity modulus randomly (uniform distribution) between 0 and v_max, then the direction randomly (uniform distribution) in [0,2*pi).
# NOTE: each time, create a couple of particles with opposite velocities to guarantee isotropy
v_max = 10

for i in range(int(N_part/2)) :
    mod_v = np.random.rand() * v_max
    theta = np.random.rand() * 2 * np.pi
    v[2*i][0] = mod_v * np.cos(theta)
    v[2*i][1] = mod_v * np.sin(theta)
    v[2*i + 1][0] = - v[2*i][0]
    v[2*i + 1][1] = - v[2*i][1]

print("\nParameters:\n")
print(f"\t-> Integration time-step = {dt}")
print(f"\t-> N = {N_part} (number of particles)")
print(f"\t-> L = {L:.2f} (side of the initial cubic box)")
print(f"\t-> v_max = {v_max:.2f} (initial maximum for the modulus of the velocity)")
print(f"\t-> Equilibration steps = {t_equil}")
print(f"\t-> Initial measure steps = {t_meas_init}")
print(f"\t-> Expansion steps = {t_expansion}")
print(f"\t-> Expanded measure steps = {t_meas_final}")
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# THERMALIZATION

print("\nStarting thermalization...")

dp = 0
time = []
maxwell_boltzmann_curve = []

for t in range(t_equil) :

    # Verlet algorithm
    x,v,a = velocity_verlet(x, v, a, dt)

    # Collisions among particles
    x,v = check_collision(x, v, R_part)

    # Check wall collision and, if detected, increase momentum variation
    x,v,dp = check_wall_rebound(x, v, L, L, dp, R_part)

    if t % (2*t_sample) == 0 :
        maxwell_boltzmann_curve.append( np.sqrt(np.square(v).sum(axis=1)).copy() )
        time.append(t)

# Plot velocity curve evolution during the equilibration
plt.ion()
fig, ax = plt.subplots(figsize = (10,10))

plt.title("Square modulus of velocity distribution: the Maxwell-Boltzmann curve")
plt.xlabel("v")
plt.ylabel("Occurrence")

curr = ax.hist(maxwell_boltzmann_curve[0], label=f" Time = {time[0]}")
ax.set_xlim( (0, 2*v_max ) )
ax.set_ylim((0, int(N_part/2)))
ax.legend(loc='upper right', fancybox=True, shadow=True, fontsize=15)

for k in range(1, len(maxwell_boltzmann_curve)) :
    ax.clear()
    curr = ax.hist(maxwell_boltzmann_curve[k], label=f"Time = {time[k]}")
    ax.legend(loc='upper right', fancybox=True, shadow=True, fontsize=15)
    plt.xlabel("v")
    plt.ylabel("Occurrence")
    ax.set_xlim( (0, 2*v_max) )
    ax.set_ylim((0, int(N_part/2)))
    fig.canvas.flush_events()

plt.ioff()
plt.close(fig)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# MEASURE WITH INITIAL VOLUME

print("\nStarting measurement of temperature and pressure before expansion...")

# Measuring temperature (NOTE: I divide by 2 and not by 3 because I am in 2 dimensions!)
T = ( np.square(v).sum() ) / (2 * N_part)
# Variation of linear momentum due to collision with walls
dp = 0

for t in range(t_meas_init) :

    # Verlet algorithm
    x,v,a = velocity_verlet(x, v, a, dt)

    # Collisions among particles
    x,v = check_collision(x, v, R_part)

    # Check wall collision and, if detected, increase momentum variation
    x,v,dp = check_wall_rebound(x, v, L, L, dp, R_part)

    # Save coordinate
    if t % t_sample == 0 :
        traj.append(x.copy())
        bool_wall.append(True)

P_i = ( dp / (t_meas_init * dt) ) / (4*L)        # Pressure = force / "area" (in this case the 1D area is the perimeter)

print(f"\nP V = {P_i * V:.0f}")
print(f"N k T = {N_part * T:.0f}")
rel_err = abs(P_i * V - N_part * T) / (P_i * V) * 100
print(f"Relative error = {rel_err:.1f} %")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPANSION

print("\n\n -> -> -> -> Expansion phase... -> -> -> ->\n")

bool_wall.append(False)
dp = 0

for t in range(t_expansion) :

    # Verlet algorithm
    x,v,a = velocity_verlet(x, v, a, dt)

    # Collisions among particles
    x,v = check_collision(x, v, R_part)

    # Check wall collision and, if detected, increase momentum variation
    x,v,dp = check_wall_rebound(x, v, L, 2*L, dp, R_part)

    # Save coordinate
    if t % t_sample == 0 :
        traj.append(x.copy())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# MEASURE WITH FINAL VOLUME

print("\nStarting measurement of expanded gas properties...")

# Measuring temperature
T = ( np.square(v).sum() ) / (2 * N_part)
# Variation of linear momentum due to collision with walls
dp = 0

for t in range(t_meas_final) :

    # Verlet algorithm
    x,v,a = velocity_verlet(x, v, a, dt)

    # Collisions among particles
    x,v = check_collision(x, v, R_part)

    # Check wall collision and, if detected, increase momentum variation
    x,v,dp = check_wall_rebound(x, v, L, 2*L, dp, R_part)

    # Save coordinate
    if t % t_sample == 0 :
        traj.append(x.copy())

V_f = 2 * np.square(L)
P_f = ( dp / (t_meas_final * dt) ) / (6*L)

print(f"\nP V = {P_f * V_f:.0f}")
print(f"N k T = {N_part * T:.0f}")
rel_err = abs(P_i * V - N_part * T) / (P_i * V) * 100
print(f"Relative error = {rel_err:.1f} %\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPANSION PLOT

# Animation
plt.ion()
fig, ax = plt.subplots(figsize = (10,5))

plt.title("Free adiabatic expansion")
plt.xlabel("x")
plt.ylabel("y")

ax.set_xlim((0, 2 * L))
ax.set_ylim((0, L))
ax.axis('off')

left_wall = ax.axvline(x = 0, ymin = 0, ymax = L)
right_wall = ax.axvline(x = 2*L, ymin = 0, ymax = L)
bottom_wall = ax.axhline(y = 0, xmin = 0, xmax = 2*L)
top_wall = ax.axhline(y = L, xmin = 0, xmax = 2*L)

area = np.square( np.ones(N_part) * 1 )
curr = ax.scatter(traj[0][:,0], traj[0][:,1], s=area, color="blue")
intermediate_wall = ax.axvline(x = L, ymin=0, ymax=L, color="red")

for k in range(1, len(traj)) :
    curr.remove()
    curr = ax.scatter(traj[k][:,0], traj[k][:,1], s=area, color="blue")
    if k < len(bool_wall) and not bool_wall[k] :
        intermediate_wall.remove()
    fig.canvas.flush_events()

plt.ioff()
plt.close(fig)