""" This module contains all the functions for the adiabatic expansion script """

# Importing modules
import numpy as np
from numba import njit, prange

# Functions

@njit
def velocity_verlet(x, v, a, dt) :

    v_dt_2 = np.zeros(v.shape)

    v_dt_2 = v + 0.5 * a * dt                       # Update velocities pt.1
    x = x + v_dt_2 * dt                             # Update position
    v = v_dt_2 + 0.5 * a * dt                       # Update velocities pt.2

    return x, v, a



@njit (parallel=True)
def check_collision(r, v, R) :

    for i in prange(r.shape[0]) :
        for j in range(i+1, r.shape[0]) :
            # Check for collision
            dr = r[i,:] - r[j,:]
            # Sort of sweep and prune algorithm in both dimensions
            if abs(dr[1]) < 2*R and dr[2] < 2*R :
                # Check if distance between centers is lower than sum of radii
                dist = np.sqrt( np.square(dr).sum() )
                if dist < 2 * R :
                    # Move the particle i so that they are in contact only externally: this prevents anomalous "bound states"
                    versor = dr / dist
                    r[i] = r[i] + versor * (2*R - dist)
                    # Store old velocities
                    v_i_old = v[i,:].copy()
                    v_j_old = v[j,:].copy()
                    # Change velocity according to conservation laws of elastic collision
                    v[i] = v_i_old - ( ((v_i_old-v_j_old) * (r[i]-r[j])).sum() ) / (np.square(r[i]-r[j]).sum()) * (r[i]-r[j])
                    v[j] = v_j_old - ( ((v_j_old-v_i_old) * (r[j]-r[i])).sum() ) / (np.square(r[i]-r[j]).sum()) * (r[j]-r[i])
    
    return r, v



@njit
def check_wall_rebound(r, v, upper_wall, right_wall, dp, R) :

    for i in range(r.shape[0]) :

        # Left wall (it is located in x=0)
        if r[i,0] < R :
            # Take into account wall penetration
            penetration_length = R - r[i,0]
            r[i,0] += 2 * penetration_length
            v[i,0] *= -1                                        # Invert motion
            dp += 2 * np.abs(v[i][0])                           # Transferred momentum

        # Right wall
        if (right_wall - r[i,0]) < R :
            # Take into account wall penetration
            penetration_length = r[i,0] - right_wall + R
            r[i,0] -= 2 * penetration_length
            v[i,0] *= -1                                        # Invert motion
            dp += 2 * np.abs(v[i][0])                           # Transferred momentum
        
        # Bottom wall (it is located in y=0)
        if r[i][1] < R :
            # Take into account wall penetration
            penetration_length = R - r[i,1]
            r[i,1] += 2 * penetration_length
            v[i,1] *= -1                                        # Invert motion
            dp += 2 * np.abs(v[i][1])                           # Transferred momentum
        
        # Upper wall
        if (upper_wall - r[i,1]) < R :
            # Take into account wall penetration
            penetration_length = r[i,1] - upper_wall + R
            r[i,1] -= 2 * penetration_length
            v[i,1] *= -1                                        # Invert motion
            dp += 2 * np.abs(v[i][1])                           # Transferred momentum

    return r, v, dp
