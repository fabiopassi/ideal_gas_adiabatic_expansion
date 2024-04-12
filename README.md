# ideal_gas_adiabatic_expansion

This script performs the simulation of the adiabatic expansion of an ideal gas using the kinetic theory of gases.

## Physics background

The employed ideal gas model has the following properties:

* Non-interacting particles (a = 0 at every instant)
* No heat exchange (adiabatic system)
* Isotropic velocity distribution
* Elastic collisions with walls (the perpendicular velocity changes sign)
* 2D elastic collisions between the particles
* The particle size is much lower than the average distance between them

The temperature is calculated as:

$$
\frac{2}{2} N k_B T = \sum_{i=1}^N \frac{1}{2} m v_i^2 \longrightarrow k_b T = \sum_{i=1}^N v_i^2 / (2 N) \quad (\text{with } m = 1)
$$

> Note: There is a $\frac{2}{2}$ at the beginning because we have a 2D gas.

The pressure is obtained as the ratio between the average force applied on the contained divided by the "surface" of the box $A$, which in 2D is the perimeter. The average force is in turn calculated by dividing the total linear momentum exchanged with the box by the duration of the measure. In this way we get:

$$
P = \frac{\overline{F}}{A} = \frac{1}{T_{\text{meas}}}\sum_{\text{collisions}} \frac{|\Delta p_{\text{coll}}|}{A} = \frac{1}{T_{\text{meas}} A} \sum_{\text{collisions}} 2 |v^{\perp}_{\text{coll}}| \quad (\text{with } m = 1)
$$

where $|v^{\perp}_{\text{coll}}|$ denotes the absolute value of the particle's velocity component perpendicular to the wall during the collision.

## Aim

This script aims at showing 3 things:

1) Even if we start with a uniform distribution for the modulus of the velocity of the particles, the elastic collisions will lead to a redistribution of the energy and the resulting curve is the Maxwell-Boltzmann distribution.

2) The results from the kinetic theory of gases are in agreement with the macroscopic thermodynamical ones: the famous law PV = NkT is found to be valid also by measuring the temperature via the average value of the kinetic evergy of the particles (equipartition theorem) and the pressure using the collisions of the particles with the container's wall (change of linear momentum due to reflection).

3) After an adiabatic expansion, the temperature (intended as average of the kinetic energy) is unchanged, and so does the product PV.

## Technical details

The scripts are written in python. The only additional packages required to run the simulation are `numpy`, `numba` and `matplotlib`; if you have conda installed, the command:

```bash
conda create -n ideal_gas numba matplotlib
```

should create an environment with all the necessary packages.

After this, you can start the simulations with the commands:

```bash
conda activate ideal_gas
python main.py
```
