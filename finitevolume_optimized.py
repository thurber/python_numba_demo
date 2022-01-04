from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Kelvin Helmholtz Instability
In the compressible Euler equations

"""


@jit(
    nopython=True,
    nogil=True,
    cache=True
)
def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	P_L          is a matrix of left-state  pressure
	P_R          is a matrix of right-state pressure
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
	en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2)

	# compute star (averaged) states
	rho_star  = 0.5 * (rho_L + rho_R)
	momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
	en_star   = 0.5 * (en_L + en_R)
	
	P_star = (gamma - 1) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2 / rho_star + P_star
	flux_Momy   = momx_star * momy_star / rho_star
	flux_Energy = (en_star + P_star) * momx_star / rho_star
	
	# find wavespeeds
	C_L = np.sqrt(gamma * P_L / rho_L) + np.abs(vx_L)
	C_R = np.sqrt(gamma * P_R / rho_R) + np.abs(vx_R)
	C = np.maximum(C_L, C_R)
	
	# add stabilizing diffusive term
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Energy


@jit(
    "float64("
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64[:, :], float64[:, :], float64[:, :], float64[:, :], "
        "float64, float64, float64, float64"
    ")",
    parallel=True,
    nopython=True,
    nogil=True,
    cache=True
)
def update(
    rho, vx, vy, P,
    rho_dx, rho_dy,
    vx_dx, vx_dy,
    vy_dx, vy_dy,
    P_dx, P_dy,
    rho_prime, vx_prime, vy_prime, P_prime,
    rho_XL, rho_XR, rho_YL, rho_YR,
    vx_XL, vx_XR, vx_YL, vx_YR,
    vy_XL, vy_XR, vy_YL, vy_YR,
    P_XL, P_XR, P_YL, P_YR,
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X,
    flux_Mass_Y, flux_Momx_Y, flux_Momy_Y, flux_Energy_Y,
    Mass, Momx, Momy, Energy,
    dx, vol, gamma, courant_fac
):
    
    for i in prange(rho.shape[0]):
        for j in np.arange(rho.shape[1]):

            # get Primitive variables
            rho[i, j] = Mass[i, j] / vol
            vx[i, j]  = Momx[i, j] / rho[i, j] / vol
            vy[i, j]  = Momy[i, j] / rho[i, j] / vol
            P[i, j]   = (Energy[i, j] / vol - 0.5 * rho[i, j] * (vx[i, j]**2. + vy[i, j]**2.)) * (gamma - 1.)

    # get time step (CFL) = dx / max signal speed
    dt = courant_fac * np.min( dx / (np.sqrt( gamma * P / rho ) + np.sqrt(vx**2 + vy**2)) )

    for i in prange(rho.shape[0]):
        for j in np.arange(rho.shape[1]):
            
            i_forward = i + 1
            if i_forward >= rho.shape[0]:
                i_forward = 0
            j_forward = j + 1
            if j_forward >= rho.shape[1]:
                j_forward = 0
            
            # calculate rolling gradients
            rho_dx[i, j] = (rho[i_forward, j] - rho[i - 1, j]) / 2. / dx
            rho_dy[i, j] = (rho[i, j_forward] - rho[i, j - 1]) / 2. / dx
            vx_dx[i, j] = (vx[i_forward, j] - vx[i - 1, j]) / 2. / dx
            vx_dy[i, j] = (vx[i, j_forward] - vx[i, j - 1]) / 2. / dx
            vy_dx[i, j] = (vy[i_forward, j] - vy[i - 1, j]) / 2. / dx
            vy_dy[i, j] = (vy[i, j_forward] - vy[i, j - 1]) / 2. / dx
            P_dx[i, j] = (P[i_forward, j] - P[i - 1, j]) / 2. / dx
            P_dy[i, j] = (P[i, j_forward] - P[i, j - 1]) / 2. / dx

    for i in prange(rho.shape[0]):
        for j in np.arange(rho.shape[1]):

            # extrapolate half-step in time
            rho_prime[i, j] = rho[i, j] - 0.5 * dt * ( vx[i, j] * rho_dx[i, j] + rho[i, j] * vx_dx[i, j] + vy[i, j] * rho_dy[i, j] + rho[i, j] * vy_dy[i, j] )
            vx_prime[i, j] = vx[i, j] - 0.5 * dt * ( vx[i, j] * vx_dx[i, j] + vy[i, j] * vx_dy[i, j] + (1. / rho[i, j]) * P_dx[i, j] )
            vy_prime[i, j] = vy[i, j] - 0.5 * dt * ( vx[i, j] * vy_dx[i, j] + vy[i, j] * vy_dy[i, j] + (1. / rho[i, j]) * P_dy[i, j] )
            P_prime[i, j] = P[i, j] - 0.5 * dt * ( gamma *P[i, j] * (vx_dx[i, j] + vy_dy[i, j])  + vx[i, j] * P_dx[i, j] + vy[i, j] * P_dy[i, j] )
            
            # extrapolate in space to face centers
            rho_XL[i - 1, j] = rho_prime[i, j] - (rho_dx[i, j] * dx / 2.)
            rho_XR[i, j] = rho_prime[i, j] + (rho_dx[i, j] * dx / 2.)
            rho_YL[i, j - 1] = rho_prime[i, j] - (rho_dy[i, j] * dx / 2.)
            rho_YR[i, j] = rho_prime[i, j] + (rho_dy[i, j] * dx / 2.)
            vx_XL[i - 1, j] = vx_prime[i, j] - (vx_dx[i, j] * dx / 2.)
            vx_XR[i, j] = vx_prime[i, j] + (vx_dx[i, j] * dx / 2.)
            vx_YL[i, j - 1] = vx_prime[i, j] - (vx_dy[i, j] * dx / 2.)
            vx_YR[i, j] = vx_prime[i, j] + (vx_dy[i, j] * dx / 2.)
            vy_XL[i - 1, j] = vy_prime[i, j] - (vy_dx[i, j] * dx / 2.)
            vy_XR[i, j] = vy_prime[i, j] + (vy_dx[i, j] * dx / 2.)
            vy_YL[i, j - 1] = vy_prime[i, j] - (vy_dy[i, j] * dx / 2.)
            vy_YR[i, j] = vy_prime[i, j] + (vy_dy[i, j] * dx / 2.)
            P_XL[i - 1, j] = P_prime[i, j] - (P_dx[i, j] * dx / 2.)
            P_XR[i, j] = P_prime[i, j] + (P_dx[i, j] * dx / 2.)
            P_YL[i, j - 1] = P_prime[i, j] - (P_dy[i, j] * dx / 2.)
            P_YR[i, j] = P_prime[i, j] + (P_dy[i, j] * dx / 2.)

    for i in prange(rho.shape[0]):
        for j in np.arange(rho.shape[1]):
            
            # compute fluxes (local Lax-Friedrichs/Rusanov)
            flux_Mass_X[i, j], flux_Momx_X[i, j], flux_Momy_X[i, j], flux_Energy_X[i, j] = getFlux(
                rho_XL[i, j], rho_XR[i, j], vx_XL[i, j], vx_XR[i, j], vy_XL[i, j], vy_XR[i, j], P_XL[i, j], P_XR[i, j], gamma)
            flux_Mass_Y[i, j], flux_Momy_Y[i, j], flux_Momx_Y[i, j], flux_Energy_Y[i, j] = getFlux(
                rho_YL[i, j], rho_YR[i, j], vy_YL[i, j], vy_YR[i, j], vx_YL[i, j], vx_YR[i, j], P_YL[i, j], P_YR[i, j], gamma)

    for i in prange(rho.shape[0]):
        for j in np.arange(rho.shape[1]):
            
            i_forward = i + 1
            if i_forward >= rho.shape[0]:
                i_forward = 0
            j_forward = j + 1
            if j_forward >= rho.shape[1]:
                j_forward = 0
                
            # update solution
            Mass[i, j] = Mass[i, j] + dt * dx * (-flux_Mass_X[i, j] + flux_Mass_X[i - 1, j] - flux_Mass_Y[i, j] + flux_Mass_Y[i, j - 1])
            Momx[i, j] = Momx[i, j] + dt * dx * (-flux_Momx_X[i, j] + flux_Momx_X[i - 1, j] - flux_Momx_Y[i, j] + flux_Momx_Y[i, j - 1])
            Momy[i, j] = Momy[i, j] + dt * dx * (-flux_Momy_X[i, j] + flux_Momy_X[i - 1, j] - flux_Momy_Y[i, j] + flux_Momy_Y[i, j - 1])
            Energy[i, j] = Energy[i, j] + dt * dx * (-flux_Energy_X[i, j] + flux_Energy_X[i - 1, j] - flux_Energy_Y[i, j] + flux_Energy_Y[i, j - 1])

    return dt


def main(resolution=128):
    """ Finite Volume simulation """

    # Simulation parameters
    N                      = resolution # resolution
    boxsize                = 1.
    gamma                  = 5/3 # ideal gas gamma
    courant_fac            = 0.4
    t                      = 0
    tEnd                   = 2
    tOut                   = 0.02 # draw frequency
    tLastPlot              = 0.0

    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5*dx, boxsize-0.5*dx, N)
    Y, X = np.meshgrid( xlin, xlin )

    # Generate Initial Conditions - opposite moving streams with perturbation
    w0 = 0.1
    sigma = 0.05/np.sqrt(2.)
    rho = 1. + (np.abs(Y-0.5) < 0.25)
    vx = -0.5 + (np.abs(Y-0.5)<0.25)
    vy = w0*np.sin(4*np.pi*X) * ( np.exp(-(Y-0.25)**2/(2 * sigma**2)) + np.exp(-(Y-0.75)**2/(2*sigma**2)) )
    P = 2.5 * np.ones(X.shape)

    # Get conserved variables
    Mass   = rho * vol
    Momx   = rho * vx * vol
    Momy   = rho * vy * vol
    Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol

    # prep figure
    plt.rcParams['figure.figsize'] = [10, 10]
    
    # initialize intermediate matrices
    rho_dx = np.empty_like(rho)
    rho_dy = np.empty_like(rho)
    vx_dx = np.empty_like(rho)
    vx_dy = np.empty_like(rho)
    vy_dx = np.empty_like(rho)
    vy_dy = np.empty_like(rho)
    P_dx = np.empty_like(rho)
    P_dy = np.empty_like(rho)
    rho_prime = np.empty_like(rho)
    vx_prime = np.empty_like(rho)
    vy_prime = np.empty_like(rho)
    P_prime = np.empty_like(rho)
    rho_XL = np.empty_like(rho)
    rho_XR = np.empty_like(rho)
    rho_YL = np.empty_like(rho)
    rho_YR = np.empty_like(rho)
    vx_XL = np.empty_like(rho)
    vx_XR = np.empty_like(rho)
    vx_YL = np.empty_like(rho)
    vx_YR = np.empty_like(rho)
    vy_XL = np.empty_like(rho)
    vy_XR = np.empty_like(rho)
    vy_YL = np.empty_like(rho)
    vy_YR = np.empty_like(rho)
    P_XL = np.empty_like(rho)
    P_XR = np.empty_like(rho)
    P_YL = np.empty_like(rho)
    P_YR = np.empty_like(rho)
    flux_Mass_X = np.empty_like(rho)
    flux_Momx_X = np.empty_like(rho)
    flux_Momy_X = np.empty_like(rho)
    flux_Energy_X = np.empty_like(rho)
    flux_Mass_Y = np.empty_like(rho)
    flux_Momx_Y = np.empty_like(rho)
    flux_Momy_Y = np.empty_like(rho)
    flux_Energy_Y = np.empty_like(rho)

    # Simulation Main Loop
    while t < tEnd:

        t += update(
            rho, vx, vy, P,
            rho_dx, rho_dy,
            vx_dx, vx_dy,
            vy_dx, vy_dy,
            P_dx, P_dy,
            rho_prime, vx_prime, vy_prime, P_prime,
            rho_XL, rho_XR, rho_YL, rho_YR,
            vx_XL, vx_XR, vx_YL, vx_YR,
            vy_XL, vy_XR, vy_YL, vy_YR,
            P_XL, P_XR, P_YL, P_YR,
            flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X,
            flux_Mass_Y, flux_Momx_Y, flux_Momy_Y, flux_Energy_Y,
            Mass, Momx, Momy, Energy,
            dx, vol, gamma, courant_fac
        )

        if ((t - tLastPlot) >= tOut) or (t >= tEnd):
            tLastPlot = t
            # plot in real time - color 1/2 particles blue, other half red
            display.clear_output(wait=True)
            plt.cla()
            plt.imshow(rho.T)
            plt.clim(0.8, 2.2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)


    plt.show()


if __name__== "__main__":
  main()

