import torch
import sympy
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sympy import Symbol
from functools import partial
from sympy.solvers import solve
from scipy.signal import savgol_filter
from sympy.utilities.lambdify import lambdify

# Make exact lambda solution function
## Define symbols
fI = Symbol('f_{I}(t)')

alpha = Symbol('alpha')
eta = Symbol('eta')
gamma = Symbol('gamma')
theta = Symbol('theta')
beta = Symbol('beta')

alphaV = Symbol('alpha_V')
etaV = Symbol('eta_V')
gammaV = Symbol('gamma_V')
thetaV = Symbol('theta_V')
betaV = Symbol('beta_V')
xi = Symbol("xi")

Ct_DI_2 = Symbol('C_{t-D_I-2}')
Et_DI_2 = Symbol('E_{t-D_I-2}')
It_DI_2 = Symbol('I_{t-D_I-2}')
St_DI_2 = Symbol('S_{t-D_I-2}')

CVt_DI_2 = Symbol('C_{V, t-D_I-2}')
EVt_DI_2 = Symbol('E_{V, t-D_I-2}')
IVt_DI_2 = Symbol('I_{V, t-D_I-2}')
SVt_DI_2 = Symbol('S_{V, t-D_I-2}')

It_obs = Symbol('I_{t}^{obs}')

lambda_t = Symbol('lambda(t-D_I-1)')
N_symb = Symbol('N')

sympy_symbols = [
    fI,
    alpha, eta, gamma, theta, beta,
    alphaV, etaV, gammaV, thetaV, betaV, xi,
    Ct_DI_2, Et_DI_2, It_DI_2, St_DI_2,
    CVt_DI_2, EVt_DI_2, IVt_DI_2, SVt_DI_2,
    It_obs, N_symb
]

## Define equation to solve
rate =  ((Ct_DI_2 + CVt_DI_2 + beta  * (It_DI_2 + IVt_DI_2)) / N_symb)
rateV = ((Ct_DI_2 + CVt_DI_2 + betaV * (It_DI_2 + IVt_DI_2)) / N_symb)
EVt_DI_1 = EVt_DI_2 + lambda_t * xi * rateV * SVt_DI_2 - gammaV * EVt_DI_2
Et_DI_1  = Et_DI_2  + lambda_t      * rate  * St_DI_2  - gamma  * Et_DI_2
CVt_DI_1 = CVt_DI_2 + gammaV * EVt_DI_2 - (1-alphaV) * etaV * CVt_DI_2 - alphaV * thetaV * CVt_DI_2
Ct_DI_1  = Ct_DI_2  + gamma  * Et_DI_2  - (1-alpha)  * eta  * Ct_DI_2  - alpha  * theta  * Ct_DI_2
CVt_DI = CVt_DI_1 + gammaV * EVt_DI_1 - (1-alphaV) * etaV * CVt_DI_1 - alphaV * thetaV * CVt_DI_1
Ct_DI  = Ct_DI_1  + gamma  * Et_DI_1  - (1-alpha)  * eta  * Ct_DI_1  - alpha  * theta  * Ct_DI_1
f = (1-fI) * ((1-alpha) * eta * Ct_DI + (1-alphaV) * etaV * CVt_DI) - It_obs

## Solve for lambda(t)
lambda_sol = solve(f, lambda_t)[0]
lambda_sol_callable = lambdify(sympy_symbols, lambda_sol, modules='numpy')


# Disease Model

def prior():
    alpha_f = (0.7**2) * ((1-0.7)/(0.17**2) - (1-0.7))
    beta_f = alpha_f * (1/0.7 - 1)
    
    Ds = np.random.lognormal(mean=np.log(8), sigma=0.2, size=(3,))
    Ds = np.round(Ds).astype(np.int64)
    return {
        #
        # Disease Model prior
        #
        'E0': np.random.gamma(shape=2, scale=30),  # E0
        # Non-vaccinated parameters
        'beta': np.random.lognormal(mean=np.log(0.25), sigma=0.3),  # beta
        'gamma': np.random.lognormal(mean=np.log(1/6.5), sigma=0.5),  # gamma
        'alpha': np.random.uniform(low=0.005, high=0.9),  # alpha
        'eta': np.random.lognormal(mean=np.log(1/3.2), sigma=0.3),  # eta
        'theta': np.random.uniform(low=1/14, high=1/3),  # theta
        'delta': np.random.uniform(low=0.00, high=0.3),  # delta
        'mu': np.random.lognormal(mean=np.log(1/8), sigma=0.2),  # mu
        'd': np.random.uniform(low=1/14, high=1/3),  # d
        # Vaccinated parameters
        'betaV': np.random.lognormal(mean=np.log(0.25), sigma=0.3),  # beta
        'gammaV': np.random.lognormal(mean=np.log(1/6.5), sigma=0.5),  # gamma
        'alphaV': np.random.uniform(low=0.005, high=0.9),  # alpha
        'etaV': np.random.lognormal(mean=np.log(1/3.2), sigma=0.3),  # eta
        'thetaV': np.random.uniform(low=1/14, high=1/3),  # theta
        'deltaV': np.random.uniform(low=0.00, high=0.3),  # delta
        'muV': np.random.lognormal(mean=np.log(1/8), sigma=0.2),  # mu
        'dV': np.random.uniform(low=1/14, high=1/3),  # d
        #'nu': np.random.uniform(0.0, 0.01),
        'xi': np.random.exponential(1),
        #'vacc_start': np.random.normal(260, 10),
        #'schwurbelrate': np.random.uniform(0.0, 0.5), # perc. of people reject vaccination
        #
        # Observation Model prior
        #
        'As': np.random.beta(a=alpha_f, b=beta_f, size=(3,)),  # As
        'Phis': stats.vonmises(kappa=0.01).rvs(size=(3,)),  # Phis
        'Ds': Ds,  # Ds
        'sigma': np.random.gamma(shape=1, scale=5, size=(3,)), # sigmas
        'lambda_start': np.random.lognormal(mean=np.log(3), sigma=0.3)
    }


def derivative(
        t, y, lambda_t,
        N, beta, gamma, alpha, eta, theta, delta, mu, d,
        betaV, gammaV, alphaV, etaV, thetaV, deltaV, muV, dV,
        xi, nu_t, **params):
    """
    Returns a callable that calculates the derivatives of the disease model
    ODE as specified in Radev et al. for given compartment vector.
    """
    S, SV, E, EV, C, CV, I, IV, R, D = y
        
    transmission_rate = lambda_t * ((C + CV + beta * (I + IV)) / N)
    transmission_rateV = lambda_t * xi * ((C + CV + betaV * (I + IV)) / N)
    
    nu_t = min(S - transmission_rate * S, nu_t)
        
    derivatives = [
        # dS / dt
        -transmission_rate * S - nu_t,
        # dSV / dt
        -transmission_rateV * SV + nu_t,
        # dE / dt
        transmission_rate * S - gamma * E,
        # dEV / dt
        transmission_rateV * SV - gammaV * EV,
        # dC / dt
        gamma * E - (1 - alpha) * eta * C - alpha * theta * C,
        # dCV / dt
        gammaV * EV - (1 - alphaV) * etaV * CV - alphaV * thetaV * CV,
        # dI / dt
        (1 - alpha) * eta * C - (1 - delta) * mu * I - delta * d * I,
        # dIV / dt
        (1 - alphaV) * etaV * CV - (1 - deltaV) * muV * IV - deltaV * dV * IV,
        # dR / dt
        alpha * theta * C + alphaV * thetaV * CV + (1 - delta) * mu * I + (1 - deltaV) * muV * IV,
        # dD / dt
        delta * d * I + deltaV * dV * IV
    ]
        
    return np.array(derivatives)


def observation_model(ys, T, sim_lag,
                      As, Phis, Ds, sigma,
                      alpha, eta, delta, mu, d,
                      alphaV, etaV, deltaV, muV, dV,
                      **params):
    # Use daily new I/R/D instead of cumulative numbers.
    # ys = np.diff(ys, axis=0, prepend=0)
    
    sigma = np.asarray(sigma)
    y_obs = np.zeros((T, 3))
    
    # Rates at which I/R/D are observable
    obs_rates = [
        ((1-alpha) * eta, (1-alphaV) * etaV),
        ((1-delta) * mu, (1-deltaV) * muV),
        (delta * d, deltaV * deltaV),
    ]

    for k, (A, Phi, D, (obs_rate, obs_rateV)) in enumerate(zip(As, Phis, Ds, obs_rates)):
        # Reported time steps (taking reporting delay into account)
        ts = np.arange(sim_lag, T + sim_lag)
        # Weekly modulation factors
        f_k = (1-A) * (1 - np.abs(np.sin(np.pi/7 * (ts-sim_lag) - 0.5*Phi)))
        # Calculate observed I/R/D values (daily new, not cumulative)
        y_obs[:, k] = (1 - f_k) * (obs_rate * ys[k, ts-D, 0] + obs_rateV * ys[k, ts-D, 1])
    
    # Add noise from t-distribution
    noise_scale = np.sqrt(np.clip(y_obs, 0.0, np.inf)) * sigma
    y_obs = stats.t(df=4, loc=y_obs, scale=noise_scale).rvs()

    return y_obs


#Simulation

def sample_simulation(N: int, data, T: int = None, params=None, sim_lag=16):
    """
    Runs simulation with given parameters or samples random
    parameters from prior.
    
    Returns SECIRD simulation data, observation data according to obeservation
    model and parameters.
    
    :param T:                 Total number of timesteps to simulate (observed)
    :param N:                 Total size of population
    :param data:              Data to calculate lambda(t) for
    :param params:            Parameters for disease model
    """
    # If no parameters given, sample from prior
    if params is None:
        params = prior()
    
    # If T not given, assume length of data
    if T is None:
        T = data.shape[0]
    
    # ODE derivatives
    ode_system = partial(derivative, N=N, **params)
    # Initialisation of compartments
    y0 = [N-params['E0'], 0, params['E0'], 0] + [0] * 6
    y0 = np.array(y0)
    
    # Smooth data by applying Savitzky-Golay filter
    # -> needed for making lambda(t) independent of
    #    weekly modulations
    savgol_data = savgol_filter(data[:, 0], 71, 9)
    savgol_data = savgol_filter(savgol_data, 51, 5)
    
    # Ensure strictly positive number of cases
    # 1) Negative is unrealistic (occurs only as consequence of corrections to data)
    # 2) > 0 because for simulation, we don't want the disease to
    #    vanish :(
    savgol_data = np.clip(savgol_data.reshape(-1, 1), 1.0, np.inf)

    secird = [y0]
    lambdas = np.full(T + sim_lag, params['lambda_start'])
    D_I = params['Ds'][0]
    
    for t in range(0, T + sim_lag):
        t_real = t - sim_lag + D_I + 1  # Time for which we can calculate lambda value
        y = secird[-1]
        S, SV, E, EV, C, CV, I, IV, R, D = y
        
        if t == 0:
            lambda_t = params['lambda_start']
        
        elif 0 <= t_real < data.shape[0]:
            I_obs = savgol_data[t_real, 0]
            
            # Values for calculating lambda(t_real)
            substitutions = [
                #(1-params['As'][0]) * (1 - np.abs(np.sin(np.pi/7 * (t_real) - 0.5* params['Phis'][0]))),
                0.36 * (1-params['As'][0]),  # Dummy for weekly modulation model which we ignore
                # Optimal (?) constant 0.36 found by manual tuning
                params['alpha'], params['eta'], params['gamma'], params['theta'], params['beta'],
                params['alphaV'], params['etaV'], params['gammaV'], params['thetaV'], params['betaV'], params['xi'],
                C, E, I, S, CV, EV, IV, SV, I_obs, N
                ]
            
            # Calculate lambda(t_real)
            try:
                lambda_t = float(lambda_sol_callable(*substitutions))
                # Restrict lambda(t_real) to realistic values
                # Note that in the initial phase, lambda(t) will take extreme values
                # in order to adjust the simulation to data
                lambda_t = np.clip(lambda_t, 0.0, 25.0)
                
            except FloatingPointError:
                assert t > 0, "Time is {}".format(t)
                lambda_t = lambdas[t-1]
        
        elif t_real > data.shape[0]:
            # For future predictions, use mean of last lambda(t)
            lambda_t = np.mean(lambdas[t-7:t])
        
        lambdas[t] = lambda_t
        nu_t = data[max(0, t - sim_lag), 3]
        # Euler Method with step size 1
        # Simple but works (see Radev et al. (2020)) and 4x faster than LSODA
        secird.append(y + ode_system(t, y, lambda_t, nu_t = nu_t))
        
    secird = np.stack(secird)

    # Observation model
    # Calculate observation from C + I
    #vaccinated = np.diff(secird[:, [1, 1]], axis=0, prepend=0)
    carrier = secird[:, [4, 5]]
    infected = secird[:, [6, 7]]
    
    observed_compartments = np.stack([carrier, infected, infected])
    obs = observation_model(observed_compartments, T, sim_lag, **params)

    return {
        'N': N,
        'T': T,
        'params': params,
        'lambda': lambdas,
        'true_data': secird,
        'observed_data': obs
    }


def params2vector(params):
    """
    Converts parameter dict to numpy vector.
    """
    parameter_names = [
        'E0', 'beta', 'gamma', 'alpha', 'eta', 'theta', 'delta', 'mu', 'd',
        'betaV', 'gammaV', 'alphaV', 'etaV', 'thetaV', 'deltaV', 'muV', 'dV', 'xi',
        'As', 'Phis', 'Ds', 'sigma', 'lambda_start'
    ]
    vector = []

    for param_name in parameter_names:

        param = params[param_name]
        if isinstance(param, np.ndarray):
            vector.extend(param.tolist())
        else:
            vector.append(param)

    return np.array(vector)


def data_generator(batch_size, data,
                   T=None, N=None,
                   weekly = True,
                   T_min=10, T_max=104,  # 104 weeks = 2 years
                   N_min=100000, N_max=2e9,  # India -> need more population
                   mean_g=None, std_g=None,
                   to_tensor=True, seed=None,
                   observ_model=True):
    np.seterr(all='raise')
    if seed is not None:
        np.random.seed(seed)

    # Variable-size t
    if T is None:
        T = np.random.randint(T_min, T_max + 1)
        
    if weekly:
        T *= 7

    # Variable size N
    if N is None:
        N = np.random.randint(N_min, N_max)

    x = []
    thetas = []

    while len(thetas) < batch_size:
        try:
            simulation = sample_simulation(N, data, T)
        except (ValueError, RuntimeWarning, FloatingPointError):
            continue

        theta_i = params2vector(simulation['params'])

        if observ_model:
            obs = simulation['observed_data']
        else:
            obs = simulation['secird'][:, [6, 8, 9, 1]]
            obs[:, 1:] = np.diff(obs[:, 1:], axis=0, prepend=0)
        
        if mean_g is not None and std_g is not None:
            obs = (obs - mean_g) / std_g

        if weekly:
            infected = obs[:, 0:1].reshape(obs.shape[0] // 7, 7)
            recovered = obs[:, 1:2].reshape(obs.shape[0] // 7, 7)
            dead = obs[:, 2:].reshape(obs.shape[0] // 7, 7)
            #vaccinated = obs[:, 3:].reshape(obs.shape[0] // 7, 7)
            obs = np.concatenate([infected, recovered, dead], axis=-1)
            
        thetas.append(theta_i)
        x.append(obs)

    x = np.stack(x)
    thetas = np.stack(thetas)

    # Convert to tensor, if specified
    if to_tensor:
        thetas = torch.from_numpy(thetas).float()
        x = torch.from_numpy(x).float()

    return {'theta': thetas, 'x': x}
