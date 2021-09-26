import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.integrate import solve_ivp
from functools import partial

#Intervention Model


def make_lambda(T, dates, values, delta: int = 3, **params):
    """
    Returns lambda(t) time dependent transmission rates.
    lambda(t) performs table lookup on precomputed values.
    
    :param T:      Total number of timesteps
    :param dates:  Intervention dates (lambda(t) changes)
    :param values: lambda(t) for intervals
    """
    starts = np.concatenate([[0], dates])
    ends = np.concatenate([dates, [T]])
    prev_values = np.concatenate([[values[0]], values[:-1]])
    
    assert len(starts) == len(ends) == len(values) == len(prev_values)
    assert list(dates) == list(sorted(dates))
    
    # Precompute lambda(t)
    lambdas = np.zeros(T)
    for start, end, value, prev_value in zip(starts, ends, values, prev_values):
        # lambda(t) is `value` for t in [start, end] and 0.0 otherwise.
        # Has warm-up period of `delta` timesteps with linear
        # interpolation between `prev_value` and `value`.
        for t in range(start, end):
            discount = 1 if delta == 0 else min((t-start) / delta, 1)
            lambdas[t] = value * discount + (1-discount) * prev_value
    
    return (lambda t: lambdas[int(t)] if t < T else values[-1])


#Disease Model

def make_derivative(N, lambd, beta, gamma, alpha, eta, theta, delta, mu, d,
                    **params):
    """
    Returns a callable that calculates the derivatives of the disease model
    ODE as specified in Radev et al. for given compartment vector.
    
    :param N: Total population size
    :param lambd: Callable where lambd(t) is transmission rate at time t
    :param beta:  Risk of infection from symptomatic patients
    :param gamma: Rate at which exposed cases become infectious
    :param alpha: Probability of remaining undetected/undiagnosed
    :param eta:   Rate at which symptoms manifest
    :param theta: Rate at which undiagnosed individuals recover
    :param delta: Probability of dying from the disease
    :param mu:    Rate at which symptomatic individuals recover
    :param d:     Rate at which critical cases die
    """
    return lambda t, y: np.array(
        [
            # dS / dt
            -lambd(t) * ((y[2] + beta * y[3]) / N) * y[0],
            # dE / dt
            lambd(t) * ((y[2] + beta * y[3]) / N) * y[0] - gamma * y[1],
            # dC / dt
            gamma * y[1] - (1 - alpha) * eta * y[2] - alpha * theta * y[2],
            # dI / dt
            (1 - alpha) * eta * y[2] - (1 - delta) * mu * y[3] - delta * d * y[3],
            # dR / dt
            alpha * theta * y[2] + (1 - delta) * mu * y[3],
            # dD / dt
            delta * d * y[3]
        ])


#Observation Model

def observation_model(ys, T, sim_lag, As, Phis, alpha, eta, Ds, sigma, delta, mu, d,
                      **params):
    """
    Transforms true simulated infected / recovered / dead numbers to observed
    numbers according to the observation model of Radev et al.
    
    Uses daily new I/R/D instead of cumulative numbers.
    
    :param ys:    Simulated I/R/D (cumulative)
    :param T:     Total number of timesteps 
                  (because of report delay, simulation is actually longer)
    :param As:    Discount factors for weekly modulation
    :param Phis:  Time-shift constants for weekly modulation
    :param alpha: Probability of remaining undetected/undiagnosed
    :param eta:   Rate at which symptoms manifest
    :param Ds:    Report delays (number of days)
    :param sigma: Noise scale factors
    :param delta: Probability of dying from the disease
    :param mu:    Rate at which symptomatic individuals recover
    :param d:     Rate at which critical cases die
    """
    # Use daily new I/R/D instead of cumulative numbers.
    # ys = np.diff(ys, axis=0, prepend=0)
    
    sigma = np.asarray(sigma)
    y_obs = np.zeros((T, 3))
    
    # Rates at which I/R/D are observable
    obs_rates = [(1-alpha) * eta, (1-delta) * mu, delta * d]

    for k, (A, Phi, D, obs_rate) in enumerate(zip(As, Phis, Ds, obs_rates)):
        # Reported time steps (taking reporting delay into account)
        ts = np.arange(sim_lag, T + sim_lag)
        # Weekly modulation factors
        f_k = (1-A) * (1 - np.abs(np.sin(np.pi/7 * (ts-sim_lag) - 0.5*Phi)))
        # Calculate observed I/R/D values (daily new, not cumulative)
        y_obs[:, k] = (1 - f_k) * obs_rate * ys[ts-D, k]
    
    # Add noise from t-distribution
    noise_scale = np.sqrt(np.clip(y_obs, 0.0, np.inf)) * sigma
    y_obs = stats.t(df=4, loc=y_obs, scale=noise_scale).rvs()

    return y_obs


#Prior

def prior(T: int, num_interventions: int = 15):
    """
    Returns values for parameters of disease model sampled from
    prior distributions based off Radev et al.
    
    Note that we discard delta = number of days for transmission to
    have full effect from parameters, because the effect is negligible
    in long term data.
    
    :param T:                 Total number of timesteps
    :param num_interventions: Number of interventions 
                              (= times transmission rate lambda(t) changes)
    """
    alpha_f = (0.7**2) * ((1-0.7)/(0.17**2) - (1-0.7))
    beta_f = alpha_f * (1/0.7 - 1)
    
    # Bounds for transmission rates
    value_bounds_low = [1.] + [0.05] * num_interventions
    value_bounds_high = [5.] + [0.8] * num_interventions
    
    #date_means = np.array(
    #    [15,  24,  34,  54,  57,  68, 144, 185, 200, 231, 304, 331, 338,
    #     370, 409, 440, 455, 492, 521, 533]
    #    )
    #dates = np.random.normal(loc=date_means)
    #dates = np.round(np.clip(dates, 0.0, np.inf)).astype(np.int64)
    #dates = np.sort(dates)
    
    #value_means = np.array(
    #    [8.8936895 , 0.71825488, 0.71736783, 0.76635909, 0.12807158,
    #     0.47248307, 0.2577032 , 0.0533663 , 0.37104318, 0.19991469,
    #     0.75851677, 0.19532738, 0.21393728, 0.44462668, 0.27876473,
    #     0.73409626, 0.60745195, 0.24960652, 0.25766388, 0.69769406,
    #     0.49236346]
    #    )
    
    #values = np.random.normal(loc=value_means, scale=value_means / 5)
    #values = np.clip(values, 0.0, np.inf)
    
    # Report delays
    Ds = np.random.lognormal(mean=np.log(8), sigma=0.2, size=(3,))
    Ds = np.round(Ds).astype(np.int64)
    
    # Intervention dates
    # Instead of using fixed intervention date means, we just sample
    # random dates. This is necessary, because we want to evaluate the
    # model for different countries.
    dates = np.random.choice(
        np.arange(0, T),
        replace=False,
        size=(num_interventions,)
        )
    # Sort intervention dates because sampling can't enforce this
    dates = np.sort(dates)
    
    # Time dependent transmission rates
    # Instead of using custom priors for each intervention date
    # we sample from the same distribution for each intervention
    transmission_rates = np.random.uniform(
        low=value_bounds_low,
        high=value_bounds_high,
        size=(num_interventions+1,)
        )
    
    return {
        #
        # Disease Model prior
        #
        'E0': np.random.gamma(shape=2, scale=30),  # E0
        'beta': np.random.lognormal(mean=np.log(0.25), sigma=0.3),  # beta
        'gamma': np.random.lognormal(mean=np.log(1/6.5), sigma=0.5),  # gamma
        'alpha':  np.random.uniform(low=0.005, high=0.9),  # alpha
        'eta': np.random.lognormal(mean=np.log(1/3.2), sigma=0.3),  # eta
        'theta': np.random.uniform(low=1/14, high=1/3),  # theta
        'delta': np.random.uniform(low=0.01, high=0.3),  # delta
        'mu': np.random.lognormal(mean=np.log(1/8), sigma=0.2),  # mu
        'd': np.random.uniform(low=1/14, high=1/3),  # d
        #
        # Observation Model prior
        #
        'As': np.random.beta(a=alpha_f, b=beta_f, size=(3,)),  # As
        'Phis': stats.vonmises(kappa=0.01).rvs(size=(3,)),  # Phis
        'Ds': Ds,  # Ds
        'sigma': np.random.gamma(shape=1, scale=5, size=(3,)), # sigmas
        #
        # Intervention Model prior
        #
        'dates': dates,
        'values': transmission_rates,
    }


def params2vector(params):
    """
    Converts parameter dict to numpy vector.
    """
    parameter_names = [
        'E0', 'beta', 'gamma', 'alpha', 'eta', 'theta', 'delta', 'mu', 'd',
        'As', 'Phis', 'Ds', 'sigma', 'dates', 'values'
        ]
    vector = []
    
    for param_name in parameter_names:

        param = params[param_name]
        if isinstance(param, np.ndarray):
            vector.extend(param.tolist())
        else:
            vector.append(param)

    return np.array(vector)


#Simulation

def sample_simulation(T: int, N: int, params=None, exact=False,
                      num_interventions: int = 15, sim_lag=16):
    """
    Runs simulation with given parameters or samples random
    parameters from prior.
    
    Returns SECIRD simulation data, observation data according to obeservation
    model and parameters.
    
    :param T:                 Total number of timesteps to simulate (observed)
    :param N:                 Total size of population
    :param params:            Parameters for disease model
    :param exact:             If True, uses exact solver for disease model ODE
                              (not recommended)
    :param num_interventions: Number of interventions
                              (= times transmission rate lambda(t) changes)
    """
    # If no parameters given, sample from prior
    if params is None:
        params = prior(T, num_interventions=num_interventions)
    
    # Time dependent transmission rate lambda(t)
    lambd = make_lambda(T + sim_lag, **params)
    # ODE derivatives
    ode_system = make_derivative(N, lambd, **params)
    # Initialisation of compartments
    y0 = np.array([N - params['E0'], params['E0'], 0, 0, 0, 0])

    # The exact model uses an ODE solver from
    # scipy to solve the ODE.
    # Note that this is slover and if the ODE is,
    # because of bad parameterisation, not well behaved,
    # the solver might diverge/crash/do stupid things
    # We therefore recommend the not exact solution
    if exact:
        secird = solve_ivp(
            ode_system, (0.0, T + sim_lag),
            y0,
            t_eval=np.arange(T + sim_lag),
            method='LSODA'
            )
        secird = secird['y'].T

    # Euler Method with step size 1
    # Simple but works (see Radev et al. (2020)) and 4x faster than LSODA
    else:
        secird = [y0]
        for t in range(0, T + sim_lag - 1):
            secird.append(secird[-1] + ode_system(t, secird[-1]))
        secird = np.stack(secird)

    # Observation model
    # Calculate observation from C + I
    # According to Radev et al. Eq. (8) - (10)
    obs = observation_model(secird[:, [2, 3, 3]], T, sim_lag, **params)

    return {
        'N': N,
        'T': T,
        'params': params,
        'true_data': secird,
        'observed_data': obs
    }


def normalise(obs, N, weekly=True, raw_values=False):
    """
    obs has shape (time, [I, R, D])
    """
    # If we want to use relative numbers
    # This makes all real data fit into [0, 2]
    # which numerically makes sense (IMO)
    if not raw_values:
        obs[:, :2] = 1000 * (obs[:, :2] / N)  # I+R per 1 000
        obs[:, 2] = 100000 * (obs[:, 2] / N)  # D per 100 000

    infected = obs[:, 0:1]
    recovered = obs[:, 1:2]
    dead = obs[:, 2:]
    
    # Here we do the infamous reshape
    # If T is not a multiple of 7, this will throw an error at you
    if weekly:
        infected = infected.reshape(obs.shape[0] // 7, 7)
        recovered = recovered.reshape(obs.shape[0] // 7, 7)
        dead = dead.reshape(obs.shape[0] // 7, 7)

        # Also for each compartment, we add N as an extra in_channel
        # so the model can learn something even when using
        # relative numbers
        Ns = np.full((infected.shape[0], 1), N / 1e9)
        obs = np.concatenate([infected, Ns, recovered, Ns, dead, Ns], axis=-1)
    
    else:
        Ns = np.full_like(infected, N / 1e9)
        obs = np.concatenate([infected, Ns, recovered, Ns, dead, Ns], axis=-1)
    
    return obs


def data_generator(batch_size,
                   T=None, N=None, T_min=10, T_max=104,  # 104 weeks = 2 years
                   N_min=100000, N_max=2e9,  # India -> need more population
                   to_tensor=True, seed=None,
                   observ_model=True, weekly=True,
                   num_interventions=15,
                   accept_only_reasonable=False):
    np.seterr(all='raise')
    if seed is not None:
        np.random.seed(seed)

    # Variable-size t
    if T is None:
        T = np.random.randint(T_min, T_max + 1)

    if weekly:
        T = T * 7

    # Variable size N
    if N is None:
        N = np.random.randint(N_min, N_max)

    x = []
    thetas = []

    while len(thetas) < batch_size:
        try:
            simulation = sample_simulation(T, N, num_interventions=num_interventions)
        except (ValueError, RuntimeWarning, FloatingPointError):
            continue
        
        theta_i = params2vector(simulation['params'])
        
        if observ_model:
            obs = simulation['observed_data']
        else:
            obs = simulation['secird'][:, [3, 4, 5]]
            obs[:, 1:] = np.diff(obs[:, 1:], axis=0, prepend=0)
        
        if accept_only_reasonable:
            total_infected = np.sum(obs[:, 0])
            if not (N / 100 < total_infected < 0.15 * N):
                continue

        x_i = normalise(obs, N=N, weekly=weekly)
        
        thetas.append(theta_i)
        x.append(x_i)
    
    x = np.stack(x)
    thetas = np.stack(thetas)

    # Convert to tensor, if specified
    if to_tensor:
        thetas = torch.from_numpy(thetas).float()
        x = torch.from_numpy(x).float()

    return {'theta': thetas, 'x': x}
