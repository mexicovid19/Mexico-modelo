###-----------###
### Importing ###
###-----------###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from scipy import integrate
import seaborn as sns; sns.set()

###---------###
### Reading ###
###---------###
# be sure to git pull upstream master before reading the data so it is up to date.
DATA_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
DATA_URL_MEX = 'https://raw.githubusercontent.com/LeonardoCastro/COVID19-Mexico/master/data/series_tiempo/'

###------------------###
### Helper Functions ###
###------------------###

## For global
def country_to_timeseries(df, country):
    df_country = df[ df['Country/Region'] == country ].drop(['Province/State','Lat','Long'], axis=1)

    return df_country.groupby('Country/Region').sum().T[country]

def province_to_timeseries(df, province):
    df_province = df[ df['Province/State'] == province ]

    return df_province.set_index('Province/State').drop(['Country/Region','Lat','Long'], axis=1).T[province]

## For Mexico
def statal_timeseries(df, log=False):
    '''
    Returns a dataframe with the number of COVID cases where each row is indexed by a date (t0 = 2020-02-28), and each column is a state of Mexico.
    If log=True, return the log of the cases.
    '''
    if log:
        return np.log10( df.drop(['Mexico_pais'], axis=1).set_index('Fecha') )
    else:
        return df.drop(['Mexico_pais'], axis=1).set_index('Fecha')

def national_timeseries(df, log=False):
    '''
    Returns a dataframe with the national number of COVID cases for Mexico where each row is indexed by a date (t0 = 2020-02-28).
    If log=True, return the log of the cases.
    '''
    if log:
        return np.log10( df.set_index('Fecha').loc[:,['Mexico_pais']] )
    else:
        return df.set_index('Fecha').loc[:,['Mexico_pais']]

## Other helper Functions

# helper to sum I+H+R+D cases
def total_cases_timeseries(sol, N):
    return (sol[:, 2] + sol[:, 3] + sol[:, 4] + sol[:, 5])*N

# mean absolute error
def MAE(x,y): return np.mean(np.abs( x - y ))

def sol_to_csv(filename, sol, initial_date, total_population):

    try:
        # our format
        t0 = datetime.datetime.strptime(initial_date, '%Y-%m-%d')
    except:
        # John Hopkins format
        t0 = datetime.datetime.strptime(initial_date, '%m/%d/%y')
    t_range = [t0 + datetime.timedelta(days=x) for x in range( sol.shape[0] )]
    CSV = pd.DataFrame(columns=['Totales','Recuperados','Muertes','Hospitalizados'])

    CSV['Totales'] = total_cases_timeseries(sol, N=total_population)
    CSV['Recuperados'] = sol[:,4] * total_population
    CSV['Muertes'] = sol[:,5] * total_population
    CSV['Hospitalizados'] = sol[:,3] * total_population
    CSV.index = t_range

    CSV.to_csv('data/'+filename)

###------------------###
### Model Definition ###
###------------------###

### Non-compartamental ###
    ## Discrete time Markovian model ##
def SEIHRD_markov_step(x, t, *params):
    '''
    Suceptible (S), Exposed (E), Infected (I), Hospitalized (H), Recovered (R), Deceased (D) epidemic model.
    The function takes a single time step in the units of days.
    '''
    λ, κ, γI, μI, α, γH, μH = params
    S,E,I,H,R,D = x
    return [S*(1-λ*I),
            λ*I*S + (1-κ)*E,
            κ*E + (1 - (γI+μI+α))*I,
            α*I + (1 - (γH+μH))*H,
            γI*I + γH*H + R,
            μI*I + μH*H + D]

### Solver ###
def solve(f, x0, t0, n_steps, *params):
    '''
    Maps the markov chain defined by `f` with initial distribution `x0` at `t0` for `n_steps` steps.
    '''
    xt = [xi for xi in x0]
    sol = np.zeros( (n_steps, len(x0)) )

    t = t0
    for (i,t) in enumerate( range(n_steps) ):

        sol[i,:] = xt

        xt = f(xt, 0.0, *params)
        t += 1

    return sol

if __name__ == "__main__":

    ## READING DATA ##
    # ElLeo data: mexican states
    mex_confirmed = pd.read_csv(DATA_URL_MEX+'covid19_mex_casos_totales.csv', )
    mex_deaths = pd.read_csv(DATA_URL_MEX+'covid19_mex_muertes.csv', )
    mex_recovered = pd.read_csv(DATA_URL_MEX+'covid19_mex_recuperados.csv', )

    ## SETTING MODEL PARAMETERS ##
    R_0 = 2.3 # ±0.3 # empirical basic reproductive ratio
    # Arenas parameters
    β = 0.06 # infectivity of the desease (per contact per day)
    k_avg = 13.3 # average contacts per day. (11.8, 13.3, 6.6) in the report
    μg = 1/3.2 # escape (from Infected) rate. (1/1, 1/3.2, 1/3.2) in the report
    γg = 0.05 # fraction of cases requiring ICU. (0.002, 0.05, 0.36) in the report
    # γg should be a weighted sum of %_young * 0.002 + %_mid * 0.05 + %_old * 0.36
    ωg = 0.42 # fatality rate of ICU patients
    ψg = 1/7  # death rate
    χg = 1/10 # ICU discharge rate

    # SEIHRD params
    κ = 1/5.2 # η^-1 + α^-1
    γI = μg*(1 - γg)
    μI = 0
    α = μg*γg
    μH = ωg*ψg
    γH = (1 - ωg)*χg
    λ = -k_avg * np.log( 1 - β ) # linearization of Π(t) of Arenas' report
    # λ = R_0 * (γI + μI + α)  # using the R0

    params = (λ, κ, γI, μI, α, γH, μH)

    ## INITIAL CONDITIONS SETUP ##
    N_mex = 128_569_304 # https://www.worldometers.info/world-population/mexico-population/ (2020-03-34)
    initial_date = '2020-03-12' # This is the date in which the trend becomes exponential

    confirmed = national_timeseries(mex_confirmed).loc[initial_date,'Mexico_pais']
    deaths = national_timeseries(mex_deaths).loc[initial_date,'Mexico_pais']
    recovered = national_timeseries(mex_recovered).loc[initial_date,'Mexico_pais']

    r = 6 # proportion of unobserved (exposed) cases
    R0 = recovered / N_mex
    D0 = deaths / N_mex
    I0 = (confirmed/N_mex) - R0 - D0  # confirmed cases
    E0 = r*I0 # cases without symptoms, which are not yet detected
    H0 = 0 # no hospitalization data yet
    S0 = (1 - E0 - I0 - R0 - D0 - H0)

    x0 = [S0, E0, I0, H0, R0, D0]

    ### DYNAMICS ###
    prediction_horizon = 5 # days
    n_days = (datetime.datetime.today() - datetime.datetime.strptime(initial_date, '%Y-%m-%d')).days + prediction_horizon

    sol = solve(SEIHRD_markov_step, x0, 0.0, n_days, *params)

    ### SAVING RESULTS ###
    sol_to_csv('covid19_mex_proyecciones.csv', sol, initial_date, N_mex)
    print('Done! Projections made for Mexico with initial condition at t0 = {} with data available until {}'.format(initial_date, mex_confirmed.Fecha.values[-1]))
