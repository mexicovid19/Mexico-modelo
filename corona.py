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

# JohnHopkins data: by country
# these time series have the old format. The new ones don't include the recovered
global_confirmed = pd.read_csv(DATA_URL+'time_series_19-covid-Confirmed.csv')
global_deaths = pd.read_csv(DATA_URL+'time_series_19-covid-Deaths.csv')
global_recovered = pd.read_csv(DATA_URL+'time_series_19-covid-Recovered.csv')

# ElLeo data: mexican states
# parse_dates=['Fecha']
mex_confirmed = pd.read_csv(DATA_URL_MEX+'covid19_mex_casos_totales.csv', )
mex_deaths = pd.read_csv(DATA_URL_MEX+'covid19_mex_muertes.csv', )
mex_recovered = pd.read_csv(DATA_URL_MEX+'covid19_mex_recuperados.csv', )

###------------------###
### Helper Functions ###
###------------------###

## For global
def df_to_timeseries(df):
    return df.drop(['Province/State','Lat','Long'], axis=1).groupby('Country/Region').sum().T

def df_to_timeseries_province(df):
    return df.drop(['Lat','Long'], axis=1).set_index('Country/Region')

def country_to_timeseries(df, country):
    return df.drop(['Lat','Long'],axis=1).set_index(['Country/Region','Province/State']).loc[country].T

def country_to_timeseries(df, country):
    df_country = df[ df['Country/Region'] == country ].drop(['Province/State','Lat','Long'], axis=1)

    return df_country.groupby('Country/Region').sum().T

def province_to_timeseries(df, province):
    df_province = df[ df['Province/State'] == province ]

    return df_province.set_index('Province/State').drop(['Country/Region','Lat','Long'], axis=1).T#[province]

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
def total_cases_timeseries(sol, N):
    return (sol[:, 2] + sol[:, 3] + sol[:, 4] + sol[:, 5])*N

def MAE(x,y): return np.mean(np.abs( x - y ))

def sol_to_csv(sol, initial_date, total_population):

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

    return CSV

###------------------###
### Model Definition ###
###------------------###

### Non-compartamental ###
    ## Discrete time model ##

def SEIHRD_markov_step(x, t, *params):
    λ, κ, γI, μI, α, γH, μH = params
    S,E,I,H,R,D = x
    return [S*(1-λ*I),
            λ*I*S + (1-κ)*E,
            κ*E + (1 - (γI+μI+α))*I,
            α*I + (1 - (γH+μH))*H,
            γI*I + γH*H + R,
            μI*I + μH*H + D]

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
