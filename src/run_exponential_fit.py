###-----------###
### Importing ###
###-----------###
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

###------------------###
### Helper Functions ###
###------------------###

## Time series management
def national_timeseries(df, log=False):
    '''
    Returns a dataframe with the national number of COVID cases for Mexico where each row is indexed by a date (t0 = 2020-02-28).
    If log=True, return the log of the cases.
    '''
    if log:
        return np.log10( df.set_index('Fecha').loc[:,['Nacional']] )
    else:
        return df.set_index('Fecha').loc[:,['Nacional']]

###-------###
### Model ###
###-------###
def exponential_model(t, y0, β): return y0 * np.exp(β*t)


if __name__ == "__main__":

    # Reading data
    DATA_URL_MEX = 'https://raw.githubusercontent.com/mexicovid19/Mexico-datos/master/datos/series_de_tiempo/'
    mex_confirmed = pd.read_csv(DATA_URL_MEX+'covid19_mex_casos_totales.csv', )

    # paths for saving
    PLOT_PATH = '../media/'
    CSV_PATH  = '../results/'
    save_ = True

    # Fit time window
    n_days = 7
    total_cases_timeseries = national_timeseries(mex_confirmed).iloc[-n_days:,0]

    # Data preparation
    xdata = np.array( range(len(total_cases_timeseries)) )
    ydata = total_cases_timeseries.values

    # Initial parameter guess
    p0 = [ydata[-1], 0.1]

    # Model parameter fit. Returns parameters and their covariance matrix
    popt, pcov = curve_fit(exponential_model, xdata, ydata, p0)

    # Projection days
    forecast_horizon = 2 # days

    # Growth rate std
    σ = np.sqrt( pcov[1,1] )

    # Fitting and projecting
    xfit = range( len(xdata) + forecast_horizon )
    yfit = exponential_model(xfit, *popt)
    yfit_min = exponential_model(xfit[-(forecast_horizon+1):], popt[0], popt[1] - 2*σ, )
    yfit_max = exponential_model(xfit[-(forecast_horizon+1):], popt[0], popt[1] + 2*σ, )

    # helper temporal values
    trange = total_cases_timeseries.index.values
    t0 = datetime.datetime.strptime( trange[0], '%Y-%m-%d')
    tfit = [(t0 + datetime.timedelta(days=t)).strftime('%Y-%m-%d') for t in xfit ]

    # Dataframe with fits and data
    Data = national_timeseries(mex_confirmed)
    Data['Nacional'] = Data['Nacional'].astype(int)
    CSV = Data.join(pd.Series( np.round(yfit), tfit, name='Fit'), how='outer' )
    CSV = CSV.join(pd.Series( np.round(yfit_min), tfit[-(forecast_horizon+1):], name='Fit_min'), how='outer' )
    CSV = CSV.join(pd.Series( np.round(yfit_max), tfit[-(forecast_horizon+1):], name='Fit_max'), how='outer' )
    CSV.index.rename('Fecha', inplace=True)
    print(CSV)


    # Plotting
    plt.figure( figsize=(10,8) )
    # plot data
    plt.plot(trange, ydata, lw=0, marker='o', ms=8)
    # plot fit
    plt.plot(tfit, yfit, c='orange', label='tasa esperada: {} % diario'.format( np.round(popt[1]*100 ,1)) )
    # error cones
    plt.fill_between(tfit[-(forecast_horizon+1):], yfit_min, yfit_max,
                         alpha=0.2, color='orange');

    plt.title( 'Casos totales de COVID-19 en {}'.format( 'México' ) , size=16)
    plt.ylabel('Número de casos', size=15);
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    # plt.yscale('log')
    plt.tight_layout()

    if save_:
        plt.savefig( PLOT_PATH+'covid19_mex_fit.png' )
        CSV.to_csv( CSV_PATH+'covid19_mex_fit.csv' )
        plt.show()
    else:
        print(CSV)
        plt.show()
