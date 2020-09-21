from spinup.utils.plot import make_plots

if __name__ == '__main__':
    values = ['EpLen', 'AverageVVals', 'LossPi', 'LossV', 'Entropy', 'KL', 'ClipFrac', 'StopIter']

    make_plots(['./data'], xaxis=['TotalEnvInteracts'], values=values)
