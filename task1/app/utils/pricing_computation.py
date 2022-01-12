import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf

from .regression_model import MyModel
from app.models.input import Input

tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float32')
print(tf.__version__)

path = './data_models/'

def _load_model(network,location):

  model = MyModel(inputs=network['inputs'],units=network['units'],outputs=network['outputs'],seed=0)
  model.load_weights(location)

  return model


def _calculate_indices(model, scenarios, benchmarks):    
    d = {'index': np.array(list(scenarios.values()))}
    tensorflow_output = model(d)
    sigma = tensorflow_output['sigma'].numpy()
    beta = tensorflow_output['beta'].numpy()
    
    prices = []
    i = 0
    for index, mapping in scenarios.items():
        S = np.exp((sigma[i] ** 2.) / 2.)
        B = np.sum(np.multiply(beta[i], benchmarks))
        price = float(S * B)
        prices.append(price)
        i+=1
        
    return prices


def get_benchmarks():
    
    benchmarks = [
        {'date': dt.datetime(2021, 10, 1), 'bmk1': 1.072119, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 2), 'bmk1': 1.5447569, 'bmk2': 1.678678, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 3), 'bmk1': 1.546745745, 'bmk2': 1.578587, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 4), 'bmk1': 1.456754475, 'bmk2': 1.678678, 'bmk3': 3.68768, 'bmk4':  5.3125, 'bmk5': 1.67868, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 5), 'bmk1': 1.4675457, 'bmk2': 1.678687, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 6), 'bmk1': 1.4756745, 'bmk2': 1.687687, 'bmk3': 4.678867, 'bmk4':  8.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 7), 'bmk1': 1.64574567, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.688677, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 8), 'bmk1': 1.4574576, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 9), 'bmk1': 1.347367, 'bmk2': 1.616101, 'bmk3': 3.68768, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 10), 'bmk1': 1.4567, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 11), 'bmk1': 1.4567, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.678686, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 12), 'bmk1': 1.45687458, 'bmk2': 1.616101, 'bmk3': 3.687867, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 13), 'bmk1': 1.65889, 'bmk2': 1.616101, 'bmk3': 3.67868, 'bmk4':  5.678687, 'bmk5': 1.341723, 'bmk6': 1.089426},
        {'date': dt.datetime(2021, 10, 14), 'bmk1': 1.9709, 'bmk2': 1.616101, 'bmk3': 3.995879, 'bmk4':  5.3125, 'bmk5': 1.341723, 'bmk6': 1.089426}
    ]
        
    benchmarks_df= pd.DataFrame(benchmarks)
    benchmarks_df.set_index(['date'], inplace=True)
    
    return benchmarks

def get_scenarios():
    
    scenarios = {}
    scenarios['1'] = [1,0,0,0,0,0] # index 1
    scenarios['2'] = [0,1,0,0,0,0] # index 2
    scenarios['3'] = [0,0,1,0,0,0] # index 3
    scenarios['4'] = [0,0,0,1,0,0] # index 4
    scenarios['5'] = [0,0,0,0,1,0] # index 5
    scenarios['6'] = [0,0,0,0,0,1] # index 6
    
    return sce

def pricing_computation(data: Input):
    data = []
    
    for scenario in data["scenarios"]:
        
        inputs = {'index': scenario["project"]["index"]}
        outputs = {'beta': ['bmk1', 
                            'bmk2', 
                            'bmk3', 
                            'bmk4', 
                            'bmk5', 
                            'bmk6'
                            ],
                'sigma': ['sigma']}
        units = {'input':800,'hidden':[800]*8}
        
        network = {'inputs': inputs,'outputs':outputs,'units':units}
        model = _load_model(network=network,location=path+'model-seed-weights')

        benchmarks = get_benchmarks()
        scenarios = get_scenarios()
        data.append({
            "project": {"index": scenario["project"]["index"] },
            "history": _calculate_indices(model, scenarios, benchmarks_df.loc[data["end_date"]])
        })
    
    return data