from lib.data import loadArff, toArff
from scripts.utils_train import make_dataset
import lib

def prueba():
    D = make_dataset(
        '/home/miguelangel/datasets/birds/birds',
        lib.Transformations(None),
        True
    )

    toArff(D, D.X_num['train'], D.X_cat['train'], 20, 'salida.arff')

prueba()