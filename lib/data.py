import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from bs4 import BeautifulSoup as bs

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
import os
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from . import env, util
from .util import load_json


CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def loadArff(arffPath, xmlPath):
    #Get labels from XML file
    with open(xmlPath, 'r') as file:
        content = file.read()
    bs_content = bs(content, features="xml")
    result = bs_content.find_all("label")
    labelNames = [x.get("name") for x in result]
    #Get values from ARFF file
    with open(arffPath, 'r') as arff:
        lines = arff.readlines()
    catAttributesIndexes = []
    numAttributesIndexes = []
    labelsIndexes = []
    aux = 0
    cont = 0
    header = ''
    #Get attribute indexes
    for i in range(len(lines)):
        header += lines[i]
        if '@data' in lines[i]:
            header = header[:-1]
            aux = i
            break
        elif '@attribute' in lines[i]:
            v = lines[i].split(' ')
            if v[2] == 'numeric\n':
                numAttributesIndexes.append(cont)
            elif v[1] not in labelNames:
                catAttributesIndexes.append(cont)
            else:
                labelsIndexes.append(cont)
            cont += 1

    #Get data
    catAttributes = []
    numAttributes = []
    labels = []
    numDecimals = [0] * len(numAttributesIndexes)
    for line in lines[(aux+1):]:
        if line[0] == '{':
            #Sparse
            sparse = True
            catAttributesAux = ['0' for c in catAttributesIndexes]
            numAttributesAux = [0 for n in numAttributesIndexes]
            pairs = line.replace('{', '').replace('}', '').replace('\n', '').split(',')
            for p in pairs:
                pair = p.split(' ')
                index = int(pair[0])
                if index in catAttributesIndexes or i in labelsIndexes:
                    catAttributesAux[catAttributesIndexes.index(index)] = pair[1]
                else:
                    numAttributesAux[numAttributesIndexes.index(index)] = float(pair[1])
                    nAux = len(pair[1].split('.')[1]) if '.' in pair[1] else 0
                    if nAux > numDecimals[numAttributesIndexes.index(index)]:
                        numDecimals[numAttributesIndexes.index(index)] = nAux
            if len(catAttributesIndexes) + len(labelsIndexes) > 0:
                catAttributes.append(catAttributesAux)
            if len(numAttributesIndexes) > 0:
                numAttributes.append(numAttributesAux)
            labels.append(0)    #Esto lo quiero quitar
        else:
            #Normal
            sparse = False
            catAttributesAux = []
            numAttributesAux = []
            values = line.split(',')
            for i in range(len(values)):
                if i in catAttributesIndexes or i in labelsIndexes:
                    catAttributesAux.append(values[i].replace('\n',''))
                else:
                    numAttributesAux.append(float(values[i]))
                    nAux = len(values[i].split('.')[1]) if '.' in values[i] else 0
                    if nAux > numDecimals[numAttributesIndexes.index(i)]:
                        numDecimals[numAttributesIndexes.index(i)] = nAux
            if len(catAttributesIndexes) + len(labelsIndexes) > 0:
                catAttributes.append(catAttributesAux)
            if len(numAttributesIndexes) > 0:
                numAttributes.append(numAttributesAux)
            labels.append(0)    #Esto lo quiero quitar

    #Transform lists of lists to numpy arrays
    numAttributesRet = np.array([np.array(xi) for xi in numAttributes]) if len(numAttributes) > 0 else None
    catAttributesRet = np.array([np.array(xi) for xi in catAttributes]) if len(catAttributes) > 0 else None
    labelsRet = np.array(labels)    #Esto lo quiero quitar

    return numAttributesRet, catAttributesRet, labelsRet, len(labelNames), header, sparse, numAttributesIndexes, catAttributesIndexes, labelsIndexes, numDecimals


def toArff(D, X_num, X_cat, num_instances, filename):
    content = D.arffHeader
    for i in range(num_instances):
        line = ''
        if D.sparse:
            line = '{'
        contNum = 0
        contCat = 0
        for j in (D.numIndexes + D.catIndexes + D.labelIndexes):
            if j in D.numIndexes:
                if D.sparse:
                    line += str(j) + ' ' + str(round(X_num[i][contNum], D.numDecimals[contNum])) + ','
                else:
                    line += str(round(X_num[i][contNum], D.numDecimals[contNum])) + ','
                contNum += 1
            else:
                if D.sparse:
                    if X_cat[i][contCat] == 1:
                        line += str(j) + ' ' + str(X_cat[i][contCat]) + ','
                else:
                    line += str(X_cat[i][contCat]) + ','
                contCat += 1
        line = line[:-1]
        if D.sparse:
            line += '}'
        content += '\n' + line
    with open(filename, 'w') as file:
        file.write(content)


@dataclass(frozen=False)
class Dataset:
    X_num: np.ndarray
    X_cat: np.ndarray
    y: np.ndarray
    n_labels: Optional[int]
    arffHeader: str
    sparse: bool
    numIndexes: list
    catIndexes: list
    labelIndexes: list
    numDecimals: list

    @classmethod
    def from_dir(cls, dir_: str, strategy: str, pct: int):

        dir_ = Path(dir_)
        X_num, X_cat, y, numLabels, header, sparse, numIndexes, catIndexes, labelIndexes, numDecimals = loadArff((str(dir_) + '.arff'), (str(dir_) + '.xml'))

        D = Dataset(
                X_num,
                X_cat,
                y,
                numLabels,
                header,
                sparse,
                numIndexes,
                catIndexes,
                labelIndexes,
                numDecimals
            )

        if strategy == "general":
            return [D]
        elif strategy == "specific":
            return D.get_minoritary_datasets(pct)
        else:
            return [D.get_minoritary_dataset(pct)]


    #Returns the labels with a IR below MeanIR (not yet though)
    def get_minoritary_labels(self):

        labelCount = [0] * self.n_labels

        for instance in self.X_cat:
            for label in range(len(labelCount)):
                labelCount[label] += int(instance[-(self.n_labels - label)])

        IRlbl = []

        for count in labelCount:
            IRlbl.append(((self.X_cat.shape[0]/count) - 1))

        meanIR = sum(IRlbl)/len(IRlbl)

        minoritaryLabels = []

        for i in range(len(IRlbl)):
            if IRlbl[i] < meanIR:
                minoritaryLabels.append((i, labelCount[i]))

        return sorted(minoritaryLabels, key=lambda x: x[1])


    def get_minoritary_instances(self, pct):

        minoritaryLabels = self.get_minoritary_labels()[:int(self.n_labels)*pct]
        minoritaryIndexes = dict((str(j), []) for (j, n) in minoritaryLabels)
        for i in range(self.X_cat.shape[0]):
            for (j, n) in minoritaryLabels:
                if int(self.X_cat[i][-(self.n_labels - j)]) == 1:
                    minoritaryIndexes[str(j)].append(i)
        return minoritaryIndexes


    #Returns a list of datasets, each one with instances with minoritary labels
    def get_minoritary_datasets(self, pct):

        minoritaryIndexes = self.get_minoritary_instances(pct)

        minoritaryDatasets = []
        for j, l in minoritaryIndexes.items():
            X_num = np.take(self.X_num, l, axis=0)
            X_cat = np.take(self.X_cat, l, axis=0)
            D = Dataset(
                X_num,
                X_cat,
                np.array(([0] * X_cat.shape[0])),
                self.n_labels,
                self.arffHeader,
                self.sparse,
                self.numIndexes,
                self.catIndexes,
                self.labelIndexes,
                self.numDecimals
            )
            minoritaryDatasets.append(D)

        return minoritaryDatasets


    #Returns a dataset with instances with minoritary labels
    def get_minoritary_dataset(self, pct):

        minoritaryIndexes = self.get_minoritary_instances(pct)

        indexes = []

        for j, l in minoritaryIndexes.items():
            for index in l:
                if index not in indexes:
                    indexes.append(index)

        X_num = np.take(self.X_num, indexes, axis=0)
        X_cat = np.take(self.X_cat, indexes, axis=0)
        D = Dataset(
            X_num,
            X_cat,
            np.array(([0] * X_cat.shape[0])),
            self.n_labels,
            self.arffHeader,
            self.sparse,
            self.numIndexes,
            self.catIndexes,
            self.labelIndexes,
            self.numDecimals
        )

        return D


    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num.shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat.shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self) -> int:
        return len(self.y)

    @property
    def nn_output_dim(self) -> int:
        return self.n_labels

    def get_category_sizes(self) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat)


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: np.ndarray, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> np.ndarray:
    X_train = X
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return normalizer.transform(X), normalizer
    return normalizer.transform(X)


def cat_encode(
    X: np.ndarray,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[np.ndarray, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X)
        encoder = make_pipeline(oe)
        encoder.fit(X)
        X = encoder.transform(X)
        max_values = X.max(axis=0)
        for column_idx in range(X.shape[1]):
            X[X[:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X)
        X = encoder.transform(X)
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X, y_train)
        X = encoder.transform(X)
        if not isinstance(X, pd.DataFrame):
            X = X.values()
    else:
        util.raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    cat_encoding: Optional[CatEncoding] = None


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform
    
    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat, is_num, cat_transform = cat_encode(
            dataset.X_cat,
            transformations.cat_encoding,
            dataset.y,
            transformations.seed,
            return_encoder=True
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=dataset.y)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
        # return dataset, num_transform, cat_transform
    return dataset


def build_dataset(
    path: Union[str, Path],
    transformations: Transformations,
    cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    X_num, X_cat, Y = (
        None if x is None else torch.as_tensor(x)
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else x.to(device)
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = Y.float()
    return X_num, X_cat, Y

###############
## DataLoader##
###############

class TabDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset : Dataset
    ):
        super().__init__()

        self.X_num = torch.from_numpy(dataset.X_num) if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat) if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y)

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            'y': self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict

def prepare_dataloader(
    dataset : Dataset,
    batch_size: int,
):

    torch_dataset = TabDataset(dataset)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    while True:
        yield from loader

def prepare_torch_dataloader(
    dataset : Dataset,
    shuffle : bool,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    torch_dataset = TabDataset(dataset)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def prepare_fast_dataloader(
    D : Dataset,
    batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num, D.X_cat], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat).float()
    else:
        X = torch.from_numpy(D.X_num).float()
    y = torch.from_numpy(D.y)
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=True)
    while True:
        yield from dataloader

def prepare_fast_torch_dataloader(
    D : Dataset,
    batch_size: int
):
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num, D.X_cat], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num).float()
    y = torch.from_numpy(D.y)
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=True)
    return dataloader

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = pd.DataFrame(D.X_cat, columns=range(D.n_features))
    elif D.X_cat is None:
        assert D.X_num is not None
        X = pd.DataFrame(D.X_num, columns=range(D.n_features))
    else:
        X = pd.concat(
                [
                    pd.DataFrame(D.X_num, columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat,
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )

    return X

def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)