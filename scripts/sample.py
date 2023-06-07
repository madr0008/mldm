import torch
import numpy as np
import zero
import os
from mldm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset
from lib import toArff
from sklearn.preprocessing import MinMaxScaler
import lib

def to_good_ohe(ohe, X):
    aux = [len(a) for a in ohe.categories_]
    indices = np.cumsum([0] + aux)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    sample_percentage = 0,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    device = torch.device('cuda:1'),
    seed = 0,
    strategy = "general",
    label_percentage=50,
    max_iter = 1000,
    output_file = "salida.arff",
    alFinal = True,
    quantileFinal=False,
    quantile = False
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    datasets = make_dataset(
        real_data_path,
        T,
        strategy,
        label_percentage
    )

    D_aux = make_dataset(
        real_data_path,
        lib.Transformations(),
        "general",
        0
    )[0]

    X_num_total = D_aux.X_num
    X_cat_total = D_aux.cat_transform.inverse_transform(D_aux.X_cat)

    for i in range(len(datasets)):

        D = datasets[i]

        num_samples = int(((sample_percentage * D.size()))/100)

        K = np.array(D.get_category_sizes())
        if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
            K = np.array([0])

        num_numerical_features_ = D.n_num_features
        d_in = np.sum(K) + num_numerical_features_
        model_params['d_in'] = int(d_in)
        model = get_model(
            model_type,
            model_params,
        )

        model.load_state_dict(
            torch.load(os.path.join(parent_dir, 'model_' + str(i) + '.pt'), map_location="cpu")
        )

        diffusion = GaussianMultinomialDiffusion(
            K,
            num_numerical_features=num_numerical_features_,
            denoise_fn=model, num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
        )

        diffusion.to(device)
        diffusion.eval()

        if strategy == "general":
            X_gen = diffusion.sample_loop(num_samples, max_iter, batch_size, D, ddim=False)
        else:
            X_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)

        num_numerical_features = D.n_num_features_before

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            if T_dict['cat_encoding'] == 'one-hot':
                X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
            X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])
            X_cat_total = np.concatenate((X_cat_total, X_cat), axis=0)

        if num_numerical_features != 0:
            X_num_ = X_gen[:, :num_numerical_features]
            if alFinal:
                X_num_ = D.num_transform.inverse_transform(X_num_)
            if not quantile or (quantile and quantileFinal):
                for col in range(num_numerical_features):
                    scaler = MinMaxScaler(feature_range=(D_aux.X_num.min(axis=0)[col], D_aux.X_num.max(axis=0)[col]))
                    scaler.fit(X_num_[:,col].reshape(-1, 1))
                    X_num_[:,col] = scaler.transform(X_num_[:,col].reshape(-1, 1)).flatten()
            X_num = X_num_[:, :num_numerical_features]

            X_num_total = np.concatenate((X_num_total, X_num), axis=0)

    nInstances = X_num_total.shape[0] if X_num_total is not None else X_cat_total.shape[0]

    toArff(D_aux, X_num_total, X_cat_total, nInstances, output_file)