import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset
from lib import round_columns, toArff
from sklearn.preprocessing import MinMaxScaler
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
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
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    strategy = "general",
    label_percentage=50,
    max_iter = 1000
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

    minLabels = D_aux.get_minoritary_labels()

    X_num_total = D_aux.X_num['train']
    X_cat_total = D_aux.X_cat['train']

    for i in range(len(datasets)):

        D = datasets[i]

        num_samples = int(((sample_percentage * D.size()))/100)

        K = np.array(D.get_category_sizes('train'))
        if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
            K = np.array([0])

        num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features_
        model_params['d_in'] = int(d_in)
        model = get_model(
            model_type,
            model_params,
            num_numerical_features_,
            category_sizes=D.get_category_sizes('train')
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

        _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)

        if strategy == "general":
            X_gen, y_gen = diffusion.sample_loop(num_samples, max_iter, batch_size, empirical_class_dist.float(), D, ddim=False)
        else:
            X_gen, y_gen = diffusion.sample_all(minLabels[i][1], batch_size, empirical_class_dist.float(), ddim=False)

        ###
        # X_num_unnorm = X_gen[:, :num_numerical_features]
        # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
        # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
        # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
        # X_gen = X_gen[np.all(idx, axis=1)]
        # y_gen = y_gen[np.all(idx, axis=1)]
        ###

        num_numerical_features = num_numerical_features_

        #ESTO TIENE QUE SER UN PARAMETRO
        alFinal = True
        quantile = False

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
            if T_dict['cat_encoding'] == 'one-hot':
                X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
            X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])
            X_cat_total = np.concatenate((X_cat_total, X_cat), axis=0)

        if num_numerical_features_ != 0:
            X_num_ = X_gen[:, :num_numerical_features]
            if alFinal or quantile:
                X_num_ = D.num_transform.inverse_transform(X_num_)
            if not quantile:    #Y no se, quizas rentaria
                for col in range(num_numerical_features):
                    scaler = MinMaxScaler(feature_range=(D_aux.X_num['train'].min(axis=0)[col], D_aux.X_num['train'].max(axis=0)[col]))
                    scaler.fit(X_num_[:,col].reshape(-1, 1))
                    X_num_[:,col] = scaler.transform(X_num_[:,col].reshape(-1, 1)).flatten()
            X_num = X_num_[:, :num_numerical_features]

            disc_cols = []
            for col in range(D.X_num['train'].shape[1]):
                uniq_vals = np.unique(D.X_num['train'][:, col])
                if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                    disc_cols.append(col)
            print("Discrete cols:", disc_cols)
            if len(disc_cols):
                X_num = round_columns(D.X_num['train'], X_num, disc_cols)
            X_num_total = np.concatenate((X_num_total, X_num), axis=0)


    toArff(D_aux, X_num_total, X_cat_total, X_num_total.shape[0], 'salida.arff')