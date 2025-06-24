from scipy.optimize import curve_fit, minimize
import warnings
import numpy as np


def uniform(a, b, shape):
    return lambda: np.random.uniform(a, b, 1)[0] * np.ones(shape)


def MAE(a, b):
    return np.mean(np.abs(a - b))


def MSE(a, b):
    return np.mean((a - b) ** 2)


def rMSE(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def MeanRelE(a, b):
    return np.mean(np.abs(a - b) / (np.abs(b) + 0.01))


def MaxRelE(a, b):
    return np.max(np.abs(a - b) / (np.abs(b) + 0.01))


def disentangle_gradients(grads, layout):
    gradients = []
    for val, size in zip(grads, layout.values()):
        if size == 1:
            gradients.append(np.atleast_1d(np.mean(val, axis=-1)))
        else:
            gradients.append(val / float(size))
    return np.concatenate(gradients).ravel()


def pack_parameters(params):
    return np.concatenate(
        [np.ravel(param) for param in params]
    )


def unpack_parameters(params_vector, layout):
    params = []
    idx = 0
    for size in layout:
        if size == 1:
            params.append(params_vector[idx: idx + 1])
        else:
            params.append(params_vector[idx: idx + size])
        idx += size
    return params


def fit_bsdf(guess, limits, model, model_der=None):
    np.random.seed(seed=42)

    param_layout = [
        np.atleast_1d(param).shape[0] for param in guess
    ]

    x0 = pack_parameters(guess)
    bounds = [
        bound
        for param_width, limit in zip(param_layout, limits)
        for bound in [limit] * param_width
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        def jac(packed):
            return model_der(*unpack_parameters(packed, param_layout))

        result = minimize(
            lambda packed: model(*unpack_parameters(packed, param_layout)),
            x0=x0,
            jac=jac if model_der != None else None,
            bounds=bounds
        )
        popt = unpack_parameters(result.x, param_layout)

    model_output = model(*popt)

    return result, popt, model_output
