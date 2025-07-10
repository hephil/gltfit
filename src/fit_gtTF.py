import json
import merl
import matplotlib.pyplot as plt
import os
import glob
from bsdf import *
from glTF import *
from fit_bsdf import *


def fit_merl_brdf(material, dir="../merl100/brdfs/", analytic_gradient=False, **kwargs):
    """
    Fit the glTF material model to a Merl BRDF
    """

    # Load Merl100 Data
    merl_data = merl.read_merl_brdf(os.path.join(dir, f"{material}.binary"))

    # # Load just the phi_d=90° slice
    # measured = merl_data[:, :, :, 90].reshape(merl_data.shape[0], -1)

    # theta_h, _, theta_d, phi_d = merl.generate_dense_half_diffs(0)
    # theta_h, theta_d, phi_d = np.meshgrid(
    #     theta_h,
    #     theta_d,
    #     phi_d[90],
    #     indexing='ij'
    # )
    # theta_h, theta_d, phi_d = theta_h.ravel(), theta_d.ravel(), phi_d.ravel()
    # theta_o, phi_o, theta_i, phi_i = merl.half_diff_to_std_coords(
    #     theta_h,
    #     np.broadcast_to(0, theta_h.shape),
    #     theta_d,
    #     phi_d,
    # )
    # n = np.array([0, 0, 1])
    # v = np.array([[
    #     np.sin(to) * np.cos(po),
    #     np.sin(to) * np.sin(po),
    #     np.cos(to)
    # ] for to, po in zip(theta_o, phi_o)])
    # l = np.array([[
    #     np.sin(ti) * np.cos(pi),
    #     np.sin(ti) * np.sin(pi),
    #     np.cos(ti)
    # ] for ti, pi in zip(theta_i, phi_i)])
    # epsilon = 1e-10
    # sample_weights = np.clip(np.sin(theta_h) * np.sin(theta_d)
    #                          * np.sqrt(np.clip(theta_h, epsilon, None)), epsilon, None)
    # sample_weights /= np.average(sample_weights)

    # print(measured.shape, l.shape)

    # Load entire dataset
    downsample_factor = 169
    num_wavelengths = merl_data.shape[-1]
    measured = merl_data.reshape(-1, num_wavelengths)
    measured = measured[0::downsample_factor, :]

    phi_h_val = 0
    theta_h, _, theta_d, phi_d = merl.generate_dense_half_diffs(phi_h_val)
    theta_h, theta_d, phi_d = np.meshgrid(
        theta_h,
        theta_d,
        phi_d,
        indexing='ij'
    )
    theta_h, theta_d, phi_d = (
        theta_h.ravel(),
        theta_d.ravel(),
        phi_d.ravel()
    )
    theta_h, theta_d, phi_d = (
        theta_h[0::downsample_factor],
        theta_d[0::downsample_factor],
        phi_d[0::downsample_factor]
    )
    theta_o, phi_o, theta_i, phi_i = merl.half_diff_to_std_coords(
        theta_h, np.broadcast_to(phi_h_val, theta_h.shape), theta_d, phi_d
    )

    n = np.array([[0, 0, 1]])
    v = np.array([[
        np.sin(to) * np.cos(po),
        np.sin(to) * np.sin(po),
        np.cos(to)
    ] for to, po in zip(theta_o, phi_o)])
    l = np.array([[
        np.sin(ti) * np.cos(pi),
        np.sin(ti) * np.sin(pi),
        np.cos(ti)
    ] for ti, pi in zip(theta_i, phi_i)])

    # reverse density of merl measurement mappings
    epsilon = 1e-10
    # sperical coordinate jacobian
    sample_weights = np.sin(theta_h) * np.sin(theta_d)
    # non-linear theta_h mappnig
    sample_weights *= np.sqrt(np.clip(theta_h, epsilon, None))
    # sample_weights *= np.cos(theta_o) # chance of hitting a surface is cos(V), V viewing direction
    # sample_weights = np.where((theta_o > (75.0/90.0 * np.pi/2)) & (theta_i > (75.0/90.0 * np.pi/2)), 0, sample_weights)
    sample_weights /= np.average(sample_weights)
    sample_weights = sample_weights[:,np.newaxis]

    # nudging the garbage collector to get rid of these large tmp variables
    theta_h, phi_h, theta_d, phi_d = None, None, None, None
    theta_o, phi_o, theta_i, phi_i = None, None, None, None

    # # purely random sampling
    # NUM_SAMPLES = 10000
    # n = np.array([0, 0, 1])
    # uv = np.random.rand(2, NUM_SAMPLES)
    # ti, pi = np.arccos(0.85 * uv[1]), uv[0] * 2 * np.pi
    # l = np.stack((np.sin(ti) * np.cos(pi), np.sin(ti)
    #               * np.sin(pi), np.cos(ti)), axis=-1)
    # uv = np.random.rand(2, NUM_SAMPLES)
    # to, po = np.arccos(0.85 * uv[1]), uv[0] * 2 * np.pi
    # v = np.stack((np.sin(to) * np.cos(po), np.sin(to)
    #               * np.sin(po), np.cos(to)), axis=-1)
    # sample_weights = np.broadcast_to(1, to.shape)
    # measured = merl.lookup_brdf_val_vectorized(merl_data, to, po, ti, pi)

    """
    Set material model and generate numpy lambda
    """

    brdf = glTF_brdf(**kwargs)
    guess = brdf.first_guess
    limits = list(brdf.bounds.values())
    brdf_np = brdf.get_np(gradient=analytic_gradient)
    # brdf_der_np = brdf.derivative_np()

    def model(*params):
        if analytic_gradient:
            y, ydx = brdf_np(v, n, l, *params)
            loss = np.mean(sample_weights * (y - measured) ** 2)
            diff = y - measured
            weighted_diff = 2 * sample_weights * diff
            gradient = np.mean(ydx * weighted_diff[:, None], axis=0)  # shape (num_params,)
            return loss,
        else:
            y = brdf_np(v, n, l, *params)
            assert (y.shape == measured.shape), f"expected: {measured.shape}, got: {y.shape}"
            # loss = np.mean((y - measured) ** 2)
            # loss = np.mean(np.abs(y - measured))
            loss = np.mean(sample_weights * (y - measured) ** 2)
            # loss = np.mean((y - measured) ** 2 / (measured + 0.01))
            return loss

    result, optimal_params, model_output = fit_bsdf(
        list(guess.values()),
        limits,
        model,
        jac=analytic_gradient
    )

    if result.success == False:
        print(f"Failed fitting material: {material}")

    param_dict = {name: value for name,
                  value in zip(guess.keys(), optimal_params)}

    # print(
    #     f"Params: {param_dict}\n Loss: {result.fun}"
    # )
    # print(
    #     f"Mean and Maximum percentage error are: {100 * MeanRelE(model_output, measured):.3f}% and {100 * MaxRelE(model_output, measured):.3f}%"
    # )
    # print()

    return param_dict, result.fun


def fit_all_merl_materials(dir, analytic_gradient=False, **kwargs):

    import tqdm
    materials = merl.get_merl_material_list(dir)
    results_dict = {
        material: fit_merl_brdf(
            material,
            dir,
            analytic_gradient=analytic_gradient,
            **kwargs
        )
        for material in tqdm.tqdm(materials)
    }

    brdf = glTF_brdf(**kwargs)

    materials_dict = {
        "materials": [brdf.to_json(material, param_dict) for material, (param_dict, _) in results_dict.items()]
    }
    loss_dict = {
        material: loss for material, (_, loss) in results_dict.items()
    }

    return json.dumps(materials_dict, indent=4, sort_keys=True), loss_dict


# print(fit_all_merl_materials("merl100/brdfs/"))
