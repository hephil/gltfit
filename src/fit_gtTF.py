import json
import merl
import matplotlib.pyplot as plt
import os
import glob
from bsdf import *
from glTF import *
from fit_bsdf import *


def fit_merl_brdf(material, dir="../merl100/brdfs/"):
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
    #     0 * np.ones_like(theta_h),
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
    downsample_factor = 69
    num_wavelengths = merl_data.shape[0]
    measured = merl_data.reshape(num_wavelengths, -1)
    measured = measured[:, 0::downsample_factor]

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
        theta_h, phi_h_val * np.ones_like(theta_h), theta_d, phi_d
    )

    n = np.array([0, 0, 1])
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
    sample_weights = np.clip(np.sin(theta_h) * np.sin(theta_d)
                             * np.sqrt(np.clip(theta_h, epsilon, None)), epsilon, None)
    sample_weights /= np.average(sample_weights)

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
    # sample_weights = np.ones_like(to)
    # measured = merl.lookup_brdf_val_vectorized(merl_data, to, po, ti, pi)

    """
    Set material model and generate numpy lambda
    """

    brdf = glTF_brdf(False)
    guess = brdf.first_guess
    limits = list(brdf.bounds.values())
    brdf_np = brdf.get_np()
    brdf_der_np = brdf.derivative_np()

    def model(*params):
        a = brdf_np(v, n, l, *params)
        # b = brdf_der_np(v, n, l, *params)
        # print(b.shape)
        assert (a.shape == measured.shape)
        # loss = np.mean((a - measured) ** 2)
        # loss = np.mean(np.abs(a - measured))
        loss = np.mean(sample_weights * (a - measured) ** 2)
        # loss = np.mean((a - measured) ** 2 / (measured + 0.01))
        return loss

    def model_der(*params):
        a = brdf_np(v, n, l, *params)
        b = brdf_der_np(v, n, l, *params)
        residual = a - measured
        grad = [2 * (residual * f).mean(axis=0) for f in b]
        return disentangle_gradients(grad, layout)

    result, optimal_params, model_output = fit_bsdf(
        list(guess.values()),
        limits,
        model,
        model_der=None
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

    return param_dict


def fit_all_merl_materials(dir, KHR_materials_ior=False):

    import tqdm
    materials = merl.get_merl_material_list(dir)
    param_dicts = {material: fit_merl_brdf(
        material, dir) for material in tqdm.tqdm(materials)}

    brdf = glTF_brdf(KHR_materials_ior=KHR_materials_ior)

    materials_dict = {
        "materials": [brdf.to_json(material, param_dict) for material, param_dict in param_dicts.items()]
    }

    return json.dumps(materials_dict, indent=4, sort_keys=True)


# print(fit_all_merl_materials("merl100/brdfs/"))
