import json
import merl
import matplotlib.pyplot as plt
import os
import glob
from bsdf import *
from glTF import *
from fit_bsdf import *


def get_merl_material_list(dir="../merl100/brdfs/"):
    return [os.path.splitext(os.path.basename(m))[0] for m in glob.glob(os.path.join(dir, "*.binary"))]


def fit_merl_brdf(material, dir="../merl100/brdfs/"):
    """
    Fit the glTF material model to a Merl BRDF
    """

    downsample_factor = 5

    # Load Merl100 Data
    merl_data = merl.read_merl_brdf(os.path.join(dir, f"{material}.binary"))
    num_wavelengths = merl_data.shape[0]
    measured = merl_data[
        :,
        0::downsample_factor,
        0::downsample_factor,
        0::downsample_factor
    ]
    measured = measured.reshape(num_wavelengths, -1)

    phi_h_val = 0
    theta_h, _, theta_d, phi_d = merl.generate_dense_half_diffs(phi_h_val)
    theta_h, theta_d, phi_d = np.meshgrid(
        theta_h[0::downsample_factor],
        theta_d[0::downsample_factor],
        phi_d[0::downsample_factor],
        indexing='ij'
    )
    theta_h, theta_d, phi_d = (
        theta_h.ravel(),
        theta_d.ravel(),
        phi_d.ravel()
    )
    theta_o, phi_o, theta_i, phi_i = merl.half_diff_to_std_coords(
        theta_h, phi_h_val * np.ones_like(theta_h), theta_d, phi_d
    )

    # measured = merl.lookup_brdf_val_vectorized(
    #     merl_data, theta_o, phi_o, theta_i, phi_i)

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

    """
    Set material model and generate numpy lambda
    """

    brdf = glTF_brdf(False)
    guess = brdf.first_guess
    limits = list(brdf.bounds.values())  # [(0, 1), (0, 1), (0, 1)]
    brdf_np = brdf.get_np()
    # brdf_der_np = brdf.get_derivatives()  # brdf.derivative_np()

    def model(*params):
        a = brdf_np(v, n, l, *params)
        assert (a.shape == measured.shape)
        # loss = np.mean((a - measured) ** 2)
        loss = np.sum(sample_weights * (a - measured) ** 2)
        return loss

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

    print(
        f"Params: {param_dict}\n Loss: {result.fun}"
    )
    print(
        f"Mean and Maximum percentage error are: {100 * MeanRelE(model_output, measured):.3f}% and {100 * MaxRelE(model_output, measured):.3f}%"
    )
    print()

    return param_dict


def fit_all_merl_materials(dir, KHR_materials_ior=False):

    import tqdm
    materials = get_merl_material_list(dir)
    param_dicts = {material: fit_merl_brdf(
        material, dir) for material in tqdm.tqdm(materials)}

    brdf = glTF_brdf(KHR_materials_ior=KHR_materials_ior)

    materials_dict = {
        "materials": [
        ]
    }
    for material, param_dict in param_dicts.items():
        material_dict = {
            "name": material,
        }

        def deep_merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_merge(d1[k], v)
                else:
                    d1[k] = v

        for p in brdf.material_params:
            p_dict = brdf.json_params[p](param_dict[p])
            deep_merge(material_dict, p_dict)

        materials_dict["materials"].append(material_dict)

    return json.dumps(materials_dict, indent=4, sort_keys=True)


print(fit_all_merl_materials("merl100/brdfs/"))
