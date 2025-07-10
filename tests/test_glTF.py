from bsdf import *
from glTF import *


def test_integral():
    brdf = glTF_brdf()
    mparams = {
        base_color_name: np.array([1, 1, 1]),
        roughness_name: 0.6**2,
        metallic_name: 0.5,
    }
    margs = [
        mparams.get(param_name, brdf.defaults[param_name]) for param_name in brdf.material_params
    ]

    theta_v = 15 / 90.0 * np.pi / 2
    N_val = np.array([0, 0, 1], dtype=np.float32)
    V_val = np.array([np.sin(theta_v), 0, np.cos(theta_v)], dtype=np.float32)
    brdf_np = brdf.get_np()

    val = integrate_spherical_function(lambda l: brdf_np(
        V_val, N_val, l, *margs) * np.abs(np_dot(l, N_val)), 100000)

    assert (np.isfinite(val) and val > 0.0 and val < 1.0)


def test_rgb_integral():
    brdf = glTF_brdf()
    mparams = {
        base_color_name: np.array([1, 1, 1]),
        roughness_name: 0.6**2,
        metallic_name: 0.5,
    }
    margs = [
        mparams.get(param_name, brdf.defaults[param_name]) for param_name in brdf.material_params
    ]

    theta_v = 15 / 90.0 * np.pi / 2
    N_val = np.array([0, 0, 1], dtype=np.float32)
    V_val = np.array([np.sin(theta_v), 0, np.cos(theta_v)], dtype=np.float32)
    brdf_np = brdf.get_np()

    val = integrate_spherical_function(lambda l: brdf_np(
        V_val, N_val, l, *margs) * np.abs(np_dot(l, N_val)), 100000)

    assert (np.isscalar(val) and np.isfinite(val) and val > 0.0 and val < 1.0)
