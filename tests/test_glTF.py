from bsdf import *
from glTF import *


def test_integral():
    theta_v = 15 / 90.0 * np.pi / 2
    rho_val = 1
    metallic_val = 0.5
    roughness_val = 0.6

    N_val = np.array([0, 0, 1], dtype=np.float32)
    V_val = np.array([np.sin(theta_v), 0, np.cos(theta_v)], dtype=np.float32)
    brdf = glTF_brdf()
    brdf_np = brdf.get_np()

    val = integrate_spherical_function(lambda l: brdf_np(
        V_val, N_val, l, rho_val, metallic_val, roughness_val**2, 1.5) * np.abs(dot(l, N_val)), 100000)

    assert (np.isfinite(val) and val > 0.0 and val < 1.0)


def test_rgb_integral():
    theta_v = 15 / 90.0 * np.pi / 2
    rho_val = np.array([1, 1, 1])
    metallic_val = 0.5
    roughness_val = 0.6

    N_val = np.array([0, 0, 1], dtype=np.float32)
    V_val = np.array([np.sin(theta_v), 0, np.cos(theta_v)], dtype=np.float32)
    brdf = glTF_brdf()
    brdf_np = brdf.get_np()

    val = integrate_spherical_function(lambda l: brdf_np(
        V_val, N_val, l, rho_val, metallic_val, roughness_val**2, 1.5) * np.abs(dot(l, N_val)), 100000)

    assert (np.isscalar(val) and np.isfinite(val) and val > 0.0 and val < 1.0)
