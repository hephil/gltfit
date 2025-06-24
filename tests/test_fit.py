from fit_bsdf import *
from bsdf import *


class test_brdf(BSDF):
    def __init__(self):
        self.params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]
        self.code_params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]
        self.material_params = [
            sp.Symbol("rho", nonnegative=True, real=True),
        ]
        import sympy.vector as spvec
        self.bsdf = sp.Piecewise(
            (sp.Symbol("rho") / np.pi, ((spvec.dot(V, N) > 0) & (spvec.dot(N, L) > 0))),
            (0, True),
        )


def test_trivial_fit():
    NUM_SAMPLES = 1000

    theta_v = 15 / 90.0 * np.pi / 2
    N_val = np.array([0, 0, 1])
    V_val = np.array([np.sin(theta_v), 0, np.cos(theta_v)], dtype=np.float32)

    uv = np.random.rand(2, NUM_SAMPLES)
    phi = uv[0] * 2 * np.pi
    costheta = -1.0 + 2.0 * uv[1]
    theta = np.arccos(costheta)
    L_val = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
        dtype=np.float32
    )

    albedo = np.array([0.15, 0.15, 0.15])
    target_albedo = np.array([0.3, 0.3, 0.3])
    guess = [albedo]

    # masking samples below horizon in target data actually not necessary because model_output is masked and gradient is 0 for those samples
    target = np.array([target_albedo / np.pi if L_val[i, 2] > 0 else np.array([0, 0, 0])
                      for i in range(NUM_SAMPLES)]).swapaxes(0, 1)
    # target = np.array([ target_albedo / np.pi for i in range(NUM_SAMPLES)]).swapaxes(0, 1)

    limits = [(0, 1)]
    brdf_np = test_brdf().get_np()

    def model(albedo):
        model_output = brdf_np(V_val, N_val, L_val, albedo)
        assert (model_output.shape == target.shape)
        loss = np.mean((model_output - target) ** 2)
        return loss

    result, popt, model_output = fit_bsdf(
        guess,
        limits,
        model,
        model_der=None
    )

    assert np.allclose(popt[0], target_albedo)
    assert np.allclose(result.fun, 0)


test_trivial_fit()
