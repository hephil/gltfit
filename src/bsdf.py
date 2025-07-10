# Common functions to reparametrize, plot and test BSDF functions

import sympy as sp
import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go
import sympy.vector as spvec

C = spvec.CoordSys3D('C')

lx, ly, lz = sp.symbols('lx ly lz', real=True)
vx, vy, vz = sp.symbols('vx vy vz', real=True)
nx, ny, nz = sp.symbols('nx ny nz', real=True)
hx, hy, hz = sp.symbols('hx hy hz', real=True)
bcx, bcy, bcz = sp.symbols('bcx bcy bcz', real=True)
scx, scy, scz = sp.symbols('scx scy scz', real=True)

L_SYM = sp.Symbol("L")
V_SYM = sp.Symbol("V")
N_SYM = sp.Symbol("N")
H_SYM = sp.Symbol("H")
C_SYM = sp.Symbol("rho")
M1_SYM = sp.Symbol("M_1")

L = lx*C.i + ly*C.j + lz*C.k
V = vx*C.i + vy*C.j + vz*C.k
N = nx*C.i + ny*C.j + nz*C.k
H = hx*C.i + hy*C.j + hz*C.k


def linear_to_srgb(c):
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        (1.0 + 0.055) * np.power(c, 1.0 / 2.4) - 0.055
    )


def atleast_nd(arr, n):
    arr = np.asarray(arr)
    if arr.ndim >= n:
        return arr
    return arr.reshape((1,) * (n - arr.ndim) + arr.shape)


def np_dot(w1, w2):
    val = np.sum(w1 * w2, keepdims=True, axis=-1)
    return val


def elementwise_min(l, axis=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 2:
        return np.minimum(l[0], l[1])
    else:
        return np.minimum(l[0], elementwise_min(l[1:]))


def elementwise_max(l, axis=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 2:
        return np.maximum(l[0], l[1])
    else:
        return np.maximum(l[0], elementwise_max(l[1:]))


def lambdify_fn(params, sympy_expr):
    """
    lambdifies a given sympy expression while replacing some element-wise functions to make it work with multi-dimensional numpy arrays
    """
    # Sympy just can't generate correct versions of some functions so we wrap the function call with fixed versions overwritten.
    # In newer versions of sympy, we might need to overwrite the array function as well because sympy puts constants and arrays
    # into a list and calls array on them which doesn't work at all mmmhh...

    def np_dot(a, b):
        return np.sum(a * b, axis=-1)

    return sp.lambdify(
        params,
        sympy_expr,
        modules=[{
            "amin": elementwise_min,
            "amax": elementwise_max,
            # "Dot": np_dot,
            'minimum': np.minimum,
        }, "numpy"],
    )

    # from sympy.utilities.autowrap import ufuncify

    # return ufuncify(
    #     params,
    #     sympy_expr
    # )


def homogenize_array(x, N):
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.shape == (1,)):
        return np.full(N, x if np.isscalar(x) else x.item())
    return x


class BSDF:
    params = []
    code_params = []
    bsdf_params = {}
    material_params = {}
    defaults = {}
    first_guess = {}
    bounds = {}
    bsdf = None

    def reparametrize_mat(self, params):
        return params

    def reparametrize_gradients(self, gradients):
        return gradients

    def get_np(self, gradient=False):
        params = list(self.code_params) + list(self.bsdf_params.keys())
        brdf_np = lambdify_fn(params, self.bsdf)

        # import inspect
        # print(inspect.getsource(brdf_np))

        brdf_der = []
        brdf_der_np = []

        if gradient == True:
            brdf_der = [
                sp.Derivative(self.bsdf, p).doit()
                for p in self.bsdf_params
            ]
            brdf_der_np = [
                lambdify_fn(params, der)
                for der in brdf_der
            ]

        def fn(v, n, l, *args):

            v = np.broadcast_to(v, l.shape)
            n = np.broadcast_to(n, l.shape)

            args = self.reparametrize_mat(*args)

            vals = brdf_np(
                v[..., 0:1], v[..., 1:2], v[..., 2:3],
                n[..., 0:1], n[..., 1:2], n[..., 2:3],
                l[..., 0:1], l[..., 1:2], l[..., 2:3],
                *args
            )

            if gradient == True:
                grads = np.array([
                    der_np(
                        v[..., 0:1], v[..., 1:2], v[..., 2:3],
                        n[..., 0:1], n[..., 1:2], n[..., 2:3],
                        l[..., 0:1], l[..., 1:2], l[..., 2:3],
                        *args
                    ) for der_np in brdf_der_np
                ])
                grads = self.reparametrize_gradients(grads)
                ret = (ret, grads)

            return vals

        return fn


def integrate_spherical_function(fun, num_samples=100000):
    from scipy.stats.qmc import Halton
    import numpy as np
    uv = Halton(d=2, scramble=False).random(num_samples).swapaxes(0, 1)
    uv = np.random.rand(2, num_samples)
    phi = uv[0] * 2 * np.pi
    costheta = -1.0 + 2.0 * uv[1]
    theta = np.arccos(costheta)
    L = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
        dtype=np.float32
    )
    vals = fun(L)

    # average of all wavelengths (usually RGB)
    vals = np.atleast_2d(vals)
    # print(f"expected shape: ({num_samples}, {3}), actual: {vals.shape}")
    vals = np.mean(vals, axis=-1)

    if np.any(np.logical_not(np.isfinite(vals))) and np.any(vals < 0):
        print("ERROR in brdf evaluation!")

    return 4 * np.pi * np.sum(vals) / num_samples


def generate_sphere_directions(theta, phi, num1=180, num2=360, axis=0):
    u, v = np.linspace(0, theta, num=num1), np.linspace(0, phi, num=num2)
    u, v = np.meshgrid(u, v)
    return np.stack(
        (np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)), axis=axis
    )


def plot_vector(name, color, v, negative=False):
    vector = go.Scatter3d(
        name=name,
        x=[-v[0], 0] if negative else [0, v[0]],
        y=[-v[1], 0] if negative else [0, v[1]],
        z=[-v[2], 0] if negative else [0, v[2]],
        marker=dict(size=[0, 10], symbol="diamond"),
        line=dict(color=color, width=6),
    )
    annotation = dict(
        showarrow=False,
        x=-v[0] if negative else v[0],
        y=-v[1] if negative else v[1],
        z=-v[2] if negative else v[2],
        text=name,
        textangle=0,
        yshift=10,
        font=dict(color="black", size=12),
    )
    return vector, annotation


def plot_brdf(name, brdf, V_val, normalize=None):
    # num1, num2 = 18, 36
    num1, num2 = 180, 360
    # num1, num2 = 720, 1440
    sphere = generate_sphere_directions(np.pi, 2 * np.pi, num1, num2, -1)
    hemisphere = generate_sphere_directions(
        0.5 * np.pi, 2 * np.pi, 90, 360, -1)
    L_val = sphere.reshape((-1, 3))
    N_val = np.array([0, 0, 1], dtype=np.float32)
    N_vec, N_ann = plot_vector("N", "blue", N_val.flatten())
    V_vec, V_ann = plot_vector("V", "brown", V_val.flatten())

    brdf_vals = brdf(V_val, N_val, L_val)
    brdf_vals = np.atleast_2d(brdf_vals)
    # print(f"expected shape: ({3}, {num1*num2}), actual: {brdf_vals.shape}")

    if normalize == True:
        brdf_vals = brdf_vals / (np.max(brdf_vals) + 0.01)

    # print("brdf_vals: ", brdf_vals.shape)
    brdf_vals = brdf_vals.reshape(num1, num2, 3)
    L_val = L_val.reshape(num1, num2, 3)

    is_valid = np.all(np.isfinite(brdf_vals)) and np.all(brdf_vals >= 0)
    if not is_valid:
        print("ERROR in brdf evaluation!")
        brdf_vals = np.where(np.isfinite(brdf_vals), brdf_vals, 0)

    # if not is_valid:
    #     return

    min_max_val = 100000
    brdf_lum = np.sum(brdf_vals, axis=-1)
    xx = np.clip(brdf_lum * L_val[..., 0], -min_max_val, min_max_val)
    yy = np.clip(brdf_lum * L_val[..., 1], -min_max_val, min_max_val)
    zz = np.clip(brdf_lum * L_val[..., 2], -min_max_val, min_max_val)

    # print(xx.shape)

    # plot = go.Surface(
    #     name=name,
    #     x=xx,
    #     y=yy,
    #     z=zz,
    #     showscale=False,
    #     cmin=-1,
    #     cmax=1,
    #     opacity=0.3,
    #     surfacecolor=brdf_lum,
    # )

    i_list, j_list, k_list = [], [], []

    for r in range(num1 - 1):
        for c in range(num2 - 1):
            # Flattened vertex indices
            v0 = r * num2 + c
            v1 = v0 + 1
            v2 = v0 + num2
            v3 = v2 + 1

            # Triangle 1 (v0, v1, v2)
            i_list.append(v0)
            j_list.append(v1)
            k_list.append(v2)

            # Triangle 2 (v1, v3, v2)
            i_list.append(v1)
            j_list.append(v3)
            k_list.append(v2)

    i, j, k = np.array(i_list), np.array(j_list), np.array(k_list)

    plot = go.Mesh3d(
        name=name,
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        i=i.flatten(),
        j=j.flatten(),
        k=k.flatten(),
        opacity=1,
        vertexcolor=linear_to_srgb(brdf_vals.reshape(-1, 3))
    )

    data = [plot, N_vec, V_vec]
    annotations = [N_ann, V_ann]
    fig = go.Figure(data=data)
    figure3d_width = 600
    figure3d_height = 400
    fig.update_layout(
        width=figure3d_width,
        height=figure3d_height,
        scene_aspectmode="data",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(annotations=annotations),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.7, y=-1.6, z=0.3),
        ),
    )
    fig.show()
    # fig.write_html("file.html") # outputs html with interactive plot
