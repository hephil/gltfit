import json
import sympy as sp
import sympy.vector as spvec

from bsdf import *

from sympy.utilities.lambdify import implemented_function

# seemingly lowest roughness that does not lead to NaNs and Infs in numpy evaluations
MIN_ROUGHNESS = 0.002

C = spvec.CoordSys3D('C')
RGB = spvec.CoordSys3D('RGB')

lx, ly, lz = sp.symbols('lx ly lz', real=True)
vx, vy, vz = sp.symbols('vx vy vz', real=True)
nx, ny, nz = sp.symbols('nx ny nz', real=True)
hx, hy, hz = sp.symbols('hx hy hz', real=True)
bcx, bcy, bcz = sp.symbols('bcx bcy bcz', real=True)
scx, scy, scz = sp.symbols('scx scy scz', real=True)

L_SYM = sp.MatrixSymbol("L", 3, 1)
V_SYM = sp.MatrixSymbol("V", 3, 1)
N_SYM = sp.MatrixSymbol("N", 3, 1)
H_SYM = sp.MatrixSymbol("H", 3, 1)
C_SYM = sp.MatrixSymbol("rho", 3, 1)
M1_SYM = sp.MatrixSymbol("M_1", 3, 1)

L = sp.Array([lx, ly, lz])
V = sp.Array([vx, vy, vz])
N = sp.Array([nx, ny, nz])
H = sp.Array([hx, hy, hz])
BC = sp.Array([bcx, bcy, bcz])
SC = sp.Array([scx, scy, scz])
C1 = sp.Array([1, 1, 1])


def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def normalize(v):
    norm = sp.sqrt(dot(v, v))
    return sp.Array([vi / norm for vi in v])


def half_vector(v, n, l):
    return normalize(v + l)


def mix(a, b, t):
    """
    Linear Interpolation (lerp)
    """
    return (1 - t) * a + t * b


def material(dielectric_brdf, metal_brdf, metallic):
    """
    The BRDF of the metallic-roughness material is a linear interpolation of a metallic BRDF and a dielectric BRDF.
    """
    return mix(dielectric_brdf, metal_brdf, metallic)


def diffuse_component(base_color):
    """
    Lambertian brdf component
    """
    return (1/sp.pi) * base_color


def diffuse_brdf(v, n, l, base_color):
    """
    Lambertian brdf restricted to upper hemisphere
    """
    return sp.Piecewise(
        (diffuse_component(base_color),
         ((spvec.dot(v, n) > 0) & (spvec.dot(n, l) > 0))),
        (0, True),
    )


def specular_D_GGX(n, h, alpha):
    """
    Trowbridge-Reitz microfacet distribution function
    """
    nh = sp.Abs(spvec.dot(n, h))
    return alpha**2 / (sp.pi * (nh**2 * (alpha**2 - 1) + 1)**2)


def specular_V_GGX(v, n, l, alpha):
    """
    Trowbridge-Reitz microfacet V combining microfacet shadowing, masking,
    reflection jacobian, and microfacet<->macro-surface transforms of (ir)radiance
    """
    nv = spvec.dot(n, v)
    nl = spvec.dot(n, l)
    return 1 / (sp.Abs(nl) + sp.sqrt(alpha**2 + (1-alpha**2) * nl**2)) * \
        1 / (sp.Abs(nv) + sp.sqrt(alpha**2 + (1-alpha**2) * nv**2))


def specular_component(V_GGX, D_GGX):
    """
    Trowbridge-Reitz (GGX) brdf component
    """
    return V_GGX * D_GGX


def conductor_fresnel(v, h, f0, bsdf):
    """
    Schlick's approximation for Fresnel reflections on conductive materials
    """
    VdotH = spvec.dot(v, h)
    return bsdf * (f0 + sp.Array([1 - f for f in f0]) * (1 - abs(VdotH)) ** 5)


# def fresnel_mix(v, h, ior, base, layer):
#     """
#     Schlick's approximation for Fresnel reflections on dielectric materials.
#     Mixes a base bsdf according to transmission f * layer + (1-fr) * base
#     """
#     VdotH = spvec.dot(v, h)
#     f0 = ((1-ior)/(1+ior)) ** 2
#     fr = f0 + (1 - f0)*(1 - abs(VdotH)) ** 5
#     return mix(base, layer, fr)


def max_value(color):
    return sp.Max(color[0], sp.Max(color[1], color[2]))


def fresnel_mix(v, h, f0_color, ior, weight, base, layer):
    """
    Schlick's approximation for Fresnel reflections on dielectric materials.
    Mixes a base bsdf according to transmission f * layer + (1-fr) * base.
    Includes weight and f0_color term to implement KHR_materials_specular
    """
    f0 = ((1-ior)/(1+ior)) ** 2 * f0_color
    # need to manually introduce min function here because sympy translates elementwise_min as amin(numpy.asarray....)
    f0 = sp.Array([sp.Function("minimum")(f, 1) for f in f0])
    fr = f0 + (C1 - f0) * (1 - abs(spvec.dot(v, h))) ** 5
    return (1 - weight * max_value(fr)) * base + weight * sp.Array([fr[i] * layer[i] for i in range(len(fr))])


def gltf(v, n, l, base_color, alpha: float, metallic: float, ior: float = 1.5, specular=1, specular_color=C1):
    nv = spvec.dot(n, v)
    nl = spvec.dot(n, l)
    h = half_vector(v, n, l)

    dielectric_brdf = fresnel_mix(
        v, h,
        specular_color,
        ior,
        specular,
        diffuse_component(base_color),
        C1 * specular_component(
            specular_V_GGX(v, n, l, alpha), specular_D_GGX(n, h, alpha)
        )
    )

    metal_brdf = conductor_fresnel(
        v, h, base_color, specular_component(
            specular_V_GGX(v, n, l, alpha), specular_D_GGX(n, h, alpha)
        )
    )

    return sp.Piecewise((mix(dielectric_brdf, metal_brdf, metallic), (nv > 0) & (nl > 0)), (0, True))


class glTF_brdf(BSDF):
    def __init__(self, KHR_materials_ior=False, KHR_materials_specular=False):

        self.KHR_materials_ior = KHR_materials_ior
        self.KHR_materials_specular = KHR_materials_specular

        self.params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]
        # self.params = [V, N, L]
        self.code_params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]
        # self.code_params = [V, N, L]

        self.base_color = BC  # sp.Symbol("rho", nonnegative=True, real=True)
        self.alpha = sp.Symbol("alpha", nonnegative=True, real=True)
        self.metallic = sp.Symbol("m", nonnegative=True, real=True)
        self.ior = sp.Symbol("ior", nonnegative=True, real=True)
        self.specular = sp.Symbol("specular", nonnegative=True, real=True)
        self.specular_color = SC

        self.defaults = {
            "base_color": np.array([1, 1, 1]),
            "alpha": 1,
            "metallic": 1,
            "ior": 1.5,
            "specular": 1.0,
            "specular_color": np.array([1, 1, 1]),
        }
        self.bsdf_params = {
            "base_color": self.base_color,
            "alpha": self.alpha,
            "metallic": self.metallic,
            "ior": self.ior if KHR_materials_ior else self.defaults["ior"],
            "specular": self.specular if KHR_materials_specular else self.defaults["specular"],
            "specular_color": self.specular_color if KHR_materials_specular else C1,
        }

        self.bsdf = gltf(V, N, L, *self.bsdf_params.values())

        # self.bsdf = conductor_fresnel(V, half_vector(V, N, L), base_color, 1)
        # self.bsdf = diffuse_brdf(V, N, L, base_color)

        self.material_params = [
            self.base_color[0],
            self.base_color[1],
            self.base_color[2],
            self.alpha,
            self.metallic,
        ]
        self.bounds = {
            "base_color": (0, 1),
            "alpha": (0, 1),
            "metallic": (0, 1)
        }
        self.first_guess = {
            "base_color": np.array([0.5, 0.5, 0.5]),
            "alpha": 0.5,
            "metallic": 0.5,
        }

        if KHR_materials_ior == True:
            self.material_params.append(self.ior)
            self.bounds["ior"] = (0, 100)
            self.first_guess["ior"] = 1.5

        if KHR_materials_specular == True:
            self.material_params.extend([
                self.specular,
                self.specular_color[0],
                self.specular_color[1],
                self.specular_color[2]
            ])
            self.bounds["specular"] = (0, 1)
            self.bounds["specular_color"] = (0, 1)
            self.first_guess["specular"] = 1
            self.first_guess["specular_color"] = np.array([1, 1, 1])

    def to_json(self, name, params):
        return {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [
                    params['base_color'][0],
                    params['base_color'][1],
                    params['base_color'][2],
                    1,
                ],
                "roughnessFactor": np.sqrt(params['alpha'].item()),
                "metallicFactor": params['metallic'].item(),
            },
            "extensions": {
                **({
                    "KHR_materials_ior": {
                        "ior": params['ior'].item()
                    }
                } if self.KHR_materials_ior else {}
                ),
                **({
                    "KHR_materials_specular": {
                        "specularFactor": params['specular'].item(),
                        "specularColor": list(params['specular_color'])
                    }
                } if self.KHR_materials_specular else {}
                ),
            }
        }


def read_glTF_materials(filename):
    with open(filename) as f:
        j = json.load(f)

        ret_dict = {}

        if "materials" not in j:
            return ret_dict

        mats = j["materials"]
        for mat in mats:
            base_color = np.array([1, 1, 1])
            alpha = 1
            metallic = 1
            ior = None
            specular = None
            specular_color = None
            if "name" in mat:
                name = mat["name"]
            if "pbrMetallicRoughness" in mat:
                pbr = mat["pbrMetallicRoughness"]
                if "baseColorFactor" in pbr:
                    base_color = np.array(
                        [pbr["baseColorFactor"][0], pbr["baseColorFactor"][1], pbr["baseColorFactor"][2]])
                if "metallicFactor" in pbr:
                    metallic = pbr["metallicFactor"]
                if "roughnessFactor" in pbr:
                    alpha = pbr["roughnessFactor"] ** 2
            if "extensions" in mat:
                ext = mat["extensions"]
                if "KHR_materials_ior" in mat:
                    if "ior" in ext["KHR_materials_ior"]:
                        ior = ext["KHR_materials_ior"]["ior"]

            ret_dict[name] = {"base_color": base_color, "alpha": alpha, "metallic": metallic,
                              "ior": ior, "specular": specular, "specular_color": specular_color}
            ret_dict[name] = {k: v for k,
                              v in ret_dict[name].items() if v is not None}

        return ret_dict
