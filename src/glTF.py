import json
import sympy as sp

from bsdf import *

from sympy.utilities.lambdify import implemented_function

# seemingly lowest roughness that does not lead to NaNs and Infs in numpy evaluations
MIN_ROUGHNESS = 0.002


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


def normalize(a):
    return a / a.magnitude()


def half_vector(v, n, l):
    return normalize(v + l)


def diffuse_component(base_color):
    """
    Lambertian brdf component
    """
    return (1/sp.pi) * base_color


def diffuse_brdf(v, n, l, base_color):
    """
    Lambertian brdf restricted to upper hemisphere
    """
    import sympy.vector as spvec
    ndotv = spvec.dot(v, n)
    ndotl = spvec.dot(n, l)
    return sp.Piecewise(
        (diffuse_component(base_color), ((ndotv > 0) & (ndotl > 0))),
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


def specular_brdf(v, n, l, h, alpha):
    """
    Trowbridge-Reitz (GGX) brdf restricted to domain
    """
    nh = spvec.dot(n, h)
    nv = spvec.dot(n, v)
    hv = spvec.dot(h, v)
    nl = spvec.dot(n, l)
    hl = spvec.dot(h, l)
    D_GGX = specular_D_GGX(n, h, alpha)
    V_GGX = specular_V_GGX(v, n, l, alpha)
    GGX = specular_component(D_GGX, V_GGX)

    return sp.Piecewise((GGX, (((hv > 0) & (nv > 0)) | ((hv < 0) & (nv < 0))) & (((hl > 0) & (nl > 0)) | ((hl < 0) & (nl < 0))) & (nh > 0)), (0, True))


def conductor_fresnel(v, h, f0, bsdf):
    """
    Schlick's approximation for Fresnel reflections on conductive materials
    """
    VdotH = spvec.dot(v, h)
    return bsdf * (f0 + (1 - f0) * (1 - abs(VdotH)) ** 5)


def fresnel_mix(v, h, ior, base, layer):
    """
    Schlick's approximation for Fresnel reflections on dielectric materials.
    Mixes a base bsdf according to transmission f * layer +  (1-fr) * base
    """
    VdotH = spvec.dot(v, h)
    f0 = ((1-ior)/(1+ior)) ** 2
    fr = f0 + (1 - f0)*(1 - abs(VdotH)) ** 5
    return mix(base, layer, fr)


def full_fresnel_mix(v, h, ior, base, layer):
    """
    Full dielectric fresnel equation for dielectric materials.
    Mixes a base bsdf according to transmission f * layer +  (1-fr) * base
    """
    VdotH = spvec.dot(v, h)
    sin_w = sp.sqrt(1 - VdotH**2)
    esin = sin_w / ior
    Rs = (
        (VdotH - ior * sp.sqrt(1 - esin**2))
        / (VdotH + ior * sp.sqrt(1 - esin**2))
    ) ** 2
    Rp = (
        (sp.sqrt(1 - esin**2) - ior * VdotH)
        / (sp.sqrt(1 - esin**2) + ior * VdotH)
    ) ** 2
    return mix(base, layer, (Rs + Rp) / 2)


# def fresnel_mix_specular(v, h, f0_color, ior, weight, base, layer):
#     """
#     KHR_materials_specular version of fresnel_mix with 2 additional parameters
#     to weigh and color the highlight.
#     """
#     def max_value(color):
#         return sp.Max(color.r, color.g, color.b)

#     VdotH = spvec.dot(v, h)
#     f0 = ((1-ior)/(1+ior)) ^ 2 * f0_color
#     f0 = min(f0, 1.0)
#     fr = f0 + (1 - f0)*(1 - abs(VdotH)) ^ 5
#     return (1 - weight * max_value(fr)) * base + weight * fr * layer


def ggx_D(n, h, alpha: float):
    ndoth = spvec.dot(n, h)
    return (alpha * alpha) / (sp.pi * (1 + (alpha * alpha) * (ndoth * ndoth) - (ndoth * ndoth)) ** 2)
    return sp.Piecewise(((alpha * alpha) / (sp.pi * (1 + (alpha * alpha) * (ndoth * ndoth) - (ndoth * ndoth)) ** 2), ndoth > 0), (0, True))


def ggx_Lambda_smith(ndotw, alpha):
    tanw = sp.sqrt(1 - ndotw**2) / ndotw
    t = sp.sqrt(1 + alpha**2 * tanw**2)
    t = (t - 1) / 2
    return t


def G1(Lambda, ndotw, hdotw):
    return 1 / (1 + Lambda)
    return sp.Piecewise(
        (1 / (1 + Lambda), ((hdotw > 0) & (ndotw > 0)) | ((hdotw < 0) & (ndotw < 0))),
        (0, True),
    )


def get_G1(ndotw, hdotw, alpha):
    return G1(ggx_Lambda_smith(ndotw, alpha), ndotw, hdotw)


def G2_uncorrelated(Gv, Gl):
    return Gv * Gl


def get_G2_uncorrelated(ndotv, hdotv, ndotl, hdotl, alpha):
    return G2_uncorrelated(get_G1(ndotv, hdotv, alpha), get_G1(ndotl, hdotl, alpha))


def macro_irradiance_to_microsurface(ndotl, hdotl):
    """
    factor to transform incident radiance onto the microsurface [1]
    """
    return sp.Abs(hdotl / ndotl)


def micro_radiance_to_macrosurface(ndotv, hdotv):
    """
    transform scattered radiance back to the macrosurface [1]
    """
    return sp.Abs(hdotv / ndotv)


def reflection_jacobian(hdotv):
    """
    absolute determinant of the jacobian matrix of the reflection/thin transmission operator [1]
    see equation 14 in [1]
    """
    return 1 / (4 * sp.Abs(hdotv))


def reflection_adjustment_factor(ndotv, hdotv, ndotl, hdotl):
    """
    combines the factors of the reflection jacobian and microsurface projection factors and makes use of their symmetry to simplify
    """
    cf1 = macro_irradiance_to_microsurface(ndotl, hdotl)
    cf2 = micro_radiance_to_macrosurface(ndotv, hdotv)
    fmr = reflection_jacobian(hdotv)  # equation 11 in [1]
    fms = fmr / sp.Abs(hdotl)  # equation 9 in [1]
    fs = cf1 * cf2 * fms  # equation 8 without FGD in [1]
    return fs.subs(hdotl, hdotv)  # hdotv == hdotl


def gltf(v, n, l, base_color, alpha: float, metallic: float, ior: float = 1.5):
    nv = spvec.dot(n, v)
    nl = spvec.dot(n, l)
    h = half_vector(v, n, l)
    dielectric_brdf = fresnel_mix(v, h, ior, diffuse_component(base_color), specular_component(
        specular_V_GGX(v, n, l, alpha), specular_D_GGX(n, h, alpha)))

    metal_brdf = conductor_fresnel(
        v, h, base_color, specular_component(
            specular_V_GGX(v, n, l, alpha), specular_D_GGX(n, h, alpha)))

    return sp.Piecewise((mix(dielectric_brdf, metal_brdf, metallic), (nv > 0) & (nl > 0)), (0, True))


class glTF_brdf(BSDF):
    def __init__(self, KHR_materials_ior=True, KHR_materials_specular=True):

        self.params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]
        self.code_params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]

        base_color = sp.Symbol("rho", nonnegative=True, real=True)
        alpha = sp.Symbol("alpha", nonnegative=True, real=True)
        metallic = sp.Symbol("m", nonnegative=True, real=True)
        ior = sp.Symbol("ior", nonnegative=True, real=True)

        bsdf_params = [
            base_color,
            alpha,
            metallic,
            ior if KHR_materials_ior else 1.5
        ]
        self.material_params = [
            base_color,
            alpha,
            metallic,
        ]
        self.json_params = {
            base_color: lambda c: {"pbrMetallicRoughness": {"baseColorFactor": [c[0], c[1], c[2], 1]}},
            alpha: lambda a: {"pbrMetallicRoughness": {"roughnessFactor": np.sqrt(a.item())}},
            metallic: lambda m: {"pbrMetallicRoughness": {"metallicFactor": m.item()}},
        }
        self.bounds = {
            base_color: (0, 1),
            alpha: (0, 1),
            metallic: (0, 1)
        }
        self.defaults = {
            base_color: np.array([1, 1, 1]),
            alpha: 1,
            metallic: 1,
        }
        self.first_guess = {
            base_color: np.array([0.5, 0.5, 0.5]),
            alpha: 0.5,
            metallic: 0.5,
        }

        if KHR_materials_ior == True:
            self.material_params.append(ior)
            self.json_params[ior] = lambda eta: {
                "extensions": {"KHR_materials_ior": {"ior": eta.item()}}
            }
            self.bounds[ior] = (0, np.inf)
            self.defaults[ior] = 1.5
            self.first_guess[ior] = 1.5

        self.bsdf = gltf(V, N, L, *bsdf_params)


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
            ior = 1.5
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

            ret_dict[name] = [base_color, alpha, metallic, ior]
        return ret_dict
