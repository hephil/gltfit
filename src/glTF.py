import json
import sympy as sp
import sympy.vector as spvec

from bsdf import *

from sympy.utilities.lambdify import implemented_function

# seemingly lowest roughness that does not lead to NaNs and Infs in numpy evaluations
MIN_ROUGHNESS = 0.002


def normalize(v):
    if isinstance(v, spvec.Vector):
        return v.normalize()
    else:
        return sp.Function("normalize")(v)


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


def specular_V_correlated_GGX(v, n, l, alpha):
    """
    Trowbridge-Reitz microfacet V combining microfacet shadowing, masking,
    reflection jacobian, and microfacet<->macro-surface transforms of (ir)radiance
    """
    nv = spvec.dot(n, v)
    nl = spvec.dot(n, l)

    def ggx_Lambda_smith(ndotw, alpha):
        tan2 = (1 - ndotw**2) / ndotw**2
        return 0.5 * (-1 + sp.sqrt(1 + alpha**2 * tan2))

    def G2_correlated(nv, nl, alpha):
        LambdaV = ggx_Lambda_smith(nv, alpha)
        LambdaL = ggx_Lambda_smith(nl, alpha)
        return 1 / (1 + LambdaV + LambdaL)

    G2 = G2_correlated(nv, nl, alpha)
    return G2 / (4 * nv * nl)


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
    return bsdf * (f0 + (1 - f0) * (1 - abs(VdotH)) ** 5)


def fresnel_mix(v, h, f0_color_0, f0_color_1, f0_color_2, ior, weight, base, layer):
    """
    Schlick's approximation for Fresnel reflections on dielectric materials.
    Mixes a base bsdf according to transmission f * layer + (1-fr) * base.
    Includes weight and f0_color term to implement KHR_materials_specular

    There is some trickery involved here: We duplicate the specular color
    (f0_color) three times with three different swizzles because the max
    operation requires recombination of different wavelength's computations
    whereas most other components of a BRDF are independent
    """
    # # non KHR_materials_ior version
    # f0 = 0.04 # f0 = 0.04 for ior=1.5
    # fr = f0 + (1 - f0)*(1 - abs(spvec.dot(v, h))) ** 5
    # return mix(base, layer, fr)

    # # non KHR_materials_specular version
    # f0 = ((1-ior)/(1+ior)) ** 2
    # fr = f0 + (1 - f0)*(1 - abs(spvec.dot(v, h))) ** 5
    # return mix(base, layer, fr)

    def max_value(x, y, z):
        return sp.Max(x, sp.Max(y, z))

    def colored_fr(v, h, f0_color, ior):
        f0 = f0_color * ((1-ior)/(1+ior)) ** 2
        f0 = sp.Min(f0, 1)
        fr = f0 + (1 - f0) * (1 - abs(spvec.dot(v, h))) ** 5
        return fr

    fr_0, fr_1, fr_2 = (
        colored_fr(v, h, f0_color_0, ior),
        colored_fr(v, h, f0_color_1, ior),
        colored_fr(v, h, f0_color_2, ior),
    )
    return (1 - weight * max_value(fr_0, fr_1, fr_2)) * base + weight * fr_0 * layer


def fresnel_coat(v, n, ior, weight, base, layer):
    f0 = ((1-ior)/(1+ior)) ** 2
    fr = f0 + (1 - f0)*(1 - abs(spvec.dot(v, n))) ** 5
    return mix(base, layer, weight * fr)


default_ior = 1.5

base_color_name = 'base_color'
roughness_name = 'alpha'
metallic_name = 'metallic'
ior_name = 'ior'
specular_name = 'specular'
specular_color_name = 'specular_color'
specular_color0_name = 'specular_color0'
specular_color1_name = 'specular_color1'
specular_color2_name = 'specular_color2'
clearcoat_name = "clearcoat"
clearcoat_roughness_name = "clearcoat_alpha"


def gltf(v, n, l, **kwargs):
    base_color = kwargs.get(base_color_name, 1)
    alpha = kwargs.get(roughness_name, 1)
    metallic = kwargs.get(metallic_name, 1)
    ior = kwargs.get(ior_name, default_ior)
    specular = kwargs.get(specular_name, 1)
    specular_color_0 = kwargs.get(specular_color0_name, 1)
    specular_color_1 = kwargs.get(specular_color1_name, 1)
    specular_color_2 = kwargs.get(specular_color2_name, 1)
    clearcoat = kwargs.get(clearcoat_name, 0)
    clearcoat_alpha = kwargs.get(clearcoat_roughness_name, 0)

    nv = spvec.dot(n, v)
    nl = spvec.dot(n, l)
    h = half_vector(v, n, l)

    specular_brdf = specular_component(
        specular_V_GGX(v, n, l, alpha), specular_D_GGX(n, h, alpha)
    )

    dielectric_brdf = fresnel_mix(
        v, h,
        specular_color_0,
        specular_color_1,
        specular_color_2,
        ior,
        specular,
        diffuse_component(base_color),
        specular_brdf
    )

    metal_brdf = conductor_fresnel(
        v, h, base_color, specular_brdf
    )

    material = mix(dielectric_brdf, metal_brdf, metallic)

    # clearcoat
    clearcoat_n = n
    clearcoat_brdf = specular_component(
        specular_V_GGX(v, clearcoat_n, l, clearcoat_alpha), specular_D_GGX(
            clearcoat_n, h, clearcoat_alpha)
    )

    # clearcoat layering
    coated_material = fresnel_coat(
        v,
        clearcoat_n,
        default_ior,
        clearcoat,
        material,
        clearcoat_brdf
    )

    return sp.Piecewise((coated_material, (nv > 0) & (nl > 0)), (0, True))


class glTF_brdf(BSDF):
    def __init__(self, KHR_materials_ior=False, KHR_materials_specular=False, KHR_materials_clearcoat=False):

        self.KHR_materials_ior = KHR_materials_ior
        self.KHR_materials_specular = KHR_materials_specular
        self.KHR_materials_clearcoat = KHR_materials_clearcoat

        self.code_params = [vx, vy, vz, nx, ny, nz, lx, ly, lz]

        bsdf_param_names = [
            base_color_name,
            roughness_name,
            metallic_name,
            *([ior_name] if self.KHR_materials_ior else []),
            *(
                [
                    specular_name,
                    specular_color0_name,
                    specular_color1_name,
                    specular_color2_name,
                ]
                if self.KHR_materials_specular else []
            ),
            *(
                [
                    clearcoat_name,
                    clearcoat_roughness_name
                ]
                if self.KHR_materials_clearcoat else []
            ),
        ]

        self.bsdf_params = {
            name: {
                base_color_name: sp.Symbol(base_color_name, nonnegative=True, real=True),
                roughness_name: sp.Symbol(roughness_name, nonnegative=True, real=True),
                metallic_name: sp.Symbol(metallic_name, nonnegative=True, real=True),
                ior_name: sp.Symbol(ior_name, nonnegative=True, real=True),
                specular_name: sp.Symbol(specular_name, nonnegative=True, real=True),
                specular_color0_name: sp.Symbol(specular_color0_name, nonnegative=True, real=True),
                specular_color1_name: sp.Symbol(specular_color1_name, nonnegative=True, real=True),
                specular_color2_name: sp.Symbol(specular_color2_name, nonnegative=True, real=True),
                clearcoat_name: sp.Symbol(clearcoat_name, nonnegative=True, real=True),
                clearcoat_roughness_name: sp.Symbol(clearcoat_roughness_name, nonnegative=True, real=True),
            }[name] for name in bsdf_param_names
        }

        self.bsdf = gltf(V, N, L, **self.bsdf_params)
        # self.bsdf = self.material_params[base_color_name] * specular_component(
        #     specular_V_GGX(V, N, L, self.material_params[roughness_name]), specular_D_GGX(
        #         N, half_vector(V, N, L), self.material_params[roughness_name])
        # )
        # self.bsdf = conductor_fresnel(V, half_vector(V, N, L), base_color, 1)
        # self.bsdf = diffuse_brdf(V, N, L, self.material_params[base_color_name])

        self.material_params = [
            base_color_name,
            roughness_name,
            metallic_name,
            *([ior_name] if self.KHR_materials_ior else []),
            *(
                [
                    specular_name,
                    specular_color_name,
                ]
                if self.KHR_materials_specular else []
            ),
            *(
                [
                    clearcoat_name,
                    clearcoat_roughness_name,
                ]
                if self.KHR_materials_clearcoat else []
            ),
        ]

        self.defaults = {
            name: {
                base_color_name: np.array([1, 1, 1]),
                roughness_name: 1,
                metallic_name: 1,
                ior_name: default_ior,
                specular_name: 1.0,
                specular_color_name: np.array([1, 1, 1]),
                clearcoat_name: 0,
                clearcoat_roughness_name: 0,
            }[name] for name in self.material_params
        }

        self.first_guess = {
            name: {
                base_color_name: np.array([0.5, 0.5, 0.5]),
                roughness_name: 0.5,
                metallic_name: 0.5,
                ior_name: default_ior,
                specular_name: 0.5,
                specular_color_name: np.array([0.5, 0.5, 0.5]),
                clearcoat_name: 0.5,
                clearcoat_roughness_name: 0.5,
            }[name] for name in self.material_params
        }

        self.bounds = {
            name: {
                base_color_name: (0, 1),
                roughness_name: (0, 1),
                metallic_name: (0, 1),
                ior_name: (0, np.finfo(np.float32).max),
                specular_name: (0, 1),
                specular_color_name: (0, 1),
                clearcoat_name: (0, 1),
                clearcoat_roughness_name: (0, 1),
            }[name] for name in self.material_params
        }

    def reparametrize_mat(self, *args):
        if self.KHR_materials_specular:
            args = list(args)
            assert specular_color_name in self.material_params
            idx = self.material_params.index(specular_color_name)
            # print(args, idx)
            assert idx == list(self.bsdf_params.keys()).index(
                specular_color0_name)

            specular_color = args[idx]
            rgb = specular_color[..., (0, 1, 2)]
            gbr = specular_color[..., (1, 2, 0)]
            brg = specular_color[..., (2, 0, 1)]
            args[idx:idx+1] = [rgb, gbr, brg]
        return args

    def reparametrize_gradients(self, gradients):
        """
        Each wavelength computation receives a differently swizzled version of specular_color.
        Therefore, the gradient for specular_color is split across three inputs:
          wavelength 0 → uses [R, G, B]
          wavelength 1 → uses [G, B, R]
          wavelength 2 → uses [B, R, G]
        We recombine them by summing each channel's total contribution across the swizzled inputs.
        """
        if self.KHR_materials_specular:
            assert specular_color_name in self.bsdf_params.keys()
            idx_rgb = list(self.bsdf_params).index(specular_color0_name)
            idx_gbr = list(self.bsdf_params).index(specular_color1_name)
            idx_brg = list(self.bsdf_params).index(specular_color2_name)
            grad_rgb = gradients[idx_rgb]
            grad_gbr = gradients[idx_gbr]
            grad_brg = gradients[idx_brg]
            # Adding together partial gradients is valid for (most?) functions due to linearity of differentiation.
            # Examples:
            #   - Add: dx(2x) = 2, dx(x+y) + dy(x+y) = 2
            #   - Mul: dx(x^2) = 2x, dx(xy) + dy(xy) = x + y = 2x (if x = y)
            #   - Pow: dx(x^x) = x^x (log(x) + 1)
            #           dx(x^y) + dy(x^y) = y x^(y-1) + x^y log(x) = x^x (log(x) + 1)
            specular_color = np.stack(
                [
                    grad_rgb[..., 0] + grad_gbr[..., 2] + grad_brg[..., 1],
                    grad_rgb[..., 1] + grad_gbr[..., 0] + grad_brg[..., 2],
                    grad_rgb[..., 2] + grad_gbr[..., 1] + grad_brg[..., 0],
                ],
                axis=-1,
            )
            assert idx_rgb == self.material_params.index(specular_color)
            gradients[idx_rgb] = specular_color
            # delete superfluous gradients
            for idx in sorted([idx_gbr, idx_brg], reverse=True):
                gradients.pop(idx)
        return gradients

    def to_json(self, name, params):
        return {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [
                    params[base_color_name][0],
                    params[base_color_name][1],
                    params[base_color_name][2],
                    1,
                ],
                "roughnessFactor": np.sqrt(params[roughness_name].item()),
                "metallicFactor": params[metallic_name].item(),
            },
            "extensions": {
                **({
                    "KHR_materials_ior": {
                        "ior": params[ior_name].item()
                    }
                } if self.KHR_materials_ior else {}),
                **({
                    "KHR_materials_specular": {
                        "specularFactor": params[specular_name].item(),
                        "specularColor": list(params[specular_color_name])
                    }
                } if self.KHR_materials_specular else {}),
                **({
                    "KHR_materials_clearcoat": {
                        "clearcoatFactor": params[clearcoat_name].item(),
                        "clearcoatRoughnessFactor": np.sqrt(params[clearcoat_roughness_name].item())
                    }
                } if self.KHR_materials_clearcoat else {}),
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
            clearcoat = None
            clearcoat_roughness = None
            if "name" in mat:
                name = mat["name"]
            if "pbrMetallicRoughness" in mat:
                pbr = mat["pbrMetallicRoughness"]
                if "baseColorFactor" in pbr:
                    base_color = np.array([
                        pbr["baseColorFactor"][0],
                        pbr["baseColorFactor"][1],
                        pbr["baseColorFactor"][2],
                    ])
                if "metallicFactor" in pbr:
                    metallic = pbr["metallicFactor"]
                if "roughnessFactor" in pbr:
                    alpha = pbr["roughnessFactor"] ** 2
            if "extensions" in mat:
                ext = mat["extensions"]
                if "KHR_materials_ior" in ext:
                    if "ior" in ext["KHR_materials_ior"]:
                        ior = ext["KHR_materials_ior"]["ior"]
                if "KHR_materials_specular" in ext:
                    if "specularFactor" in ext["KHR_materials_specular"]:
                        specular = ext["KHR_materials_specular"]["specularFactor"]
                    if "specularColor" in ext["KHR_materials_specular"]:
                        specular_color = np.array([
                            ext["KHR_materials_specular"]["specularColor"][0],
                            ext["KHR_materials_specular"]["specularColor"][1],
                            ext["KHR_materials_specular"]["specularColor"][2],
                        ])
                if "KHR_materials_clearcoat" in ext:
                    if "clearcoatFactor" in ext["KHR_materials_clearcoat"]:
                        clearcoat = ext["KHR_materials_clearcoat"]["clearcoatFactor"]
                    if "clearcoatRoughnessFactor" in ext["KHR_materials_clearcoat"]:
                        clearcoat_roughness = ext["KHR_materials_clearcoat"]["clearcoatRoughnessFactor"] ** 2

            ret_dict[name] = {
                base_color_name: base_color,
                roughness_name: alpha,
                metallic_name: metallic,
                ior_name: ior,
                specular_name: specular,
                specular_color_name: specular_color,
                clearcoat_name: clearcoat,
                clearcoat_roughness_name: clearcoat_roughness,
            }
            ret_dict[name] = {
                k: v for k, v in ret_dict[name].items() if v is not None
            }

        return ret_dict
