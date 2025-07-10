import numpy as np
import os
import glob

# Constants
BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0
M_PI = np.pi


def get_merl_material_list(dir="../merl100/brdfs/"):
    return [os.path.splitext(os.path.basename(m))[0] for m in glob.glob(os.path.join(dir, "*.binary"))]


def np_normalize(a):
    """ normalize the first dimension of an input vector"""
    l2 = np.sum(a**2, axis=-1, keepdims=True)
    return np.divide(a, np.sqrt(l2, where=l2 > 1e-6), where=l2 > 0.0)


def rotate_vector(vec, axis, angle):
    angle = angle[..., np.newaxis]
    axis = np_normalize(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cross = np.cross(axis, vec, axis=-1)
    dot = np.sum(axis * vec, keepdims=True, axis=-1)
    return (vec * cos_a) + (cross * sin_a) + (axis * dot * (1 - cos_a))


def half_diff_to_std_coords(theta_half, fi_half, theta_diff, fi_diff):
    """
    Convert half/diff coordinates to standard coordinates
    """

    # Step 1: reconstruct half-vector
    half_vec = np.stack([
        np.sin(theta_half) * np.cos(fi_half),
        np.sin(theta_half) * np.sin(fi_half),
        np.cos(theta_half)
    ], axis=-1)
    half_vec = np_normalize(half_vec)

    # Step 2: reconstruct diff-vector in local frame
    diff_vec = np.stack([
        np.sin(theta_diff) * np.cos(fi_diff),
        np.sin(theta_diff) * np.sin(fi_diff),
        np.cos(theta_diff)
    ], axis=-1)

    # Step 3: rotate diff into world coordinates
    normal = np.array([0, 0, 1])
    # bi_normal = np_normalize(np.cross(half_vec, normal, axis=0))
    bi_normal = np.array([0, 1, 0])
    # bi_normal = np.where(half_vec[2] < 0.999, np_normalize(np.cross(half_vec, normal, axis=0)), np.array([[1.0], [0.0], [0.0]]))

    temp = rotate_vector(diff_vec, bi_normal, np.atleast_1d(theta_half))
    in_vec = rotate_vector(temp, normal, np.atleast_1d(fi_half))
    in_vec = np_normalize(in_vec)

    # Step 4: reconstruct out-vector using: out = 2 * half - in
    out_vec = 2 * np.sum(half_vec * in_vec, keepdims=True,
                         axis=-1) * half_vec - in_vec
    out_vec = np_normalize(out_vec)

    # Step 5: convert to spherical coordinates
    theta_in = np.arccos(in_vec[..., 2])
    fi_in = np.arctan2(in_vec[..., 1], in_vec[..., 0])

    theta_out = np.arccos(out_vec[..., 2])
    fi_out = np.arctan2(out_vec[..., 1], out_vec[..., 0])

    return theta_in, fi_in, theta_out, fi_out


def std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out):
    """
    Convert standard coordinates to half/diff coordinates
    """

    # Compute incoming vector
    in_vec = np.stack(
        [
            np.sin(theta_in) * np.cos(phi_in),
            np.sin(theta_in) * np.sin(phi_in),
            np.cos(theta_in),
        ], axis=-1
    )
    in_vec = np_normalize(in_vec)

    # Compute outgoing vector
    out_vec = np.stack(
        [
            np.sin(theta_out) * np.cos(phi_out),
            np.sin(theta_out) * np.sin(phi_out),
            np.cos(theta_out),
        ], axis=-1
    )
    out_vec = np_normalize(out_vec)

    # Compute halfway vector
    half_vec = np_normalize(in_vec + out_vec)

    # Compute theta_half and phi_half
    theta_half = np.arccos(half_vec[..., 2])
    phi_half = np.arctan2(half_vec[..., 1], half_vec[..., 0])

    # Compute diff vector by rotating incoming vector
    normal = np.array([0, 0, 1])
    bi_normal = np.array([0, 1, 0])

    temp = rotate_vector(in_vec, normal, -phi_half)
    diff_vec = rotate_vector(temp, bi_normal, -theta_half)

    # Compute theta_diff and phi_diff
    theta_diff = np.arccos(diff_vec[..., 2])
    phi_diff = np.arctan2(diff_vec[..., 1], diff_vec[..., 0])

    return theta_half, phi_half, theta_diff, phi_diff


def theta_half_index(theta_half):
    """
    Lookup theta_half index
    This is a non-linear mapping!
    In:  [0 .. pi/2]
    Out: [0 .. 89]
    """
    theta_half_deg = (theta_half / (M_PI / 2.0)) * BRDF_SAMPLING_RES_THETA_H
    temp = np.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H)
    ret_val = np.array(temp).astype(np.int32)
    return np.where(theta_half <= 0, 0, np.clip(ret_val, 0, BRDF_SAMPLING_RES_THETA_H - 1))


def theta_diff_index(theta_diff):
    """ Lookup theta_diff index """
    tmp = np.array(theta_diff / (M_PI * 0.5) *
                   BRDF_SAMPLING_RES_THETA_D).astype(np.int32)
    return np.clip(tmp, 0, BRDF_SAMPLING_RES_THETA_D - 1)


def phi_diff_index(phi_diff):
    """
    Lookup phi_diff index  
    In: phi_diff in [0 .. pi]
    Out: tmp in [0 .. 179]
    """
    phi_diff = np.where(phi_diff < 0.0, phi_diff + M_PI, phi_diff)
    tmp = np.array(phi_diff / M_PI *
                   (BRDF_SAMPLING_RES_PHI_D / 2)).astype(np.int32)
    return np.clip(tmp, 0, BRDF_SAMPLING_RES_PHI_D // 2 - 1)


def lookup_brdf_val_vectorized(data, theta_in, phi_in, theta_out, phi_out):
    """ Vectorized BRDF lookup for multiple incoming/outgoing directions """

    theta_half, phi_half, theta_diff, phi_diff = std_coords_to_half_diff_coords(
        theta_in, phi_in, theta_out, phi_out
    )

    # Compute indices (these should return arrays)
    th_h_idx = theta_half_index(theta_half)
    th_d_idx = theta_diff_index(theta_diff)
    ph_d_idx = phi_diff_index(phi_diff)

    # Gather BRDF values and apply scaling
    brdf = data[th_h_idx, th_d_idx, ph_d_idx]

    below_horizon_mask = np.logical_or(
        theta_out >= np.pi / 2, theta_in >= np.pi / 2
    )[..., np.newaxis]

    brdf = np.where(below_horizon_mask, 0, brdf)
    return np.clip(brdf, 0, np.inf)


def generate_dense_half_diffs(phi_h_val):
    theta_h = np.array([(((i + 0.5)**2) * (np.pi / 2.0)) / (BRDF_SAMPLING_RES_THETA_H**2)
                        for i in range(BRDF_SAMPLING_RES_THETA_H)])

    phi_h = np.broadcast_to(phi_h_val, theta_h.shape)

    theta_d = np.array([((i + 0.5) * (np.pi * 0.5) / BRDF_SAMPLING_RES_THETA_D)
                        for i in range(BRDF_SAMPLING_RES_THETA_D)])

    phi_d = np.array([((i + 0.5) * np.pi / (BRDF_SAMPLING_RES_PHI_D / 2))
                      for i in range(BRDF_SAMPLING_RES_PHI_D // 2)])

    return theta_h, phi_h, theta_d, phi_d


def read_merl_brdf(filename):
    try:
        with open(filename, "rb") as f:
            # Read the dimensions
            dims = np.fromfile(f, dtype=np.int32, count=3)
            n = dims[0] * dims[1] * dims[2]

            # Check if dimensions match expected resolution
            expected_n = (
                BRDF_SAMPLING_RES_THETA_H
                * BRDF_SAMPLING_RES_THETA_D
                * (BRDF_SAMPLING_RES_PHI_D // 2)
            )
            if n != expected_n:
                raise ValueError("Dimensions don't match")

            # Read the BRDF data (3 channels for RGB)
            brdf = np.fromfile(f, dtype=np.float64, count=3 * n)

            brdf = brdf.reshape((3, dims[0], dims[1], dims[2]))
            brdf = np.stack(
                [RED_SCALE * brdf[0], GREEN_SCALE * brdf[1], BLUE_SCALE * brdf[2]],
                axis=-1
            )
            return np.clip(brdf, 0, np.inf)
    except Exception as e:
        print(f"Error reading BRDF file: {e}")
        return None


def merl_brdf_eval(v, n, l, merl_data):
    ret = lookup_brdf_val_vectorized(
        merl_data,
        np.arccos(v[..., 2]),
        np.arctan2(v[..., 1], v[..., 0]),
        np.arccos(l[..., 2]),
        np.arctan2(l[..., 1], l[..., 0])
    )
    print(ret.shape)
    return ret
