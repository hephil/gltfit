import numpy as np
import merl

""" Rotate Vector """

N = 100


def test_zero_rotation():
    vec = np.random.randn(N, 3)
    axis = np.random.randn(N, 3)
    angle = np.zeros(N)
    rotated = merl.rotate_vector(vec, axis, angle)

    assert np.allclose(rotated, vec)


def test_nd_rotation():
    vec = np.random.randn(N, N, 3)
    axis = np.random.randn(N, N, 3)
    angle = np.zeros((N, N))
    rotated = merl.rotate_vector(vec, axis, angle)

    assert np.allclose(rotated, vec)


def test_right_angle_rotation():
    vec = np.tile(np.array([1, 0, 0]), (N, 1))
    axis = np.tile(np.array([0, 0, 1]), (N, 1))
    angle = np.full(N, np.pi / 2)
    rotated = merl.rotate_vector(vec, axis, angle)
    expected = np.tile(np.array([0, 1, 0]), (N, 1))

    assert np.allclose(rotated, expected, atol=1e-6)

    norms_in = np.linalg.norm(vec, axis=-1)
    norms_out = np.linalg.norm(rotated, axis=-1)

    assert np.allclose(norms_in, norms_out)


def test_inverse_rotation_identity():
    N = 1000
    vec = np.random.randn(N, 3)
    axis = np.random.randn(N, 3)
    angle = np.random.randn(N)
    rotated = merl.rotate_vector(vec, axis, angle)
    rev_rotated = merl.rotate_vector(rotated, axis, -angle)

    assert np.allclose(rev_rotated, vec)


""" Half-Diff Transform """


def test_half_diff_density():
    theta_h, phi_h, theta_d, phi_d = merl.generate_dense_half_diffs(0)

    assert (np.all(merl.theta_half_index(theta_h) == np.array(
        [i for i in range(0, merl.BRDF_SAMPLING_RES_THETA_H)])))
    assert np.all(merl.theta_diff_index(theta_d) == np.array(
        [i for i in range(0, merl.BRDF_SAMPLING_RES_THETA_D)]))
    assert (np.all(merl.phi_diff_index(phi_d) == np.array(
        [i for i in range(0, merl.BRDF_SAMPLING_RES_PHI_D // 2)])))


def test_half_diff_transform():

    downsample_factor = 5

    theta_h, phi_h, theta_d, phi_d = merl.generate_dense_half_diffs(0)
    theta_h, phi_h, theta_d, phi_d = (
        theta_h[0::downsample_factor],
        phi_h[0::downsample_factor],
        theta_d[0::downsample_factor],
        phi_d[0::downsample_factor]
    )
    th, fh, td, fd = np.meshgrid(theta_h, phi_h, theta_d, phi_d, indexing='ij')
    th, fh, td, fd = th.ravel(), fh.ravel(), td.ravel(), fd.ravel()
    ti, fi, to, fo = merl.half_diff_to_std_coords(th, fh, td, fd)
    th2, fh2, td2, fd2 = merl.std_coords_to_half_diff_coords(ti, fi, to, fo)

    assert np.allclose(th, th2)
    assert np.allclose(fh, fh2)
    assert np.allclose(td, td2)
    assert np.allclose(fd, fd2)


def test_brdf_lookup():

    downsample_factor = 1

    th, _, td, fd = merl.generate_dense_half_diffs(0)
    th, td, fd = np.meshgrid(
        th[0::downsample_factor],
        td[0::downsample_factor],
        fd[0::downsample_factor],
        indexing='ij'
    )
    th, td, fd = (
        th.ravel(),
        td.ravel(),
        fd.ravel()
    )
    ti, fi, to, fo = merl.half_diff_to_std_coords(
        th, np.zeros_like(th), td, fd
    )

    merl_data = merl.read_merl_brdf("merl100/brdfs/alum-bronze.binary")
    num_wavelengths = merl_data.shape[-1]
    measured = merl_data[
        0::downsample_factor,
        0::downsample_factor,
        0::downsample_factor,
        :,
    ]
    measured = measured.reshape(-1, num_wavelengths)
    below_horizon_mask = np.logical_or(
        to >= np.pi / 2, ti >= np.pi / 2
    )[..., np.newaxis]
    measured = np.where(below_horizon_mask, 0, measured)

    measured2 = merl.lookup_brdf_val_vectorized(merl_data, ti, fi, to, fo)

    assert measured.shape == measured2.shape
    assert np.allclose(measured, measured2)
