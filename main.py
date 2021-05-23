import time

import nibabel as nib
import numpy as np
from PIL import Image


def unopt_sphere(radius, size, n_dims):
    """
    Utility function for creating a sphere like mask in N dimensions

    from: https://stackoverflow.com/a/61235190
    """

    if radius > size:
        raise ValueError("radius cannot be > array size")

    shape_tuple = (size,) * n_dims
    A = np.zeros(shape_tuple)

    # define centre
    # (x0, y0, z0) : coordinates of center of circle inside A. '''
    c0 = int(np.floor(A.shape[0] / 2))
    # x0, y0, z0 = int(np.floor(A.shape[0]/2)), \
    # int(np.floor(A.shape[1]/2)), int(np.floor(A.shape[2]/2))

    # from: https://stackoverflow.com/a/17372925
    indices = np.ndindex(shape_tuple)
    for idx in indices:
        # unroll, centre and sum indices
        idx_sum = sum([(c0 - i) ** 2 for i in idx])
        deb = radius - np.sqrt(idx_sum)

        if deb >= 0:
            A[idx] = 1

    return A


def opt_sphere(radius, size, n_dims):
    """
    Utility function for creating a sphere like mask in N dimensions

    Implementing optimised and generalised to n_dims version of: https://stackoverflow.com/a/17372925

    Making use of numpy's matrix compute, takes more memory but is much faster than original code
    """

    if radius > size:
        raise ValueError("radius cannot be > array size")

    # make grids
    start, end, step = 0, size - 1, size
    grid_builder_one = np.linspace(start, end, step)
    grid_builder = []
    for _ in range(n_dims):
        grid_builder.append(grid_builder_one)
    meshgrids = list(np.meshgrid(*grid_builder))

    # define centre
    c0 = int(np.floor(size / 2))

    # compute circle/sphere equation and make corresponding mask
    idx_sum = np.sqrt(sum([(m - c0) ** 2 for m in meshgrids]))
    A = ((radius - idx_sum) >= 0).astype(np.float32)

    return A


def test_functions():
    radius_to_check = [10, 250, 10, 100]
    size_to_check = [500, 500, 100, 256]
    n_dims_to_check = [2, 2, 3, 3]

    buffer = ""
    for radius, size, n_dims in zip(radius_to_check, size_to_check, n_dims_to_check):
        tic = time.time()
        s1 = unopt_sphere(radius, size, n_dims)
        toc1 = time.time()
        s1_time = toc1 - tic
        s2 = opt_sphere(radius, size, n_dims)
        toc2 = time.time()
        s2_time = toc2 - toc1
        buffer += "nDims: {} | Radius: {} | size: {}\n".format(n_dims, radius, size)
        buffer += "Unopt time: {}\n".format(s1_time)
        buffer += "Opt time: {}\n\n".format(s2_time)
        np.testing.assert_allclose(s1, s2, rtol=1e-5)

    print(buffer)
    with open("data/timing.txt", "w") as fp:
        fp.write(buffer)


def pilsave(data, path):
    Image.fromarray((data * 255).astype(np.uint8)).save(path)


def nibabelsave(data, path):
    nibimage = nib.Nifti1Image(data.astype(np.float64), affine=np.eye(4))
    nib.save(nibimage, path)


def save(data, path):
    if data.ndim == 2:
        path += ".png"
        savefunc = pilsave
    elif data.ndim == 3:
        path += ".nii.gz"
        savefunc = nibabelsave
    else:
        print("cannot save {}d data".format(data.ndim))

    savefunc(data, path)


if __name__ == "__main__":
    # test
    test_functions()

    save(opt_sphere(100, 256, 2), "data/out")
    save(opt_sphere(100, 256, 3), "data/out")
