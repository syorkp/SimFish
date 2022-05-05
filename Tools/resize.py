"""My version of skimage resize function available in cupy."""

from Tools.underlying_resize import SimilarityTransform, AffineTransform, ProjectiveTransform, warp

HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


def resize(image, output_shape, chosen_math_library, order=None, mode='reflect', cval=0, clip=True,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape = input_shape + (1,) * (output_ndim - image.ndim)
        image = chosen_math_library.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim - 1:
        raise ValueError("len(output_shape) cannot be smaller than the image "
                         "dimensions")

    if anti_aliasing is None:
        anti_aliasing = not image.dtype == bool

    if image.dtype == bool and anti_aliasing:
        print("Input image dtype is bool. Gaussian convolution is not defined "
             "with bool data type. Please set anti_aliasing to False or "
             "explicitely cast input image to another data type. Starting "
             "from version 0.19 a ValueError will be raised instead of this "
             "warning.")

    factors = (chosen_math_library.asarray(input_shape, dtype=float) /
               chosen_math_library.asarray(output_shape, dtype=float))

    # 2-dimensional interpolation
    if len(output_shape) == 2 or (len(output_shape) == 3 and
                                  output_shape[2] == input_shape[2]):
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = chosen_math_library.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = chosen_math_library.zeros(src_corners.shape, dtype=chosen_math_library.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform(chosen_math_library=chosen_math_library)
            tform.estimate(src_corners, dst_corners, chosen_math_library)

        # Make sure the transform is exactly metric, to ensure fast warping.
        tform.params[2] = chosen_math_library.array([0, 0, 1])
        tform.params[0, 1] = 0
        tform.params[1, 0] = 0

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range, chosen_math_library=chosen_math_library)

    else:  # n-dimensional interpolation
        print("Requires modification...")
        # order = _validate_interpolation_order(image.dtype, order)
        #
        # coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
        #                 for i, d in enumerate(output_shape)]
        #
        # coord_map = np.array(np.meshgrid(*coord_arrays,
        #                                  sparse=False,
        #                                  indexing='ij'))
        #
        # image = convert_to_float(image, preserve_range)
        #
        # ndi_mode = _to_ndimage_mode(mode)
        # out = ndi.map_coordinates(image, coord_map, order=order,
        #                           mode=ndi_mode, cval=cval)
        #
        # _clip_warp_output(image, out, order, mode, cval, clip)

    return out


