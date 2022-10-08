#!/usr/bin/env python
import numpy as np
from scipy import ndimage, interpolate
import itertools

def calc_datacor(
    data1,
    data2,
    mode="cube",
    method="ZNCC",
    threshold1=0,
    threshold2=0,
#    range1=[],
#    range2=[],
#    range3=[],
    ranges=[],
    axes=[],
    #ax1=None,
    #ax2=None,
    #ax3=None,
    #axn1=None,
    norm1=False,
    norm2=False,
    interp_option={},
):
    ## Set axes

    axes_data = data1.get_axes()
    axes_newgrid = []
    for _ax_data, _range, _ax_user in itertools.zip_longest(axes_data, ranges, axes):
        if _range is None:
            continue
        cond = (_range[0] < _ax_data) & (_ax_data < _range[1])
        newax = _ax_data[cond] if _ax_user is None else _ax_user
        axes_newgrid.append(newax)

    """
    if mode == "cube":
        ax1 = data1.xau if ax1 is None else ax1
        ax2 = data1.yau if ax2 is None else ax2
        ax3 = data1.vkms if ax3 is None else ax3
        axes = [ax1, ax2, ax3]
    elif mode == "image":
        ax1 = data1.xau if ax1 is None else ax1
        ax2 = data1.yau if ax2 is None else ax2
        axes = [ax1, ax2]
    elif mode == "pv":
        ax1 = data1.xau if ax1 is None else ax1
        ax2 = data1.vkms if ax2 is None else ax2
        axes = [ax1, ax2]
    else:
        raise Exception("Unknown mode. You can inprement own data tructure mode.")
    """

    ## Trim axes
    """
    _axes = []
    for _ax, _range in zip([ax1, ax2, ax3], [range1, range2, range3]):
        if _range is None:
            continue
        _ax = _ax[(_range[0] < _ax) & (_ax < _range[1])]
    """

    #newgrid = np.stack(np.meshgrid(*axes), axis=-1)
    newgrid = np.stack(np.meshgrid(*axes_newgrid), axis=-1)
    _interp_option = {"bounds_error":False, "fill_value":0}
    _interp_option.update(interp_option)

    ## Make data
    im1 = interpolate.interpn(data1.get_axes(), data1.get_I(), newgrid, **_interp_option)
    im2 = interpolate.interpn(data2.get_axes(), data2.get_I(), newgrid, **_interp_option)

    #ifunc_Ipv1 = interpolate.interpn((cube1.xau, cube1.yau, cube1.vkms), cube1.Ippv, )
    #ifunc_Ipv1 = interpolate.RectBivariateSpline(pv1.vkms, pv1.xau, pv1.Ipv)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.vkms, pv2.xau, pv2.Ipv)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.xau, pv2.vkms, pv2.Ipv)
    #interped_Ipv1 = ifunc_Ipv1(vkms, xau)
    #interped_Ipv2 = ifunc_Ipv2(vkms, xau)
    #interped_Ipv1 = ifunc_Ipv1(xau, vkms)
    #interped_Ipv2 = ifunc_Ipv2(xau, vkms)


    if norm1:
        im1 /= np.max(im1)

    if norm2:
        im2 /= np.max(im2)

    if threshold1 is not None:
        im1 = np.where(im1 > threshold1, im1, 0)

    if threshold2 is not None:
        im2 = np.where(im2 > threshold2, im2, 0)

    if method == "ZNCC":
        res = calc_ZNCC_data(im1, im2)
    elif method == "SSD":
        res = calc_SSD_data(im1, im2)

    #res = calc_correlation(
    #    interped_Ippv1,
    #    interped_Ippv2,
    #    method=method,
    #    threshold=threshold,
    #    with_noise=with_noise,
    #)

    if np.isnan(res):
        raise Exception("Nan!")

    return res

def calc_ZNCC_data(im1, im2):
    indmax1 = np.array( im1.shape )
    Vpix = np.prod(indmax1)
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1 ** 2)
    sumBsq = np.sum(im2 ** 2)
    ZNCC_AB = (
        ( Vpix * sumAB - sumA * sumB)
        / np.sqrt( (Vpix * sumAsq - sumA**2) * (Vpix * sumBsq - sumB**2) )
    )
    return ZNCC_AB

def calc_SSD_data(im1, im2):
    return np.sum((im1 - im2) ** 2)


###########################################################################
####                          LEGACY                                   ####
###########################################################################

# For PV

def calc_pv_correlation(
    pv1,
    pv2,
    method="ZNCC",
    threshold1=None,
    threshold2=None,
    with_noise=False,
    range_xau=[],
    range_vkms=[],
    xau=None,
    vkms=None,
):
    xau = pv1.xau if xau is None else xau
    vkms = pv1.vkms if vkms is None else vkms
    # trim
    if range_xau:
        xau = xau[(range_xau[0] < xau) & (xau < range_xau[1])]
    if range_vkms:
        vkms = vkms[(range_vkms[0] < vkms) & (vkms < range_vkms[1])]

    #ifunc_Ipv1 = interpolate.RectBivariateSpline(pv1.vkms, pv1.xau, pv1.Ipv)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.vkms, pv2.xau, pv2.Ipv)
    ifunc_Ipv1 = interpolate.RectBivariateSpline(pv1.xau, pv1.vkms, pv1.Ipv)
    ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.xau, pv2.vkms, pv2.Ipv)

    #interped_Ipv1 = ifunc_Ipv1(vkms, xau)
    #interped_Ipv2 = ifunc_Ipv2(vkms, xau)
    interped_Ipv1 = ifunc_Ipv1(xau, vkms)
    interped_Ipv2 = ifunc_Ipv2(xau, vkms)

    res = calc_correlation(
        interped_Ipv1,
        interped_Ipv2,
        method=method,
        threshold1=threshold1,
        threshold2=threshold2,
        with_noise=with_noise,
    )

    return res

def calc_correlation(im1, im2_original, method="ZNCC", threshold1=None, threshold2=None, with_noise=False):
    imax1, jmax1 = im1.shape
    imax2, jmax2 = im2_original.shape
    im2 = ndimage.zoom(im2_original, (imax1 / imax2, jmax1 / jmax2), order=1)
    im1 = np.where(im1 > 0, im1, 0)
    im2 = np.where(im2 > 0, im2, 0)
    if with_noise:
        noise_level = 0.33
        im1 += np.random.uniform(-noise_level, noise_level, size=im1.shape)
    # Noise cut
    if threshold1 is not None:
        im1 = np.where(im1 > threshold1, im1, 0)
    if threshold2 is not None:
        im2 = np.where(im2 > threshold2, im2, 0)
    im1 = np.where(im1 / np.max(im1) > 0.00, im1, 0)
    im2 = np.where(im2 / np.max(im2) > 0.00, im2, 0)
    if method == "ZNCC":
        return calc_ZNCC(im1, im2)
    elif method == "SSD":
        return calc_SSD(im1, im2)
    else:
        raise Exception("Unknown method type.")


def calc_SSD(im1, im2):
    return np.sum((im1 - im2) ** 2)


def calc_ZNCC(im1, im2):
    imax1, jmax1 = im1.shape
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1 ** 2)
    sumBsq = np.sum(im2 ** 2)
    # print(imax1, jmax1, sumAB, sumA, sumB)
    ZNCC_AB = (
        (imax1 * jmax1 * sumAB - sumA * sumB)
        / np.sqrt(imax1 * jmax1 * sumAsq - sumA ** 2)
        / np.sqrt(imax1 * jmax1 * sumBsq - sumB ** 2)
    )
    return ZNCC_AB

# For Cube

def calc_cube_correlation(
    cube1,
    cube2,
    method="ZNCC",
    threshold1=0,
    threshold2=0,
    with_noise=False,
    range_xau=[],
    range_yau=[],
    range_vkms=[],
    xau=None,
    yau=None,
    vkms=None,
    normalize1=False,
    normalize2=False,
):
    xau = cube1.xau if xau is None else xau
    yau = cube1.yau if yau is None else yau
    vkms = cube1.vkms if vkms is None else vkms
    # trim
    if range_xau:
        xau = xau[(range_xau[0] < xau) & (xau < range_xau[1])]
    if range_yau:
        yau = yau[(range_yau[0] < yau) & (yau < range_yau[1])]
    if range_vkms:
        vkms = vkms[(range_vkms[0] < vkms) & (vkms < range_vkms[1])]

    #ifunc_Ipv1 = interpolate.RectBivariateSpline(pv1.vkms, pv1.xau, pv1.Ipv)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.vkms, pv2.xau, pv2.Ipv)

    newgrid = np.stack(np.meshgrid(xau, yau, vkms), axis=-1)

    #ifunc_Ipv1 = interpolate.interpn((cube1.xau, cube1.yau, cube1.vkms), cube1.Ippv, )
    im1 = interpolate.interpn((cube1.xau, cube1.yau, cube1.vkms), cube1.Ippv, newgrid)
    im2 = interpolate.interpn((cube2.xau, cube2.yau, cube2.vkms), cube2.Ippv, newgrid)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.xau, pv2.vkms, pv2.Ipv)
    #interped_Ipv1 = ifunc_Ipv1(vkms, xau)
    #interped_Ipv2 = ifunc_Ipv2(vkms, xau)
    #interped_Ipv1 = ifunc_Ipv1(xau, vkms)
    #interped_Ipv2 = ifunc_Ipv2(xau, vkms)

    if normalize1 is not None:
        im1 /= np.max(im1)

    if normalize2 is not None:
        im2 /= np.max(im2)

    if threshold1 is not None:
        im1 = np.where(im1 > threshold1, im1, 0)

    if threshold2 is not None:
        im2 = np.where(im2 > threshold2, im2, 0)
#    im1 = np.where(im1 / np.max(im1) > 0.00, im1, 0)
#    im2 = np.where(im2 / np.max(im2) > 0.00, im2, 0)

    res = calc_ZNCC_cube(im1, im2)
    #res = calc_correlation(
    #    interped_Ippv1,
    #    interped_Ippv2,
    #    method=method,
    #    threshold=threshold,
    #    with_noise=with_noise,
    #)

    return res


def calc_ZNCC_cube(im1, im2):
    imax1, jmax1, kmax1 = im1.shape
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1 ** 2)
    sumBsq = np.sum(im2 ** 2)
    ZNCC_AB = (
        (imax1 * jmax1 * kmax1 * sumAB - sumA * sumB)
        / np.sqrt(imax1 * jmax1 * kmax1 * sumAsq - sumA ** 2)
        / np.sqrt(imax1 * jmax1 * kmax1 * sumBsq - sumB ** 2)
    )
    return ZNCC_AB

# For nD Data

def calc_cube_correlation(
    data1,
    data2,
    axes, # (ax1, ax2, ax3, ..., axn)
    axranges, # ((ax1_in, ax1_out), ..., (axn_in, axn_out))
    method="ZNCC",
    threshold1=0,
    threshold2=0,
    with_noise=False,
    range_xau=[],
    range_yau=[],
    range_vkms=[],
    normalize1=False,
    normalize2=False,
):
    # triming
    axes = [ _ax[(_axrange[0] < _ax) & (_ax < _axrange[1])] for _ax, _axrange in zip(axes, axranges) ]

    #if range_xau:
    #    xau = xau[(range_xau[0] < xau) & (xau < range_xau[1])]
    ##if range_yau:
    #    yau = yau[(range_yau[0] < yau) & (yau < range_yau[1])]
    #if range_vkms:
    #    vkms = vkms[(range_vkms[0] < vkms) & (vkms < range_vkms[1])]

    #ifunc_Ipv1 = interpolate.RectBivariateSpline(pv1.vkms, pv1.xau, pv1.Ipv)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.vkms, pv2.xau, pv2.Ipv)

    newgrid = np.stack(np.meshgrid(xau, yau, vkms), axis=-1)

    #ifunc_Ipv1 = interpolate.interpn((cube1.xau, cube1.yau, cube1.vkms), cube1.Ippv, )
    im1 = interpolate.interpn((cube1.xau, cube1.yau, cube1.vkms), cube1.Ippv, newgrid)
    im2 = interpolate.interpn((cube2.xau, cube2.yau, cube2.vkms), cube2.Ippv, newgrid)
    #ifunc_Ipv2 = interpolate.RectBivariateSpline(pv2.xau, pv2.vkms, pv2.Ipv)
    #interped_Ipv1 = ifunc_Ipv1(vkms, xau)
    #interped_Ipv2 = ifunc_Ipv2(vkms, xau)
    #interped_Ipv1 = ifunc_Ipv1(xau, vkms)
    #interped_Ipv2 = ifunc_Ipv2(xau, vkms)

    if normalize1 is not None:
        im1 /= np.max(im1)

    if normalize2 is not None:
        im2 /= np.max(im2)

    if threshold1 is not None:
        im1 = np.where(im1 > threshold1, im1, 0)

    if threshold2 is not None:
        im2 = np.where(im2 > threshold2, im2, 0)
#    im1 = np.where(im1 / np.max(im1) > 0.00, im1, 0)
#    im2 = np.where(im2 / np.max(im2) > 0.00, im2, 0)

    res = calc_ZNCC_cube(im1, im2)
    #res = calc_correlation(
    #    interped_Ippv1,
    #    interped_Ippv2,
    #    method=method,
    #    threshold=threshold,
    #    with_noise=with_noise,
    #)

    return res


def calc_ZNCC_cube(im1, im2):
    imax1, jmax1, kmax1 = im1.shape
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1 ** 2)
    sumBsq = np.sum(im2 ** 2)
    ZNCC_AB = (
        (imax1 * jmax1 * kmax1 * sumAB - sumA * sumB)
        / np.sqrt(imax1 * jmax1 * kmax1 * sumAsq - sumA ** 2)
        / np.sqrt(imax1 * jmax1 * kmax1 * sumBsq - sumB ** 2)
    )
    return ZNCC_AB

