#!/usr/bin/env python
import numpy as np
from scipy import ndimage, interpolate


def calc_PV_correlation(
    PV1,
    PV2,
    method="ZNCC",
    threshold=0,
    with_noise=False,
    range_xau=[],
    range_vkms=[],
    xau=None,
    vkms=None,
):
    xau = PV1.xau if xau is None else xau
    vkms = PV1.vkms if vkms is None else vkms
    # trim
    if range_xau:
        xau = xau[(range_xau[0] < xau) & (xau < range_xau[1])]
    if range_vkms:
        vkms = vkms[(range_vkms[0] < vkms) & (vkms < range_vkms[1])]

    ifunc_Ipv1 = interpolate.RectBivariateSpline(PV1.vkms, PV1.xau, PV1.Ipv)
    ifunc_Ipv2 = interpolate.RectBivariateSpline(PV2.vkms, PV2.xau, PV2.Ipv)

    interped_Ipv1 = ifunc_Ipv1(vkms, xau)
    interped_Ipv2 = ifunc_Ipv2(vkms, xau)

    res = calc_correlation(
        interped_Ipv1,
        interped_Ipv2,
        method=method,
        threshold=threshold,
        with_noise=with_noise,
    )

    return res


def calc_correlation(im1, im2_original, method="ZNCC", threshold=0, with_noise=False):
    jmax1, imax1 = im1.shape
    jmax2, imax2 = im2_original.shape
    im2 = ndimage.zoom(im2_original, (jmax1 / jmax2, imax1 / imax2), order=1)
    im1 = np.where(im1 > 0, im1, 0)
    im2 = np.where(im2 > 0, im2, 0)
    if with_noise:
        noise_level = 0.33
        im1 += np.random.uniform(-noise_level, noise_level, size=im1.shape)
    # Noise cut
    im1 = np.where(im1 > threshold, im1, 0)
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
