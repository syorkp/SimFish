"""Uses coarsly gathered SNR data points from environment test to get an estimate of how visibility in the three
channels changes with distance. DEPRECATED"""

import matplotlib.pyplot as plt
import numpy as np


def display_snr(luminance, bk):
    with open(f"LuminanceCalibration/UVDistance-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        distances = np.load(f)

    with open(f"LuminanceCalibration/UVSNR-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        uv = np.load(f)

    with open(f"LuminanceCalibration/RedSNR-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        red = np.load(f)

    with open(f"LuminanceCalibration/Red2SNR-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        red2 = np.load(f)

    uv_ok = np.isfinite(uv)

    uv = uv[uv_ok]
    distances = distances[uv_ok]

    z = np.polyfit(distances, uv, 2)
    p = np.poly1d(z)
    plt.scatter(distances, uv)
    plt.axis([0, 1500, 0, 1.0])
    plt.title(f"Luminance: {luminance}, bk_scatter: {bk}")
    plt.plot(distances, p(distances), color="r")
    plt.show()


def display_distinguishability_score(luminance, bk):
    with open(f"LuminanceCalibration/UVDistance-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        distances = np.load(f)

    with open(f"LuminanceCalibration/Dist-L{luminance}-S0.0006-W1500-BK{bk}.npy", "rb") as f:
        distinguishability = np.load(f)

    distinguishability_ok = np.isfinite(distinguishability)

    distinguishability = distinguishability[distinguishability_ok]
    distances = distances[distinguishability_ok]

    z = np.polyfit(distances, distinguishability, 2)
    p = np.poly1d(z)

    plt.axis([0, 1200, 0, 100])
    plt.scatter(distances, distinguishability)
    plt.title(f"Luminance: {luminance}, bk_scatter: {bk}")
    plt.plot(distances, p(distances), color="r")
    plt.show()


def display_parameter_relationships(background_brightness_values, luminance_values):
    m_values = np.zeros((len(background_brightness_values), len(luminance_values)))
    c_values = np.zeros((len(background_brightness_values), len(luminance_values)))

    for b, bkg in enumerate(background_brightness_values):
        for l, lum in enumerate(luminance_values):
            with open(f"LuminanceCalibration/UVDistance-L{lum}-S0.0006-W1500-BK{bkg}.npy", "rb") as f:
                distances = np.load(f)

            with open(f"LuminanceCalibration/UVSNR-L{lum}-S0.0006-W1500-BK{bkg}.npy", "rb") as f:
                uv = np.load(f)

            uv_ok = np.isfinite(uv)

            uv = uv[uv_ok]
            distances = distances[uv_ok]

            model = np.polyfit(distances, uv, 1)

            m_values[b, l] = model[0]
            c_values[b, l] = model[1]

    plt.plot(background_brightness_values, m_values[:, 0])
    plt.show()

    plt.plot(background_brightness_values, c_values[:, 0])
    plt.show()

    plt.plot(background_brightness_values, m_values[:, 1])
    plt.show()

    plt.plot(background_brightness_values, c_values[:, 1])
    plt.show()

    plt.plot(background_brightness_values, m_values[:, 2])
    plt.show()

    plt.plot(background_brightness_values, c_values[:, 2])
    plt.show()

    plt.plot(background_brightness_values, m_values[:, 3])
    plt.show()

    plt.plot(background_brightness_values, c_values[:, 3])
    plt.show()

display_parameter_relationships([0.0008, 0.0015, 0.002], [1.0, 0.8, 0.4, 0.2])


# display_snr(1.0, 0.005)
# display_snr(0.5, 0.005)
# display_distinguishability_score(1.0, 0.005)
# display_distinguishability_score(0.5, 0.005)


# THESE
display_snr(1.0, 0.002)
display_snr(0.8, 0.002)
display_snr(0.4, 0.002)
display_snr(0.2, 0.002)
display_distinguishability_score(1.0, 0.002)
display_distinguishability_score(0.8, 0.002)
display_distinguishability_score(0.4, 0.002)
display_distinguishability_score(0.2, 0.002)

display_snr(1.0, 0.0015)
display_snr(0.8, 0.0015)
display_snr(0.4, 0.0015)
display_snr(0.2, 0.0015)
display_distinguishability_score(1.0, 0.0015)
display_distinguishability_score(0.8, 0.0015)
display_distinguishability_score(0.4, 0.0015)
display_distinguishability_score(0.2, 0.0015)


# display_snr(1.0, 0.001)
# display_snr(0.5, 0.001)
# display_snr(0.1, 0.001)
# display_distinguishability_score(1.0, 0.001)
# display_distinguishability_score(0.5, 0.001)
# display_distinguishability_score(0.1, 0.001)

# TheseO
display_snr(1.0, 0.0008)
display_snr(0.8, 0.0008)
display_snr(0.4, 0.0008)
display_snr(0.2, 0.0008)
display_distinguishability_score(1.0, 0.0008)
display_distinguishability_score(0.8, 0.0008)
display_distinguishability_score(0.4, 0.0008)
display_distinguishability_score(0.2, 0.0008)
#
# display_snr(1.0, 0.0005)
# display_snr(0.5, 0.0005)
# display_distinguishability_score(1.0, 0.0005)
# display_distinguishability_score(0.5, 0.0005)
