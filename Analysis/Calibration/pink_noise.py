
import numpy as np
import matplotlib.pyplot as plt

def smoothNoise(x, y, width, height, noise):
    fractX = x - int(x)
    fractY = y - int(y)

    x1 = (int(x) + width) % width
    y1 = (int(y) + height) % height

    x2 = (x1 + width - 1) % width
    y2 = (y1 + height - 1) % height
    value = 0.0
    value += fractX * fractY * noise[y1, x1]
    value += (1-fractX) * fractY * noise[y1, x2]
    value += fractX * (1-fractY) * noise[y2, x1]
    value += (1-fractX) * (1-fractY) * noise[y2, x2]

    return value

def turbulence(x, y, width, height, size, noise):
    value = 0.0
    initialSize = size

    while size >= 1:
        value += smoothNoise(x/size, y/size, width, height, noise) * size
        size /= 2.0
    return 128 * value/initialSize


def marble_texture(width, height, x, y):
    # xp, yp = np.arange(width), np.arange(height)
    # background = 255 * np.sin(xp[:, None]*x + yp[None, :]*y)

    # Generate these randomly so grid can have any orientation.
    xPeriod = 5.0
    yPeriod = 10.0
    turbPower = 1.0
    turbSize = 162.0

    noise = np.absolute(np.random.randn(1500, 1500))
    background = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            xyValue = i * xPeriod / width + j * yPeriod / height + turbPower * turbulence(i, j, width, height, turbSize, noise)/256.0
            sineValue = 256 * np.abs(np.sin(xyValue * 3.14159))
            background[i, j] = sineValue

    plt.imshow(background)
    plt.show()


marble_texture(1500, 1500, 0.01, 0.1)


def get_background_grating(frequency, width, height):
    base_unit = np.concatenate((np.ones((1, frequency)),
                                                      np.zeros((1, frequency))), axis=1)
    number_required = int(width / frequency)
    full_width = np.tile(base_unit, number_required)[:, :width]
    full_arena = np.repeat(full_width, height, axis=0)
    noise = np.absolute(np.random.randn(1500, 1500))
    return full_arena * noise

width, height = 1500, 1500

bk = get_background_grating(50, 1500, 1500)


xp, yp = np.arange(width), np.arange(height)
#
# x = xp[:, None],
# y = yp[None, :]
#
# init = np.random.randint(0, width, 2)
# i, j = init[0], init[1]
#
# positional_mask = (((x - i) ** 2 + (y - j) ** 2) ** 0.5)[0] + 1  # Measure of distance from init at every point.

noise = np.random.randn(1500, 1500)
noise = np.fft.fft2(noise)
f = 500.5
vals = [1.1, 10.1, 100.1, 1000.1, 5000.1]
xp_1 = np.tile(xp, (1500, 1)) + 1
yp_1 = np.tile(yp, (1500, 1)).T + 1

for f in vals:
    xp = xp_1 % f
    yp = yp_1 % f
    inverse_distance_f_origin = 5/((xp**2 + yp ** 2)**(0.5))
    pink = noise * inverse_distance_f_origin
    pink = np.fft.ifft2(pink).real
    # plt.imshow(pink)
    # plt.show()
