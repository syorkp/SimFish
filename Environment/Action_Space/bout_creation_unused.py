def twoD_Gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y, theta, ):
    """Given parameters:
    - Max val at centre (shouldnt really matter...)
    """
    xo = float(mean_x)
    yo = float(mean_y)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    g = np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g#.ravel()


