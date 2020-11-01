import numpy as np

import skimage.draw as draw
from skimage import io

from Tools.ray_cast import rays


class DrawingBoard:

    def __init__(self, width, height):

        self.width = width
        self.height = height
        self.db = None
        self.erase()

    def erase(self, bkg=0):
        if bkg == 0:
            self.db = np.zeros((self.height, self.width, 3), dtype=np.double)
        else:
            self.db = np.ones((self.height, self.width, 3), dtype=np.double) * bkg

    def apply_light(self, dark_col, dark_gain, light_gain):
        self.db[:, :dark_col] *= dark_gain
        self.db[:, dark_col:] *= light_gain
        
    def circle(self, center, rad, color):
        rr, cc = draw.circle(center[1], center[0], rad, self.db.shape)
        self.db[rr, cc, :] = color

    @staticmethod
    def multi_circles(cx, cy, rad):
        rr, cc = draw.circle(0, 0, rad)
        rrs = np.tile(rr, (len(cy), 1)) + np.tile(np.reshape(cy, (len(cy), 1)), (1, len(rr)))
        ccs = np.tile(cc, (len(cx), 1)) + np.tile(np.reshape(cx, (len(cx), 1)), (1, len(cc)))
        return rrs, ccs
    
    def line(self, p1, p2, color):
        rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
        self.db[rr, cc, :] = color

    def get_size(self):
        return self.width, self.height

    def read_rays(self, xmat, ymat, dark_gain, light_gain, bkg_scatter, dark_col=0):
        res = rays(xmat.astype(np.int), ymat.astype(np.int), self.db, self.height, self.width, dark_gain, light_gain,
                   bkg_scatter, dark_col=dark_col)
        return res        

    def read(self, xmat, ymat):
        n_arms = xmat.shape[0]
        res = np.zeros((n_arms, 3))
        for arm in range(n_arms):
            [rr, cc] = draw.line(ymat[arm, 0].astype(int), xmat[arm, 0].astype(int), ymat[arm, 1].astype(int),
                                 xmat[arm, 1].astype(int))
            prfl = self.db[rr, cc]
            # prfl = np.array(profile_line(self.db, (ymat[arm,0], xmat[arm,0]), (ymat[arm,1], xmat[arm,1]), order=0, cval=1.))
            ps = np.sum(prfl, 1)
            if len(np.nonzero(ps)[0]) > 0:
                res[arm, :] = prfl[np.nonzero(ps)[0][0], :]
            else:
                res[arm, :] = [0, 0, 0]

        # xmat_ = np.where((xmat<0) | (xmat>=self.width), 0, xmat)
        # ymat_ = np.where((ymat<0) | (ymat>=self.height), 0, ymat)
        #
        # res = self.db[ymat_, xmat_, :]
        # res[np.where((xmat<0)|xmat>=self.width)|(ymat<0)|(ymat>=self.height), :] = [1, 0, 0]
        return res

    def show(self):
        io.imshow(self.db)
        io.show()


if __name__ == "__main__":
    d = DrawingBoard(500, 500)
    d.circle((100, 200), 100, (1, 0, 0))
    d.line((50, 50), (100, 200), (0, 1, 0))
    d.show()
