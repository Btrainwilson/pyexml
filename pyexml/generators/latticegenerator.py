import numpy as np

class LatticeGenerator():
    
    def genSquarePos(self, side_sites = 5, lattice_constant = 1.0):
        return self.genRectPos(h=side_sites, w=side_sites)

    def genRectPos(self, h=5, w=5, lattice_constant=1.0):
        return self.genCappedRect(h=h, w=w, lattice_constant=lattice_constant)

    def genCappedRect(self, h=5, w=5, lattice_constant = 1.0, cap = -1):

        if cap == -1:
            max_site = h * w
        else:
            max_site = cap

        X = np.zeros([max_site, 2])

        for i in range(h):
            for j in range(w):
                idx = i * w + j
                if idx == max_site:
                    break
                X[idx] = [lattice_constant * i, lattice_constant * j]
            else:
                continue
            break

        return X

    def genApproxSquare(self, sites = 5, lattice_constant = 1.0):
        side_site = int(sites ** (0.5) + 1)
        return self.genCappedRect(h=side_site, w=side_site, lattice_constant=lattice_constant, cap = sites)
