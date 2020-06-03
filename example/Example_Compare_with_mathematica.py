import numpy as np
import matplotlib.pyplot as plt
from DataSet import DataSet as DS

#%% Functions for $\vec{A}$
def Ax(Coords: dict) -> np.ndarray:
    return Coords["X"]*Coords["Y"]*Coords["Z"]


def Ay(Coords: dict) -> np.ndarray:
    return Coords["X"]*Coords["Y"]*Coords["Y"]*Coords["Z"]


def Az(Coords: dict) -> np.ndarray:
    return Coords["X"] * Coords["Y"] * Coords["Y"] * Coords["Y"] * \
            Coords["Z"] * Coords["Z"]


#%% Functions for $\vec{B}$
def Bx(Coords: dict) -> np.ndarray:
    return -Coords["X"] * Coords["Y"] * Coords["Y"] + \
        3 * Coords["X"] * Coords["Y"] * Coords["Y"] * \
             Coords["Z"] * Coords["Z"]


def By(Coords: dict) -> np.ndarray:
    return Coords["X"] * Coords["Y"] - \
        Coords["Y"]**3 * Coords["Z"]**2


def Bz(Coords: dict) -> np.ndarray:
    return -Coords["X"] * Coords["Z"] + \
        Coords["Y"]**2 * Coords["Z"]

#%% Mesh
nx, ny, nz = 40, 40, 40  # Number of Cells in x-, y-, and z-direction
x0, y0, z0 = -1.0, -1.0, -1.0  # starting values of mesh in x-, y-, and z-direction
x1, y1, z1 = 1.0, 1.0, 1.0  # ending values of mesh in x-, y-, and z-direction

ds1 = DS(SystemOfCoords="CAR", NCell=(nx, ny, nz), startval=(x0, y0, z0), endval=(x1, y1, z1))

ds1.Scalar("Ax", "EdgeX", Ax)
ds1.Scalar("Ay", "EdgeY", Ay)
ds1.Scalar("Az", "EdgeZ", Az)

#%% Check if "CurlEdgeToFace" method works as expected
# building exact vaulues for B to compare with numerical results
ds1.Scalar("BEx", "FaceX", Bx)
ds1.Scalar("BEy", "FaceY", By)
ds1.Scalar("BEz", "FaceZ", Bz)
ds1.CurlEdgeToFace("Ax", "Ay", "Az", "Bx", "By", "Bz")

plt.ion()
ErrorZ = ds1.vars["Bz"]['val'] - ds1.vars["BEz"]['val']

for k in range(nz):
    plt.figure()
    plt.imshow((ErrorZ[:,:,k] / ds1.vars["BEz"]['val'][:,:,k]).T, origin='lower left')
    plt.colorbar()
    plt.title("k={0:02}".format(k))
    plt.pause(1)
    input("Press Enter...")
    plt.close()

#%% Now extract Vector potential and reproduce B to check if extraction method works as expected
ds1.ExtractAFromBFace("BEx", "BEy", "BEz", "AEx", "AEy", "AEz")
ds1.CurlEdgeToFace("AEx", "AEy", "AEz", "BBBx", "BBBy", "BBBz")

plt.ion()

ErrorX = ds1.vars["BBBx"]['val'] - ds1.vars["BEx"]['val']
for k in range(nx):
    plt.figure()
    plt.imshow((ErrorX[:,:,k] / ds1.vars["BEx"]['val'][:,:,k]).T, origin='lower left')
    plt.colorbar()
    plt.title("k={0:02}".format(k))
    plt.pause(1)
    input("Press Enter...")
    plt.close()

ErrorY = ds1.vars["BBBy"]['val'] - ds1.vars["BEy"]['val']
for k in range(ny):
    plt.figure()
    plt.imshow((ErrorY[:,:,k] / ds1.vars["BEy"]['val'][:,:,k]).T, origin='lower left')
    plt.colorbar()
    plt.title("k={0:02}".format(k))
    plt.pause(1)
    input("Press Enter...")
    plt.close()

ErrorZ = ds1.vars["BBBz"]['val'] - ds1.vars["BEz"]['val']
for k in range(nz):
    plt.figure()
    plt.imshow((ErrorZ[:,:,k] / ds1.vars["BEz"]['val'][:,:,k]).T, origin='lower left')
    plt.colorbar()
    plt.title("k={0:02}".format(k))
    plt.pause(1)
    input("Press Enter...")
    plt.close()
