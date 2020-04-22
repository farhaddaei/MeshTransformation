#!/usr/bin/env python
# coding=utf-8

import numpy as np
from DataSet import DataSet

B0 = 1.0
Lx = 1.0
K1 = 2.0 * np.pi / Lx


# Just some functions to evalute different parameters
def Rho(Coords):
    return np.cos(np.pi * Coords["Z"]) * \
           np.exp(-(Coords["X"] * Coords["X"] + Coords["Y"] * Coords["Y"]))


def BFieldX(Coords):
    return B0 * np.exp(- K1 * Coords["Z"]) * np.cos(K1 * Coords["X"])


def BFieldY(Coords):
    return 0.0 * Coords["Y"]


def BFieldZ(Coords):
    return - B0 * np.exp(- K1 * Coords["Z"]) * np.sin(K1 * Coords["X"])


# Bfield - Uniform
def BxUni(Coord):
    return np.ones_like(Coord["X"])


def ByUni(Coord):
    return np.ones_like(Coord["Y"])


def BzUni(Coord):
    return np.ones_like(Coord["Z"])


# Magnetic Vector Potential components
def Ax(Coord):
    return np.zeros_like(Coord['X'])


def Ay(Coord):
    return np.zeros_like(Coord['Y'])


def Az(Coord):
    L = 100.0
    x0, y0 = -2.0, 3.0
    X = Coord['X'] - x0
    Y = Coord['Y'] - y0
    Coeff = 10.0
    return Coeff * (np.log(2.0 * L) - np.log(np.sqrt(X*X + Y*Y)))


# uniform \vec{A}
def Axu(Coord):
    return Coord['Z'] * Coord['Y']


def Ayu(Coord):
    return np.zeros_like(Coord['Y'])


def Azu(Coord):
    return Coord['X']


if __name__ == "__main__":
    DSfiles = DataSet(SystemOfCoords="CAR", BaseAddress="/home/sigma/PycharmProjects/extract/", NBlocks=20, UseBlock=0)
if False:
    DS1 = DataSet((40, 40, 40), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")
#
#     DS1.Scalar("bx", "Cells", BFieldX)  # BxUni
#     DS1.Scalar("by", "Cells", BFieldY)  # ByUni
#     DS1.Scalar("bz", "Cells", BFieldZ)  # BzUni
#     DS1.DivCell("bx", "by", "bz", "DivB")  # Dircet calculation of B
#
    # DS1.Scalar("bxfd", "FaceX", BFieldX)  # BxUni
    # DS1.Scalar("byfd", "FaceY", BFieldY)  # ByUni
    # DS1.Scalar("bzfd", "FaceZ", BFieldZ)  # BzUni
    # DS1.DivFace("bxfd", "byfd", "bzfd", "DivBfd")  # Dircet calculation of B
    # DS1.AvgFaceToCell("bxfd", "byfd", "bzfd", "BxCell", "ByCell", "BzCell")  # Dircet calculation of B
    # DS1.DivCellFD("BxCell", "ByCell", "BzCell", "DivCellB")
#
    DS1.Scalar("Ax", "EdgeX", Ax)
    DS1.Scalar("Ay", "EdgeY", Ay)
    DS1.Scalar("Az", "EdgeZ", Az)
#
    # DS1.CurlEdgeToCell("Ax", "Ay", "Az", "BXa", "BYa", "BZa")
    # DS1.DivCell("BXa", "BYa", "BZa", "DivBa")  # using A on Edges and B on Cells
#
    DS1.CurlEdgeToFace("Ax", "Ay", "Az", "BXf", "BYf", "BZf")
    DS1.DivFace("BXf", "BYf", "BZf", "DivBf")  # using A on Edges and B on Cells

    DS1.CurlFaceToEdge("BXf", "BYf", "BZf", "JXe", "JYe", "JZe")

    DS1.AvgFaceToCell("BXf", "BYf", "BZf", "BXc", "BYc", "BZc")
    DS1.AvgEdgeToCell("JXe", "JYe", "JZe", "JXc", "JYc", "JZc")
    DS1.Cross("JXc", "JYc", "JZc", "BXc", "BYc", "BZc", "LFXc", "LFYc", "LFZc")
    DS1.Write2HDF5("DataSet1111.h5")

# # Try transfering to new Sys. of Coords.
    DS2 = DataSet((38, 38, 40), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")

    print("A ::\n 1")
    DS1.ToNewMesh(DS2, "Ax", "EdgeX")
    print(" 2")
    DS1.ToNewMesh(DS2, "Ay", "EdgeY")
    print(" 3")
    DS1.ToNewMesh(DS2, "Az", "EdgeZ")
    print("A :: Done!")

    DS2.CurlEdgeToFace("Ax", "Ay", "Az", "BXf", "BYf", "BZf")
    DS2.DivFace("BXf", "BYf", "BZf", "DivBf")  # using A on Edges and B on Cells
    DS2.CurlFaceToEdge("BXf", "BYf", "BZf", "JXe", "JYe", "JZe")
    DS2.AvgFaceToCell("BXf", "BYf", "BZf", "BXc", "BYc", "BZc")
    DS2.AvgEdgeToCell("JXe", "JYe", "JZe", "JXc", "JYc", "JZc")
    DS2.Cross("JXc", "JYc", "JZc", "BXc", "BYc", "BZc", "LFXc", "LFYc", "LFZc")
    DS2.Write2HDF5("DataSet2222.h5")
