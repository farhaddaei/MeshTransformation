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
    return -0.5 * B0 * Coord['Y']


def Ay(Coord):
    return 0.5 * B0 * Coord['X']


def Az(Coord):
    return np.zeros_like(Coord['Z'])


# uniform \vec{A}
def Axu(Coord):
    return Coord['Z'] * Coord['Y']


def Ayu(Coord):
    return np.zeros_like(Coord['Y'])


def Azu(Coord):
    return Coord['X']


if __name__ == "__main__":
    DS1 = DataSet((40, 40, 40), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")

    DS1.Scalar("bx", "Cells", BFieldX)  # BxUni
    DS1.Scalar("by", "Cells", BFieldY)  # ByUni
    DS1.Scalar("bz", "Cells", BFieldZ)  # BzUni
    DS1.DivCell("bx", "by", "bz", "DivB")  # Dircet calculation of B

    DS1.Scalar("bxfd", "FaceX", BFieldX)  # BxUni
    DS1.Scalar("byfd", "FaceY", BFieldY)  # ByUni
    DS1.Scalar("bzfd", "FaceZ", BFieldZ)  # BzUni
    DS1.DivFace("bxfd", "byfd", "bzfd", "DivBfd")  # Dircet calculation of B

    DS1.Scalar("Ax", "EdgeX", Ax)
    DS1.Scalar("Ay", "EdgeY", Ay)
    DS1.Scalar("Az", "EdgeZ", Az)

    DS1.CurlEdgeToCell("Ax", "Ay", "Az", "BXa", "BYa", "BZa")
    DS1.DivCell("BXa", "BYa", "BZa", "DivBa")  # using A on Edges and B on Cells

    DS1.CurlEdgeToFace("Ax", "Ay", "Az", "BXf", "BYf", "BZf")
    DS1.DivFace("BXf", "BYf", "BZf", "DivBf")  # using A on Edges and B on Cells

    DS1.Write2HDF5("DataSet1.h5")

# Try transfering to new Sys. of Coords.
    DS2 = DataSet((30, 30, 25), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")

# Direct transformation of B located @ Cell center
    print("B ::\n 1")
    DS1.ToNewMesh(DS2, "bx", "Cells")
    print(" 2")
    DS1.ToNewMesh(DS2, "by", "Cells")
    print(" 3")
    DS1.ToNewMesh(DS2, "bz", "Cells")
    print("B :: Done!")
    DS2.DivCell("bx", "by", "bz", "DivB")

# Direct transformation of B located @ Cell Faces
    print("B @ face ::\n 1")
    DS1.ToNewMesh(DS2, "bxfd", "FaceX")
    print(" 2")
    DS1.ToNewMesh(DS2, "byfd", "FaceY")
    print(" 3")
    DS1.ToNewMesh(DS2, "bzfd", "FaceZ")
    print("B @ face:: Done!")
    DS2.DivFace("bxfd", "byfd", "bzfd", "DivBfd")

    print("A ::\n 1")
    DS1.ToNewMesh(DS2, "Ax", "EdgeX")
    print(" 2")
    DS1.ToNewMesh(DS2, "Ay", "EdgeY")
    print(" 3")
    DS1.ToNewMesh(DS2, "Az", "EdgeZ")
    print("A :: Done!")

    DS2.CurlEdgeToCell("Ax", "Ay", "Az", "Bxa", "Bya", "Bza")
    DS2.DivCell("Bxa", "Bya", "Bza", "DivBa")

    DS2.CurlEdgeToFace("Ax", "Ay", "Az", "BXf", "BYf", "BZf")
    DS2.DivFace("BXf", "BYf", "BZf", "DivBf")  # using A on Edges and B on Cells

    DS2.Write2HDF5("DataSet2.h5")
