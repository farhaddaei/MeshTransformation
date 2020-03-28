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
# uniform \vec{A}

def Ax(Coord):
    return -0.5 * B0 * Coord['Y']


def Ay(Coord):
    return 0.5 * B0 * Coord['X']


def Az(Coord):
    return np.zeros_like(Coord['Z'])


if __name__ == "__main__":
    DS1 = DataSet((40, 40, 40), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")

    DS1.Scalar("bx", "Cells", BFieldX)  # BxUni
    DS1.Scalar("by", "Cells", BFieldY)  # ByUni
    DS1.Scalar("bz", "Cells", BFieldZ)  # BzUni

    DS1.Scalar("Ax", "EdgeX", Ax)
    DS1.Scalar("Ay", "EdgeY", Ay)
    DS1.Scalar("Az", "EdgeZ", Az)

    DS1.DivCell("bx", "by", "bz", "DivB")
    DS1.Write2HDF5("1classtest.h5")

    # DS2 = DataSet((20, 20, 20), (0.0, 0.0, 0.0), (Lx, Lx, Lx), "CAR")
    # print("Start\n1")
    # DS1.ToNewMesh(DS2, "bx", "Cells")
    # print("2")
    # DS1.ToNewMesh(DS2, "by", "Cells")
    # print("3")
    # DS1.ToNewMesh(DS2, "bz", "Cells")
    # print("Done!")
    # DS2.DivCell("bx", "by", "bz", "DivB")
    # DS2.Write2HDF5("2classtest.h5")
