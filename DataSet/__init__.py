#!/usr/bin/env python
# coding=utf-8

"""
    This is a simple code just to check
    the correctness of magnetic field
    and its physical constrains between
    two different meshes, even with different
    geometries.
"""

import sys
import h5py as h5
import numpy as np
from scipy.interpolate import griddata


class DataSet:
    """
    DataSet

    """

    def __init__(self, NCell, startval, endval, SystemOfCoords):
        """
        A Class to create uniform mesh in 1, 2, or 3 dimensions
        in Cartesian or system of coordinates.

        inputs ::
        ---------
        :param NCell: number of cells in each dimension
        :type NCell: Iterable, positive int

        :param startval: starting position in each direction
        :type startval: Iterable, float

        :param endval: starting position in each direction
        :type endval: Iterable, float

        :param SystemOfCoords: System of Coordinates
        :type SystemOfCoords: str ("CAR" or "SPH")

        :returns:

    """
        if not (SystemOfCoords in ["CAR", "SPH"]):
            sys.exit("System of coordinates must be one of the 'CAR' or 'SPH'")
        self.SystemOfCoords = SystemOfCoords
        self.NCell = NCell
        self.startval = startval
        self.endval = endval
        self.LocationOnGrid =\
            ["Cells", "Nodes", "FaceX", "FaceY", "FaceZ", "EdgeX", "EdgeY", "EdgeZ"]
        Direction = ['X', 'Y', 'Z']
        self.Direction = ['X', 'Y', 'Z']
        self.nodevect = dict()
        self.cellvect = dict()
        self.Cells = dict()
        self.Nodes = dict()
        self.EdgeX = dict()
        self.EdgeY = dict()
        self.EdgeZ = dict()
        self.FaceX = dict()
        self.FaceY = dict()
        self.FaceZ = dict()
        self.Ax = dict()
        self.Ay = dict()
        self.Az = dict()
        self.vars = dict()

        self.NAxes = 3

        ## building X, Y, and Z coordinates of Nodes and then Cells as seperate vectors,
        # then constructing Matrix form of Cells and Nodes coordinates
        for i in range(self.NAxes):
            self.nodevect[Direction[i]] = \
                np.linspace(start=self.startval[i], stop=self.endval[i], num=(self.NCell[i]+1))
            self.cellvect[Direction[i]] = 0.5 * (self.nodevect[Direction[i]][1:] + self.nodevect[Direction[i]][:-1])

        self.Cells['X'], self.Cells['Y'], self.Cells['Z'] = \
            np.meshgrid(self.cellvect['X'], self.cellvect['Y'], self.cellvect['Z'], indexing='ij')

        self.Nodes['X'], self.Nodes['Y'], self.Nodes['Z'] = \
            np.meshgrid(self.nodevect['X'], self.nodevect['Y'], self.nodevect['Z'], indexing='ij')

    ##
        self.EdgeX['X'], self.EdgeX['Y'], self.EdgeX['Z'] = \
            np.meshgrid(self.cellvect['X'], self.nodevect['Y'], self.nodevect['Z'], indexing='ij')

        self.EdgeY['X'], self.EdgeY['Y'], self.EdgeY['Z'] = \
            np.meshgrid(self.nodevect['X'], self.cellvect['Y'], self.nodevect['Z'], indexing='ij')

        self.EdgeZ['X'], self.EdgeZ['Y'], self.EdgeZ['Z'] = \
            np.meshgrid(self.nodevect['X'], self.nodevect['Y'], self.cellvect['Z'], indexing='ij')
    ##

        self.FaceX['X'], self.FaceX['Y'], self.FaceX['Z'] = \
            np.meshgrid(self.nodevect['X'], self.cellvect['Y'], self.cellvect['Z'], indexing='ij')

        self.FaceY['X'], self.FaceY['Y'], self.FaceY['Z'] = \
            np.meshgrid(self.cellvect['X'], self.nodevect['Y'], self.cellvect['Z'], indexing='ij')

        self.FaceZ['X'], self.FaceZ['Y'], self.FaceZ['Z'] = \
            np.meshgrid(self.cellvect['X'], self.cellvect['Y'], self.nodevect['Z'], indexing='ij')

    ##
        dy, dz = np.meshgrid(
            self.nodevect["Y"][1:] - self.nodevect["Y"][:-1],
            self.nodevect["Z"][1:] - self.nodevect["Z"][:-1], indexing='ij')
        ayz = dy * dz
        self.Ax["Location"] = "YZ"
        self.Ax["Area"] = np.empty_like(self.FaceX['X'])

        for i in range(self.Ax["Area"].shape[0]):
            self.Ax["Area"][i, :, :] = ayz
    ##
        dx, dz = np.meshgrid(
            self.nodevect["X"][1:] - self.nodevect["X"][:-1],
            self.nodevect["Z"][1:] - self.nodevect["Z"][:-1], indexing='ij')
        axz = dx * dz
        self.Ay["Location"] = "XZ"
        self.Ay["Area"] = np.empty_like(self.FaceY['Y'])

        for i in range(self.Ay["Area"].shape[1]):
            self.Ay["Area"][:, i, :] = axz

    ##
        dx, dy = np.meshgrid(
            self.nodevect["X"][1:] - self.nodevect["X"][:-1],
            self.nodevect["Y"][1:] - self.nodevect["Y"][:-1], indexing='ij')
        axy = dx * dy
        self.Az["Location"] = "XY"
        self.Az["Area"] = np.empty_like(self.FaceZ['Z'])

        for i in range(self.Az["Area"].shape[2]):
            self.Az["Area"][:, :, i] = axy
        return

    def Scalar(self, VarName, Location, function):
        """
        inputs ::
        ---------
        :param VarName:
        :type VarName:

        :param Location:
        :type Location:

        :param function:
        :type function:
        """
        if Location in self.LocationOnGrid:
            self.vars[VarName] = dict()
            self.vars[VarName]['Location'] = Location
            self.vars[VarName]["val"] = function(self.__dict__[Location])
        else:
            print(Location, " is not supported.")
            print("Possible locations for scalar variables are :: ", *self.LocationOnGrid)

    def DivCell(self, VarNameX, VarNameY, VarNameZ, NewVarName):
        """
        :param VarNameX:
        :type VarNameX:
        :param VarNameY:
        :type VarNameY:
        :param VarNameZ:
        :type VarNameZ:
        :param NewVarName:
        :type NewVarName:
        :return:
        :rtype:
        """
        if any(loc["Location"] != "Cells" for loc in
               (self.vars[VarNameX], self.vars[VarNameY], self.vars[VarNameZ])):
            print("Input parameters must be repesented on cell center!!")
            return

        VarX = self.vars[VarNameX]["val"]
        VarY = self.vars[VarNameY]["val"]
        VarZ = self.vars[VarNameZ]["val"]

        Areax = self.Ax['Area']
        Areay = self.Ay['Area']
        Areaz = self.Az['Area']

        inter = slice(1, -1)
        self.vars[NewVarName] = dict()
        self.vars[NewVarName]["Location"] = "Cells"
        self.vars[NewVarName]["val"] = np.zeros_like(self.vars[VarNameX]["val"])
        self.vars[NewVarName]["val"][1:-1, 1:-1, 1:-1] = \
            -0.5 * (VarX[:-2, inter, inter] + VarX[inter, inter, inter]) * Areax[1:-2, inter, inter] + \
            0.5 * (VarX[inter, inter, inter] + VarX[2:, inter, inter]) * Areax[2:-1, inter, inter] + \
            -0.5 * (VarY[inter, :-2, inter] + VarY[inter, inter, inter]) * Areay[inter, 1:-2, inter] + \
            0.5 * (VarY[inter, inter, inter] + VarY[inter, 2:, inter]) * Areay[inter, 2:-1, inter] + \
            -0.5 * (VarZ[inter, inter, :-2] + VarZ[inter, inter, inter]) * Areaz[inter, inter, 1:-2] + \
            0.5 * (VarZ[inter, inter, inter] + VarZ[inter, inter, 2:]) * Areaz[inter, inter, 2:-1]
        return

    def Write2HDF5(self, filename, databasename="Timestep_0"):
        """
        inputs ::
        ---------
        :param filename:
        :type filename:

        :param databasename:
        :type databasename:
        """

        fout = h5.File(filename, 'w')
        GTS = fout.create_group(databasename)
        GTS.attrs["Time"] = 0.0

        GVarsCell = GTS.create_group("VarsOnCell")
        GVarsCell.attrs['coords'] = \
            np.array([b'/cell_coords/X', b'/cell_coords/Y', b'/cell_coords/Z'], dtype='|S15')

        GVarsNode = GTS.create_group("VarsOnNode")
        GVarsNode.attrs['coords'] = \
            np.array([b'/node_coords/X', b'/node_coords/Y', b'/node_coords/Z'], dtype='|S15')

        for k in self.vars.keys():
            if self.vars[k]['Location'] == "Cells":
                GVarsCell.create_dataset(k, data=np.transpose(self.vars[k]["val"]))
            elif self.vars[k]['Location'] == "EdgeX":  # save Edge-located vars on Cell centers
                GVarsCell.create_dataset(k, data=np.transpose(
                    0.25 * (self.vars[k]["val"][:, :-1, :-1] + self.vars[k]["val"][:, 1:, :-1] +
                            self.vars[k]["val"][:, :-1, 1:] + self.vars[k]["val"][:, 1:, 1:])
                ))
            elif self.vars[k]['Location'] == "EdgeY":  # save Edge-located vars on Cell centers
                GVarsCell.create_dataset(k, data=np.transpose(
                    0.25 * (self.vars[k]["val"][:-1, :, :-1] + self.vars[k]["val"][1:, :, :-1] +
                            self.vars[k]["val"][:-1, :, 1:] + self.vars[k]["val"][1:, :, 1:])
                ))
            elif self.vars[k]['Location'] == "EdgeZ":  # save Edge-located vars on Cell centers
                GVarsCell.create_dataset(k, data=np.transpose(
                    0.25 * (self.vars[k]["val"][:-1, :-1, :] + self.vars[k]["val"][1:, :-1, :] +
                            self.vars[k]["val"][:-1, 1:, :] + self.vars[k]["val"][1:, 1:, :])
                ))

            elif self.vars[k]['Location'] == "Nodes":
                GVarsNode.create_dataset(k, data=np.transpose(self.vars[k]["val"]))
        GCell = fout.create_group("cell_coords")
        GNode = fout.create_group("node_coords")
        for i in self.Direction:
            GCell.create_dataset(i, data=np.transpose(self.Cells[i]))
            GNode.create_dataset(i, data=np.transpose(self.Nodes[i]))
        fout.close()

    def ToNewMesh(self, idata2, VarName, NewLocation):
        """
        ToNewMesh Method transfers a variable from one mesh to another one,
        Two meshes may be in the same dataset (same mesh) on different mesh points
        or on two different datasets with different system of coordinates or mesh spacing.

        inputs ::
        ---------

        :param idata2: destenation dataset
        :type idata2: dataset

        :param NewLocation: location of transfered variable
        :type NewLocation: str

        :param VarName:
        :type VarName:
        """
        if self.vars[VarName]["Location"] not in self.LocationOnGrid:
            print("Dataset 1 ::", self.vars[VarName]["Location"], " is not supported.")
            print("Possible locations for variables are :: ", *self.LocationOnGrid)

        if NewLocation not in self.LocationOnGrid:
            print("Dataset 2 ::", idata2.vars[VarName]["Location"], " is not supported.")
            print("Possible locations for variables are :: ", *self.LocationOnGrid)

        Loc1 = self.vars[VarName]["Location"]
        shape1 = self.__dict__[Loc1]['X'].shape
        Len1 = np.array(shape1).prod()
        values = np.reshape(self.vars[VarName]['val'], (Len1,))
        points1X = np.reshape(self.__dict__[Loc1]['X'], (Len1,))
        points1Y = np.reshape(self.__dict__[Loc1]['Y'], (Len1,))
        points1Z = np.reshape(self.__dict__[Loc1]['Z'], (Len1,))

        if self.SystemOfCoords == "SPH":
            RSinTheta1 = np.sin(points1Y) * points1X
            points1 = np.array([RSinTheta1 * np.cos(points1Z),
                                RSinTheta1 * np.sin(points1Z),
                                points1X * np.cos(points1Y)])
        else:
            points1 = (points1X, points1Y, points1Z)

        idata2.vars[VarName] = dict()
        idata2.vars[VarName]["Location"] = NewLocation

        if idata2.SystemOfCoords == "SPH":
            shape2 = idata2.__dict__[NewLocation]['X'].shape
            RSinTheta2 = idata2.__dict__[NewLocation]['X'] * np.sin(idata2[NewLocation]['Y'])
            points2X = RSinTheta2 * np.cos(idata2.__dict__[NewLocation]['Z'])
            points2Y = RSinTheta2 * np.sin(idata2.__dict__[NewLocation]['Z'])
            points2Z = idata2.__dict__[NewLocation]['X'] * np.cos(idata2.__dict__[NewLocation]['Y'])
            idata2.vars[VarName]["val"] = \
                griddata(points1, values, (points2X, points2Y, points2Z), method="linear")
        else:
            idata2.vars[VarName]["val"] = griddata(
                points1, values, (idata2.__dict__[NewLocation]['X'], idata2.__dict__[NewLocation]['Y'],
                                  idata2.__dict__[NewLocation]['Z']), method="linear")  # ''

    def CurlEdgeToCell(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
        """
        :param VarNameX:
        :type VarNameX:
        :param VarNameY:
        :type VarNameY:
        :param VarNameZ:
        :type VarNameZ:
        :param NewVarX:
        :type NewVarX:
        :param NewVarY:
        :type NewVarY:
        :param NewVarZ:
        :type NewVarZ:
        :return:
        :rtype:
        """
        if self.vars[VarNameX]["Location"] != "EdgeX":
            print("{} must be located at EdgeX.".format(VarNameX))
            return
        elif self.vars[VarNameY]["Location"] != "EdgeY":
            print("{} must be located at EdgeY.".format(VarNameY))
            return
        elif self.vars[VarNameZ]["Location"] != "EdgeZ":
            print("{} must be located at EdgeZ.".format(VarNameZ))
            return
        # Curl A |x
        self.vars[NewVarX] = dict()
        self.vars[NewVarX]["Location"] = "Cells"
        Temp1 = 0.5 * (self.vars[VarNameZ]["val"][:-1, :, :] +
                       self.vars[VarNameZ]["val"][1:, :, :])
        Temp2 = 0.5 * (self.vars[VarNameY]["val"][:-1, :, :] +
                       self.vars[VarNameY]["val"][1:, :, :])
        LeftTerm = (Temp1[:, 1:, :] - Temp1[:, :-1, :]) / \
                   (self.FaceY['Y'][:, 1:, :] -
                    self.FaceY['Y'][:, :-1, :])
        self.vars[NewVarX]['val'] = \
            LeftTerm - (Temp2[:, :, 1:] - Temp2[:, :, :-1]) / \
            (self.FaceZ['Z'][:, :, 1:] - self.FaceZ['Z'][:, :, :-1])

        # Curl A |y
        self.vars[NewVarY] = dict()
        self.vars[NewVarY]["Location"] = "Cells"
        Temp1 = 0.5 * \
            (self.vars[VarNameX]["val"][:, :-1, :] + self.vars[VarNameX]["val"][:, 1:, :])
        Temp2 = 0.5 * \
            (self.vars[VarNameZ]["val"][:, :-1, :] + self.vars[VarNameZ]["val"][:, 1:, :])
        LeftTerm = (Temp1[:, :, 1:] - Temp1[:, :, :-1]) / \
                   (self.FaceZ['Z'][:, :, 1:] - self.FaceZ['Z'][:, :, :-1])
        self.vars[NewVarY]['val'] = \
            LeftTerm - (Temp2[1:, :, :] - Temp2[:-1, :, :]) / \
            (self.FaceX['X'][1:, :, :] - self.FaceX['X'][:-1, :, :])

        # Curl A |z
        self.vars[NewVarZ] = dict()
        self.vars[NewVarZ]["Location"] = "Cells"
        Temp1 = 0.5 * \
            (self.vars[VarNameY]["val"][:, :, :-1] + self.vars[VarNameY]["val"][:, :, 1:])
        Temp2 = 0.5 * \
            (self.vars[VarNameX]["val"][:, :, :-1] + self.vars[VarNameX]["val"][:, :, 1:])
        LeftTerm = (Temp1[1:, :, :] - Temp1[:-1, :, :]) / \
            (self.FaceX['X'][1:, :, :] - self.FaceX['X'][:-1, :, :])
        self.vars[NewVarZ]['val'] = LeftTerm - \
            (Temp2[:, 1:, :] - Temp2[:, :-1, :]) / \
            (self.FaceY['Y'][:, 1:, :] - self.FaceY['Y'][:, :-1, :])
