#!/usr/bin/env python3
# coding=utf-8


"""
    This is a simple code just to check
    the correctness of magnetic field
    and its physical constrains between
    two different meshes, even with different
    geometries.
"""

from __future__ import print_function
import os
import sys
import fnmatch
import h5py as h5
import numpy as np
import vtk
from scipy.interpolate import RegularGridInterpolator
from memory_profiler import profile


class DataSet:
    """
    DataSet
    """
    def __init__(self, SystemOfCoords, NCell:[tuple]= None, startval=None, endval=None,
                 Functions=('linear', 'linear', 'linear'),
                 BaseAddress=None, Pattern=None, NBlocks:int = None, UseBlock:int = None):
        """
        A Class to create uniform mesh in 1, 2, or 3 dimensions
        in Cartesian or system of coordinates.

        inputs :
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

        self.LocationOnGrid = \
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
        if not (SystemOfCoords in ["CAR", "SPH"]):
            sys.exit("System of coordinates must be one of the 'CAR' or 'SPH'")
        self.SystemOfCoords = SystemOfCoords
        if BaseAddress is None and Pattern is None and NBlocks is None:
            self.LoadFromFile = False
            self.NCell = NCell
            self.startval = startval
            self.endval = endval
            for i in range(self.NAxes):
                if Functions[i] == 'linear':
                    self.nodevect[Direction[i]] = \
                        np.linspace(start=self.startval[i], stop=self.endval[i], num=(self.NCell[i] + 1))
                else:
                    # TODO :: Call Non-Uniform Mesh Function
                    print("Calling non-uniform mesh on {} direction".format(Direction[i]))
                    self.NCell[i], self.nodevect[Direction[i]] = \
                    self.NonUniformMesh(Functions=Functions[i],
                                        StartVals=startval[i],
                                        EndVals=endval[i],
                                        NumInterval=self.NCell[i])
        elif NCell is None and startval is None and endval is None:
            self.LoadFromFile = True
            if Pattern is None:
                Pattern = '*.npz'
            if BaseAddress is None:
                BaseAddress = '.'
            if UseBlock is None:
                UseBlock = 0
            self.NBlocks = NBlocks
            self.BaseAddress = BaseAddress
            self.UseBlock = UseBlock
            self.Pattern = Pattern

            self.geo = self.FindGeometry(NBlocks, BaseAddress, Pattern, UseBlock)
            print("Following parameters are found in the given NPZ file/s :: ")
            for v in self.geo["vars"].keys():
                print(" ", v, " (defined on ", self.geo["vars"][v]["Location"], ")", sep='')
            self.NCell = (self.geo["x"].size - 1, self.geo["y"].size - 1, self.geo["z"].size - 1)
            self.nodevect[self.Direction[0]] = self.geo["x"]
            self.nodevect[self.Direction[1]] = self.geo["y"]
            self.nodevect[self.Direction[2]] = self.geo["z"]
        else:
            sys.exit("---<( Only one set input parameters must be used !!! )>---")

        for i in range(self.NAxes):
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

        if self.LoadFromFile:
            self.LoadData()
        self.XMFHeader = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="node_mesh" GridType="Uniform">
     <Topology TopologyType="3DSMesh" NumberOfElements="{2} {1} {0}"/>"""
        self.XMFFooter = """   </Grid>
 </Domain>
</Xdmf>"""
        self.Geometry = """ <Geometry GeometryType="X_Y_Z">
   <DataItem Dimensions="{2} {1} {0}" NumberType="Float" Precision="8" Format="HDF">
    .//{3}:/node_coords/X
   </DataItem>
   <DataItem Dimensions="{2} {1} {0}" NumberType="Float" Precision="8" Format="HDF">
    .//{3}:/node_coords/Y
   </DataItem>
   <DataItem Dimensions="{2} {1} {0}" NumberType="Float" Precision="8" Format="HDF">
    .//{3}:/node_coords/Z
   </DataItem>
 </Geometry>"""
        self.Attribute = """ <Attribute Name="{0}" AttributeType="Scalar" Center="{1}">
   <DataItem Dimensions="{4} {3} {2}" NumberType="Float" Precision="8" Format="HDF">
    .//{5}:/{6}/{0}
   </DataItem>
 </Attribute>"""
        return

    def NonUniformMesh(self, Functions, StartVals, EndVals, NumInterval):
        """

        :param Functions:
        :param StartVals:
        :param EndVals:
        :param NumInterval:
        :return:
        """
        if (len(Functions) != len(NumInterval)) or \
            len(NumInterval) != len(StartVals) or \
            len(StartVals) != len(EndVals):
            sys.exit("Number of some shits must be equal")

        totalcells = sum(NumInterval)
        result = np.empty([totalcells+1], dtype=np.float64)
        # DEBUG
        print("  Total Number of Cells {0}".format(totalcells))
        firstpoint = 0
        for f in range(len(Functions)):
            if Functions[f] == 'linear':
                temp = np.linspace(StartVals[f], EndVals[f],
                                     num=(NumInterval[f] + 1), endpoint=True)
            else:
                temp = Functions[f](StartVals[f], EndVals[f], NumInterval[f])
            # DEBUG
            print("  Start {0} End {1} size {2}".format(firstpoint, firstpoint+NumInterval[f], temp.size))
            result[firstpoint:(firstpoint + NumInterval[f] + 1)] = temp[:]
            firstpoint += NumInterval[f]

        return totalcells, result

    def Scalar(self, VarName, Location, function, InitialVal=None):
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
        # To make sure that new var will not overwrite old ones
        if Location in self.LocationOnGrid:
            if VarName in self.vars.keys():
                tempName = VarName + '_' + self.vars[VarName]["Location"]
                if tempName in self.vars.keys():
                    i = 1
                    while tempName + "_{0}".format(i) in self.vars.keys():
                        i += 1
                    tempName = tempName + "_{0}".format(i)
                print("---<( {0} already exists, old variable renamed to {1} )>---".
                      format(VarName, tempName))
                self.vars[tempName] = self.vars.pop(VarName)

            self.vars[VarName] = dict()
            self.vars[VarName]['Location'] = Location
            self.vars[VarName]["val"] = function(self.__dict__[Location])
            if InitialVal is not None:
                self.vars[VarName]["val"][:, :, :] = InitialVal
        else:
            print(Location, " is not supported.")
            print("Possible locations for scalar variables are :: ", *self.LocationOnGrid)

    # Defining Divergence Operators
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
            print("Input parameters must be represented on cell center!!")
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

    def DivCellFD(self, VarNameX, VarNameY, VarNameZ, NewVarName):
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
            (self.vars[VarNameX]["val"][2:, 1:-1, 1:-1] - self.vars[VarNameX]["val"][:-2, 1:-1, 1:-1]) / \
            (self.Cells["X"][2:, 1:-1, 1:-1] - self.Cells["X"][:-2, 1:-1, 1:-1]) + \
            (self.vars[VarNameY]["val"][1:-1, 2:, 1:-1] - self.vars[VarNameY]["val"][1:-1, :-2, 1:-1]) / \
            (self.Cells["Y"][1:-1, 2:, 1:-1] - self.Cells["Y"][1:-1, :-2, 1:-1]) + \
            (self.vars[VarNameZ]["val"][1:-1, 1:-1, 2:] - self.vars[VarNameZ]["val"][1:-1, 1:-1, :-2]) / \
            (self.Cells["Z"][1:-1, 1:-1, 2:] - self.Cells["Z"][1:-1, 1:-1, :-2])
        return

    def DivFace(self, VarNameX, VarNameY, VarNameZ, NewVarName):
        """
        DivFace calculates Div. of vector varialble located
        on cell faces on cell centers

        Inputs:
        -------
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
        if (self.vars[VarNameX]["Location"] != "FaceX"
                or self.vars[VarNameY]["Location"] != "FaceY"
                or self.vars[VarNameZ]["Location"] != "FaceZ"):
            print("Input parameters must be repesented on Faces!!")
            return

        self.vars[NewVarName] = dict()
        self.vars[NewVarName]["Location"] = "Cells"
        self.vars[NewVarName]["val"] = \
            self.vars[VarNameX]["val"][1:, :, :] * self.Ax["Area"][1:, :, :] - \
            self.vars[VarNameX]["val"][:-1, :, :] * self.Ax["Area"][:-1, :, :] + \
            self.vars[VarNameY]["val"][:, 1:, :] * self.Ay["Area"][:, 1:, :] - \
            self.vars[VarNameY]["val"][:, :-1, :] * self.Ay["Area"][:, :-1, :] + \
            self.vars[VarNameZ]["val"][:, :, 1:] * self.Az["Area"][:, :, 1:] - \
            self.vars[VarNameZ]["val"][:, :, :-1] * self.Az["Area"][:, :, :-1]
        return

    # Defining Curl Operators
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
        Temp1 = 0.5 * (self.vars[VarNameZ]["val"][:-1, :, :]
                       + self.vars[VarNameZ]["val"][1:, :, :])
        Temp2 = 0.5 * (self.vars[VarNameY]["val"][:-1, :, :]
                       + self.vars[VarNameY]["val"][1:, :, :])
        LeftTerm = (Temp1[:, 1:, :] - Temp1[:, :-1, :]) / \
                   (self.FaceY['Y'][:, 1:, :]
                    - self.FaceY['Y'][:, :-1, :])
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

    def CurlEdgeToFace(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
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
        # Curl A | X-Component
        self.vars[NewVarX] = dict()
        self.vars[NewVarX]["Location"] = "FaceX"
        self.vars[NewVarX]["val"] = \
            (self.vars[VarNameZ]["val"][:, 1:, :] - self.vars[VarNameZ]["val"][:, :-1, :]) / \
            (self.EdgeZ["Y"][:, 1:, :] - self.EdgeZ["Y"][:, :-1, :]) - \
            (self.vars[VarNameY]["val"][:, :, 1:] - self.vars[VarNameY]["val"][:, :, :-1]) / \
            (self.EdgeY["Z"][:, :, 1:] - self.EdgeY["Z"][:, :, :-1])

        # Curl A | Y-Component
        self.vars[NewVarY] = dict()
        self.vars[NewVarY]["Location"] = "FaceY"
        self.vars[NewVarY]["val"] = \
            (self.vars[VarNameX]["val"][:, :, 1:] - self.vars[VarNameX]["val"][:, :, :-1]) / \
            (self.EdgeX["Z"][:, :, 1:] - self.EdgeX["Z"][:, :, :-1]) - \
            (self.vars[VarNameZ]["val"][1:, :, :] - self.vars[VarNameZ]["val"][:-1, :, :]) / \
            (self.EdgeZ["X"][1:, :, :] - self.EdgeZ["X"][:-1, :, :])

        # Curl A | Z-Component
        self.vars[NewVarZ] = dict()
        self.vars[NewVarZ]["Location"] = "FaceZ"
        self.vars[NewVarZ]["val"] = \
            (self.vars[VarNameY]["val"][1:, :, :] - self.vars[VarNameY]["val"][:-1, :, :]) / \
            (self.EdgeY["X"][1:, :, :] - self.EdgeY["X"][:-1, :, :]) - \
            (self.vars[VarNameX]["val"][:, 1:, :] - self.vars[VarNameX]["val"][:, :-1, :]) / \
            (self.EdgeX["Y"][:, 1:, :] - self.EdgeX["Y"][:, :-1, :])

    def CurlFaceToEdge(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
        """
        CurlFaceToEdge calculates the Curl of vector defined
        on cell faces on cell centers

        Inputs:
        -------
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
        if (self.vars[VarNameX]["Location"] != "FaceX"
                or self.vars[VarNameY]["Location"] != "FaceY"
                or self.vars[VarNameZ]["Location"] != "FaceZ"):
            print("@CurlFaceToEdge :: Input parameters must be repesented on Faces!!")
            return
        # X-Component
        self.vars[NewVarX] = dict()
        self.vars[NewVarX]["Location"] = "EdgeX"
        self.vars[NewVarX]["val"] = np.zeros_like(self.EdgeX["X"])
        self.vars[NewVarX]["val"][:, 1:-1, 1:-1] = \
            (self.vars[VarNameZ]["val"][:, 1:, 1:-1] - self.vars[VarNameZ]["val"][:, :-1, 1:-1]) / \
            (self.FaceZ["Y"][:, 1:, 1:-1] - self.FaceZ["Y"][:, :-1, 1:-1]) - \
            (self.vars[VarNameY]["val"][:, 1:-1, 1:] - self.vars[VarNameY]["val"][:, 1:-1, :-1]) / \
            (self.FaceY["Z"][:, 1:-1, 1:] - self.FaceY["Z"][:, 1:-1, :-1])

        # Y-Component
        self.vars[NewVarY] = dict()
        self.vars[NewVarY]["Location"] = "EdgeY"
        self.vars[NewVarY]["val"] = np.zeros_like(self.EdgeY["Y"])
        self.vars[NewVarY]["val"][1:-1, :, 1:-1] = \
            (self.vars[VarNameX]["val"][1:-1, :, 1:] - self.vars[VarNameX]["val"][1:-1, :, :-1]) / \
            (self.FaceX["Z"][1:-1, :, 1:] - self.FaceX["Z"][1:-1, :, :-1]) - \
            (self.vars[VarNameZ]["val"][1:, :, 1:-1] - self.vars[VarNameZ]["val"][:-1, :, 1:-1]) / \
            (self.FaceZ["X"][1:, :, 1:-1] - self.FaceZ["X"][:-1, :, 1:-1])

        # Z-Component
        self.vars[NewVarZ] = dict()
        self.vars[NewVarZ]["Location"] = "EdgeZ"
        self.vars[NewVarZ]["val"] = np.zeros_like(self.EdgeZ["Z"])
        self.vars[NewVarZ]["val"][1:-1, 1:-1, :] = \
            (self.vars[VarNameY]["val"][1:, 1:-1, :] - self.vars[VarNameY]["val"][:-1, 1:-1, :]) / \
            (self.FaceY["X"][1:, 1:-1, :] - self.FaceY["X"][:-1, 1:-1, :]) - \
            (self.vars[VarNameX]["val"][1:-1, 1:, :] - self.vars[VarNameX]["val"][1:-1, :-1, :]) / \
            (self.FaceX["Y"][1:-1, 1:, :] - self.FaceX["Y"][1:-1, :-1, :])
        return

    def AvgFaceToCell(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
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
        if self.vars[VarNameX]["Location"] != "FaceX":
            print("{} must be located at FaceX.".format(VarNameX))
            return
        elif self.vars[VarNameY]["Location"] != "FaceY":
            print("{} must be located at FaceY.".format(VarNameY))
            return
        elif self.vars[VarNameZ]["Location"] != "FaceZ":
            print("{} must be located at FaceZ.".format(VarNameZ))
            return

        self.vars[NewVarX] = dict()
        self.vars[NewVarX]["Location"] = "Cells"
        self.vars[NewVarX]["val"] = np.empty_like(self.Cells["X"])
        self.vars[NewVarX]["val"] = \
            0.5 * (self.vars[VarNameX]["val"][:-1, :, :] + self.vars[VarNameX]["val"][1:, :, :])

        self.vars[NewVarY] = dict()
        self.vars[NewVarY]["Location"] = "Cells"
        self.vars[NewVarY]["val"] = np.empty_like(self.Cells["Y"])
        self.vars[NewVarY]["val"] = \
            0.5 * (self.vars[VarNameY]["val"][:, :-1, :] + self.vars[VarNameY]["val"][:, 1:, :])

        self.vars[NewVarZ] = dict()
        self.vars[NewVarZ]["Location"] = "Cells"
        self.vars[NewVarZ]["val"] = np.empty_like(self.Cells["Z"])
        self.vars[NewVarZ]["val"] = \
            0.5 * (self.vars[VarNameZ]["val"][:, :, :-1] + self.vars[VarNameZ]["val"][:, :, 1:])
        return

    def AvgEdgeToCell(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
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

        self.vars[NewVarX] = dict()
        self.vars[NewVarX]["Location"] = "Cells"
        self.vars[NewVarX]["val"] = np.empty_like(self.Cells["X"])
        self.vars[NewVarX]["val"] = \
            0.25 * (self.vars[VarNameX]["val"][:, :-1, :-1] + self.vars[VarNameX]["val"][:, 1:, 1:]
                    + self.vars[VarNameX]["val"][:, 1:, :-1] + self.vars[VarNameX]["val"][:, :-1, 1:])

        self.vars[NewVarY] = dict()
        self.vars[NewVarY]["Location"] = "Cells"
        self.vars[NewVarY]["val"] = np.empty_like(self.Cells["Y"])
        self.vars[NewVarY]["val"] = \
            0.25 * (self.vars[VarNameY]["val"][1:, :, 1:] + self.vars[VarNameY]["val"][:-1, :, :-1]
                    + self.vars[VarNameY]["val"][1:, :, :-1] + self.vars[VarNameY]["val"][:-1, :, 1:])

        self.vars[NewVarZ] = dict()
        self.vars[NewVarZ]["Location"] = "Cells"
        self.vars[NewVarZ]["val"] = np.empty_like(self.Cells["Z"])
        self.vars[NewVarZ]["val"] = \
            0.25 * (self.vars[VarNameZ]["val"][1:, 1:, :] + self.vars[VarNameZ]["val"][:-1, :-1, :]
                    + self.vars[VarNameZ]["val"][1:, :-1, :] + self.vars[VarNameZ]["val"][:-1, 1:, :])
        return

    def Cross(self, Var1X, Var1Y, Var1Z, Var2X, Var2Y, Var2Z, ResX, ResY, ResZ):
        """

        :param Var1X:
        :type Var1X:
        :param Var1Y:
        :type Var1Y:
        :param Var1Z:
        :type Var1Z:
        :param Var2X:
        :type Var2X:
        :param Var2Y:
        :type Var2Y:
        :param Var2Z:
        :type Var2Z:
        :param ResX:
        :type ResX:
        :param ResY:
        :type ResY:
        :param ResZ:
        :type ResZ:
        :return:
        :rtype:
        """
        self.vars[ResX] = dict()
        self.vars[ResX]["Location"] = "Cells"
        self.vars[ResX]["val"] = self.vars[Var1Y]["val"] * self.vars[Var2Z]["val"] - \
            self.vars[Var1Z]["val"] * self.vars[Var2Y]["val"]

        self.vars[ResY] = dict()
        self.vars[ResY]["Location"] = "Cells"
        self.vars[ResY]["val"] = self.vars[Var1Z]["val"] * self.vars[Var2X]["val"] - \
            self.vars[Var1X]["val"] * self.vars[Var2Z]["val"]

        self.vars[ResZ] = dict()
        self.vars[ResZ]["Location"] = "Cells"
        self.vars[ResZ]["val"] = self.vars[Var1X]["val"] * self.vars[Var2Y]["val"] - \
            self.vars[Var1Y]["val"] * self.vars[Var2X]["val"]

        return

    def Write2HDF5(self, filename, BaseAddress: str='./') -> None:
        """
        Parameters
        ----------
        filename : str
            Name of the file to save with ``.h5`` extension

        BaseAddress : str
            Relative or absolute path in which file must be saved in,
            default value is current directory
        """

        fout = h5.File(BaseAddress+filename, 'w')
        Vars = fout.create_group('vars')
        Vars.attrs["Time"] = 0.0
        fpXMF = open(BaseAddress+filename+".xmf", 'wt')
        print(self.XMFHeader.format(*self.Nodes['X'].shape), file=fpXMF)
        print(self.Geometry.format(*self.Nodes['X'].shape, filename), file=fpXMF)

        for k in self.vars.keys():
            Vars.create_dataset(k, data=np.transpose(self.vars[k]["val"]))
            if self.vars[k]['Location'] == "Cells":
                print(self.Attribute.format(k, "Cell", *self.vars[k]['val'].shape, filename, 'vars'),
                      file=fpXMF)
            elif self.vars[k]['Location'] in ["EdgeX", "EdgeY", "EdgeZ"]:
                print(self.Attribute.format(k, "Edge", *self.vars[k]['val'].shape, filename, 'vars'),
                      file=fpXMF)
            elif self.vars[k]['Location'] in ["FaceX", "FaceY", "FaceZ"]:
                print(self.Attribute.format(k, "Face", *self.vars[k]['val'].shape, filename, 'vars'),
                      file=fpXMF)
            elif self.vars[k]['Location'] == "Nodes":
                print(self.Attribute.format(k, "Node", *self.vars[k]['val'].shape, filename, 'vars'),
                      file=fpXMF)
        GCell = fout.create_group("cell_coords")
        GNode = fout.create_group("node_coords")
        for i in self.Direction:
            GCell.create_dataset(i, data=np.transpose(self.Cells[i]))
            # print(self.Attribute.format(i, "Cell", *self.Cells[i].shape, filename, 'cell_coords'),
            #       file=fpXMF)
            GNode.create_dataset(i, data=np.transpose(self.Nodes[i]))
        fout.close()
        print(self.XMFFooter, file=fpXMF)
        fpXMF.close()
        print("Data is written to {0},\n use the companion file ({1}) for loading the data into Paraview or Visit".
              format(BaseAddress+filename, BaseAddress+filename+".xmf"))
        return

    def ToNewMesh(self, idata2, VarName, NewLocation):
        """
        ToNewMesh Method transfers a variable from one mesh to another one,
        Two meshes may be in the same dataset (same mesh) on different mesh points
        or on two different datasets with different system of coordinates or mesh spacing.

        inputs ::
        ---------

        :param idata2: destination dataset
        :type idata2: dataset

        :param NewLocation: location of transferred variable
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
        points1X = self.__dict__[Loc1]['X'][:, 0, 0]
        points1Y = self.__dict__[Loc1]['Y'][0, :, 0]
        points1Z = self.__dict__[Loc1]['Z'][0, 0, :]

        # TODO figure out the dimension and of shape of coordinates arrays \
        # when transforming from SPH to CAR

        InterpolateFunc = RegularGridInterpolator((points1X, points1Y, points1Z), self.vars[VarName]["val"])

        idata2.vars[VarName] = dict()
        idata2.vars[VarName]["Location"] = NewLocation
        points2X = idata2.__dict__[NewLocation]['X'][:, 0, 0]
        points2Y = idata2.__dict__[NewLocation]['Y'][0, :, 0]
        points2Z = idata2.__dict__[NewLocation]['Z'][0, 0, :]
        OutputShape = idata2.__dict__[NewLocation]['X'].shape

        print(idata2.__dict__[NewLocation]['X'].shape,
              self.__dict__[Loc1]['X'].shape,
              NewLocation)

        print("Last - 1")
        Points2 = np.vstack(np.meshgrid(points2X, points2Y, points2Z, indexing="ij")).reshape(3, -1).T
        print("Last")
        idata2.vars[VarName]["val"] = InterpolateFunc(Points2).reshape(OutputShape)
        return

    def FindGeometry(self, BlockSize, BaseAddress, Pattern, StartingBlock):
        """
        FindGeometry finds out the mesh structure, the data
        stored in files, and also where the data is located on
        the mesh (Cell, Node, Face, or Edge)

        Input:
        -----
        :param BlockSize: (type int)
            number of files in each blocks
        :param BaseAddress: (type str, Defaul ".", default is set in class)
            Directory of files to load
        :param Pattern: (type str, Defaul "*.npz", default is set in class)
            Pattern of file names to load
        :param StartingBlock: (type int)
            Number of block (block of files) to load

        Output:
        -------

        """

        FilesNames = fnmatch.filter(os.listdir(BaseAddress), Pattern)
        FilesNames.sort()
        NumFiles = len(FilesNames)
        if (NumFiles // BlockSize) * BlockSize != NumFiles:
            sys.exit("---<( Error :: some file are missing )>---")
        elif NumFiles == 0:
            sys.exit("---<( No file found )>---")

        Geo = dict()  # everything will store here
        xs, ys, zs = [], [], []
        # extract important info from files in a block
        f = BaseAddress + FilesNames[0]
        data = np.load(f, allow_pickle=True, encoding="bytes")
        xs, ys, zs = data["x"], data["y"], data["z"]
        try:
            Geo['num_ghost_cells'] = data.f.num_ghost_cells
            print("---<( Input files include Ghost-Cells, those cells will be ignored )>---")
        except:
            print("---<( Input files did not include Ghost-Cells )>---")
            Geo['num_ghost_cells'] = np.array([0, 0, 0], dtype=np.int)

        for ii in range(1, BlockSize):
            f = BaseAddress + FilesNames[ii]
            data = np.load(f, allow_pickle=True, encoding="bytes")
            xs = np.concatenate((xs, data["x"]))
            ys = np.concatenate((ys, data["y"]))
            zs = np.concatenate((zs, data["z"]))

        if Geo["num_ghost_cells"][0] > 0:
            xBeg = Geo["num_ghost_cells"][0]
            Geo["x"] = np.unique(xs)[xBeg:-xBeg]  # find unique values and sort them
        else:
            Geo["x"] = np.unique(xs)  # find unique values and sort them

        if Geo["num_ghost_cells"][1] > 0:
            yBeg = Geo["num_ghost_cells"][1]
            Geo["y"] = np.unique(ys)[yBeg:-yBeg]  # find unique values and sort them
        else:
            Geo["y"] = np.unique(ys)  # find unique values and sort them

        if Geo["num_ghost_cells"][2] > 0:
            zBeg = Geo["num_ghost_cells"][2]
            Geo["z"] = np.unique(zs)[zBeg:-zBeg]  # find unique values and sort them
        else:
            Geo["z"] = np.unique(zs)  # find unique values and sort them

        # to check the location of var (Face, Cell, Edge, Node)
        Nodes = (data["x"].size, data["y"].size, data["z"].size)
        Cells = (data["x"].size - 1, data["y"].size - 1, data["z"].size - 1)

        FaceX = (data["x"].size, data["y"].size - 1, data["z"].size - 1)
        FaceY = (data["x"].size - 1, data["y"].size, data["z"].size - 1)
        FaceZ = (data["x"].size - 1, data["y"].size - 1, data["z"].size)

        EdgeX = (data["x"].size - 1, data["y"].size, data["z"].size)
        EdgeY = (data["x"].size, data["y"].size - 1, data["z"].size)
        EdgeZ = (data["x"].size, data["y"].size, data["z"].size - 1)

        Geo["vars"] = dict()
        for t in data.files:
            if isinstance(data[t], np.ndarray):
                if len(data[t].shape) == 3:
                    Geo["vars"][t] = dict()
                    tLocation = data[t].shape
                    if tLocation == Nodes:
                        Geo["vars"][t]["Location"] = "Nodes"
                    elif tLocation == Cells:
                        Geo["vars"][t]["Location"] = "Cells"
                    elif tLocation == FaceX:
                        Geo["vars"][t]["Location"] = "FaceX"
                    elif tLocation == FaceY:
                        Geo["vars"][t]["Location"] = "FaceY"
                    elif tLocation == FaceZ:
                        Geo["vars"][t]["Location"] = "FaceZ"
                    elif tLocation == EdgeX:
                        Geo["vars"][t]["Location"] = "EdgeX"
                    elif tLocation == EdgeY:
                        Geo["vars"][t]["Location"] = "EdgeY"
                    elif tLocation == EdgeZ:
                        Geo["vars"][t]["Location"] = "EdgeZ"
                    else:
                        print("Can not find the location for : ", t)
                        del Geo["vars"][t]

        Geo["Files"] = FilesNames[(StartingBlock * BlockSize):(StartingBlock * BlockSize + BlockSize)]
        return Geo

    def Adummy(self, Coord):
        """
        Adummy returns an array with same shape as input coordinate array
        with zero values.
        """
        return np.zeros_like(Coord['X'], dtype=np.float64)

    def ExtractAFromBFace(self, BxName, ByName, BzName, AxName, AyName, AzName):
        """

        :param BxName:
        :type BxName:
        :param ByName:
        :type ByName:
        :param BzName:
        :type BzName:
        :param AxName:
        :type AxName:
        :param AyName:
        :type AyName:
        :param AzName:
        :type AzName:
        :return:
        :rtype:
        """
        self.Scalar(AxName, "EdgeX", self.Adummy)
        self.Scalar(AyName, "EdgeY", self.Adummy)
        self.Scalar(AzName, "EdgeZ", self.Adummy)

        dx = self.nodevect['X'][1:] - self.nodevect['X'][:-1]
        dy = self.nodevect['Y'][1:] - self.nodevect['Y'][:-1]
        dz = self.nodevect['Z'][1:] - self.nodevect['Z'][:-1]
        BetaX = np.empty_like(self.EdgeX["X"][:, :, 0])
        BetaY = np.empty_like(self.EdgeY["Y"][:, :, 0])

        BetaX[:, 0] = 0.0
        BetaY[0, :] = 0.0
        BzOnTop = self.vars[BzName]['val'][:, :, -1]
        for j in range(1, dy.size + 1):
            for i in range(dx.size):
                BetaX[i, j] = -0.5 * np.sum(dy[:j] * BzOnTop[i, :j])

        for i in range(1, dx.size + 1):
            for j in range(dy.size):
                BetaY[i, j] = 0.5 * np.sum(dx[:i] * BzOnTop[:i, j])

        self.vars[AxName]['val'][:, :, -1] = BetaX[:, :]
        self.vars[AyName]['val'][:, :, -1] = BetaY[:, :]

        for k in range(-2, -dz.size-2, -1):
            BetaX[:, :] = BetaX[:, :] - dz[k + 1] * self.vars[ByName]['val'][:, :, k + 1]
            BetaY[:, :] = BetaY[:, :] + dz[k + 1] * self.vars[BxName]['val'][:, :, k + 1]
            self.vars[AxName]['val'][:, :, k] = BetaX[:, :]
            self.vars[AyName]['val'][:, :, k] = BetaY[:, :]
        return

    def LoadData(self):
        # Check number of Ghost-Cells in each axis and use as Starting index for input arrays
        BegX = self.geo['num_ghost_cells'][0]
        BegY = self.geo['num_ghost_cells'][1]
        BegZ = self.geo['num_ghost_cells'][2]

        # Allocating the memory for parameters stored in input file/s
        for v in self.geo["vars"].keys():
            self.Scalar(v, self.geo["vars"][v]["Location"], self.Adummy, InitialVal=-1.0e10)

        for f in self.geo["Files"]:
            data = np.load(self.BaseAddress + f, allow_pickle=True, encoding="bytes")
            # Check number of points in input arrays, remove Ghost-Cells and use as ending index in each axis
            EndX = data["x"].size - BegX - 1
            EndY = data["y"].size - BegY - 1
            EndZ = data["z"].size - BegZ - 1

            SPX = np.searchsorted(self.nodevect[self.Direction[0]], data["x"][BegX])
            EPX = np.searchsorted(self.nodevect[self.Direction[0]], data["x"][-1 - BegX], side='right')

            SPY = np.searchsorted(self.nodevect[self.Direction[1]], data["y"][BegY])
            EPY = np.searchsorted(self.nodevect[self.Direction[1]], data["y"][-1 - BegY], side='right')

            SPZ = np.searchsorted(self.nodevect[self.Direction[2]], data["z"][BegZ])
            EPZ = np.searchsorted(self.nodevect[self.Direction[2]], data["z"][-1 - BegZ], side='right')
            for v in self.geo["vars"].keys():
                ExtraX, ExtraY, ExtraZ = 0, 0, 0
                if self.geo["vars"][v]['Location'] in ['FaceX', 'EdgeY', 'EdgeZ']:
                    ExtraX = 1
                if self.geo["vars"][v]['Location'] in ['FaceY', 'EdgeZ', 'EdgeX']:
                    ExtraY = 1
                if self.geo["vars"][v]['Location'] in ['FaceZ', 'EdgeX', 'EdgeY']:
                    ExtraZ = 1
                # print(self.geo["vars"][v]['Location'])
                # print("Where to put :: ",
                #       SPX, EPX+ExtraX-1, SPY, EPY+ExtraY-1, SPZ, EPZ+ExtraZ-1)
                # print("Where to pick :: ",
                #       BegX, EndX + ExtraX, BegY, EndY + ExtraY, BegZ, EndZ + ExtraZ)
                self.vars[v]["val"][SPX:(EPX+ExtraX-1), SPY:(EPY+ExtraY-1), SPZ:(EPZ+ExtraZ-1)] = \
                    data[v][BegX:(EndX + ExtraX), BegY:(EndY + ExtraY), BegZ:(EndZ + ExtraZ)]
        return

    def LoadVTR(self, Pattern: str, BaseAddress: str="", Variables: list=None, Scale: float=1.0):
        """

        :param Pattern:
        :type Pattern:
        :param BaseAddress:
        :type BaseAddress:
        :param Variables:
        :type Variables:
        :return:
        :rtype:
        """
        FilesNames = fnmatch.filter(os.listdir(BaseAddress), Pattern)
        FilesNames.sort()
        NumFiles = len(FilesNames)
        OnNode = []
        OnCell = []

        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(BaseAddress+FilesNames[0])
        reader.Update()
        data = reader.GetOutput()
        for i in range(data.GetPointData().GetNumberOfArrays()):
            var = data.GetPointData().GetArrayName(i)
            if (Variables is None) or (var in Variables):
                OnNode.append(var)

        for i in range(data.GetCellData().GetNumberOfArrays()):
            var = data.GetCellData().GetArrayName(i)
            if (Variables is None) or (var in Variables):
                OnCell.append(var)

        # Allocating the memory for parameters stored in input file/s
        print("Following parameters are found in the given VTR file/s :: ")
        for var in OnNode:
            print(" {0} (defined on Nodes)".format(var))
            self.Scalar(var, "Nodes", self.Adummy, InitialVal=-1.0e10)
        for var in OnCell:
            print(" {0} (defined on Cells)".format(var))
            self.Scalar(var, "Cells", self.Adummy, InitialVal=-1.0e10)

        for f in FilesNames:
            # First find Data and the their location in vtr files
            reader = vtk.vtkXMLRectilinearGridReader()
            reader.SetFileName(BaseAddress+f)
            reader.Update()
            data = reader.GetOutput()
            x = np.array(data.GetXCoordinates()) * Scale
            y = np.array(data.GetYCoordinates()) * Scale
            z = np.array(data.GetZCoordinates()) * Scale
            nx = x.size
            ny = y.size
            nz = z.size

            SPX = np.searchsorted(self.nodevect[self.Direction[0]], x[0])
            EPX = np.searchsorted(self.nodevect[self.Direction[0]], x[-1], side='right')

            SPY = np.searchsorted(self.nodevect[self.Direction[1]], y[0])
            EPY = np.searchsorted(self.nodevect[self.Direction[1]], y[-1], side='right')

            SPZ = np.searchsorted(self.nodevect[self.Direction[2]], z[0])
            EPZ = np.searchsorted(self.nodevect[self.Direction[2]], z[-1], side='right') #

            for v in OnNode:
                # print("Loading {0} in Nodes... ".format(v))
                self.vars[v]["val"][SPX:EPX, SPY:EPY, SPZ:EPZ] = \
                    np.array(data.GetPointData().GetArray(v)).reshape((nx, ny, nz), order='F')

            for v in OnCell:
                # print("Loading {0} in Cells... ".format(v))
                self.vars[v]["val"][SPX:(EPX-1), SPY:(EPY-1), SPZ:(EPZ-1)] = \
                    np.array(data.GetCellData().GetArray(v)).reshape((nx-1, ny-1, nz-1), order='F')

        return

    def ToPluto(self, Vars: list=[], Address: str = "./", BaseName: str = ""):
        """

        Parameters
        ----------
        Vars :
        Address :
        BaseName :

        Returns
        -------

        """
        if len(Vars) == 0:
            Vars = list(self.vars.keys())
        flog = open(Address+BaseName+"to_pluto.log", "wt")
        fgrid = open(Address+BaseName+"GRID.dat", "wt")
        print("Grid file :\n {0}".format(Address+BaseName+"GRID.dat"), file=flog)
        print("Number of Cells : ", self.NCell, file=flog)

        print("# GEOMETRY:   CARTESIAN", end='\n', file=fgrid)
        for d in self.Direction:
            nCells = self.nodevect[d].size-1
            print("{0} : {1:12.6e}   {2:12.6e}".format(d, self.nodevect[d][0], self.nodevect[d][-1]),
                  file=flog)
            print("{0}".format(nCells), end='\n', file=fgrid)
            for p in range(nCells):
                print("{0:d}   {1:12.6e}  {2:12.6e}".
                      format(p+1, self.nodevect[d][p], self.nodevect[d][p+1]),
                      end='\n', file=fgrid)

        fgrid.close()
        for v in Vars:
            if self.vars[v]["Location"] == "Cells":
                OutFile = Address + BaseName + v + '.dbl'
                self.vars[v]["val"].T.tofile(OutFile)
                print('\n', v, " is stored in ", OutFile, end='\n', file=flog)
            elif self.vars[v]["Location"] == "FaceX":
                OutFile = Address + BaseName + v + '_FaceX.dbl'
                self.vars[v]["val"].T.tofile(OutFile)
                print('\n', v, " (on FaceX) is stored in ", OutFile, end='\n', file=flog)

                OutFile = Address + BaseName + v + '_Avg.dbl'
                (0.5 * (self.vars[v]["val"][:-1, :, :] + self.vars[v]["val"][1:, :, :])).T.tofile(OutFile)
                print('\n', v, " (Averaged to Cell) is stored in ", OutFile, end='\n', file=flog)
            elif self.vars[v]["Location"] == "FaceY":
                OutFile = Address + BaseName + v + '_FaceY.dbl'
                self.vars[v]["val"].T.tofile(OutFile)
                print('\n', v, " (on FaceY) is stored in ", OutFile, end='\n', file=flog)

                OutFile = Address + BaseName + v + '_Avg.dbl'
                (0.5 * (self.vars[v]["val"][:, :-1, :] + self.vars[v]["val"][:, 1:, :])).T.tofile(OutFile)
                print('\n', v, " (Averaged to Cell) is stored in ", OutFile, end='\n', file=flog)
            elif self.vars[v]["Location"] == "FaceZ":
                OutFile = Address + BaseName + v + '_FaceZ.dbl'
                self.vars[v]["val"].T.tofile(OutFile)
                print('\n', v, " (on FaceZ) is stored in ", OutFile, end='\n', file=flog)

                OutFile = Address + BaseName + v + '_Avg.dbl'
                (0.5 * (self.vars[v]["val"][:, :, :-1] + self.vars[v]["val"][:, :, 1:])).T.tofile(OutFile)
                print('\n', v, " (Averaged to Cell) is stored in ", OutFile, end='\n', file=flog)
            elif self.vars[v]["Location"] == "Nodes":
                OutFile = Address + BaseName + v + '_Nodes.dbl'
                self.vars[v]["val"].T.tofile(OutFile)
                print('\n', v, " (on Nodes) is stored in ", OutFile, end='\n', file=flog)

                OutFile = Address + BaseName + v + '_Nodes_Avg.dbl'
                (0.125 * (self.vars[v]["val"][:-1, :-1, :-1] + self.vars[v]["val"][:-1, :-1, 1:] +
                          self.vars[v]["val"][:-1, 1:, :-1] + self.vars[v]["val"][1:, :-1, :-1] +
                          self.vars[v]["val"][1:, 1:, :-1] + self.vars[v]["val"][1:, :-1, 1:] +
                          self.vars[v]["val"][:-1, 1:, 1:] + self.vars[v]["val"][1:, 1:, 1:])
                 ).T.tofile(OutFile)
                print('\n', v, " (Averaged to Cell) is stored in ", OutFile, end='\n', file=flog)
            else:
                print('\n', "Can not store ", v, ", Location is not supported for now", end='\n', file=flog)

        flog.close()
        return
