#!/usr/bin/env python
from DataSet import DataSet as DS
import astropy.constants as Const
import numpy as np

class ModelParameters(object): pass


BaseAddress = "/home/sigma/PycharmProjects/"
ds1 = DS(SystemOfCoords="CAR", BaseAddress=BaseAddress,
         Pattern="div_E_omega_2017-09-06T09-24-00.npz",
         NBlocks=1, UseBlock=0)
# print("Number of cells in each direction : ", ds1.NCell)
# # Note :: After importing ```vtr``` files, B[xyz] are located on the cell centers and magnetic fields
# #         on cell faces are renamed to B[xyz]_on_Face[XYZ]
ds1.LoadVTR(Pattern="div_E_omega_2017-09-06T09-24-00.vtr",
            BaseAddress=BaseAddress, Scale=1.e6)

for v in ['Jx', 'Jy', 'Jz', 'nonohmic_resistivity', 'Bx', 'By', 'Bz']:
    del ds1.vars[v]


NX, NY, NZ = ds1.NCell

DividZAtIndx = 25
NZ1 = 2 * DividZAtIndx
NZ2 = NZ - DividZAtIndx
FStart = [ds1.nodevect['X'][0] , ds1.nodevect['Y'][0] , [ds1.nodevect['Z'][0]           , ds1.nodevect['Z'][DividZAtIndx]]]
FEnd   = [ds1.nodevect['X'][-1], ds1.nodevect['Y'][-1], [ds1.nodevect['Z'][DividZAtIndx], ds1.nodevect['Z'][-1]]]
NCell = [NX, NY, [NZ1, NZ2]]
Functions = ['linear', 'linear', ['linear', 'linear']]

ds2 = DS(SystemOfCoords="CAR", NCell=NCell, Functions=Functions, startval=FStart, endval=FEnd)

ds1.ExtractAFromBFace('Bx_FaceX', 'By_FaceY', 'Bz_FaceZ', "Ax", "Ay", "Az")
#
# # def ToNewMesh(self, idata2, VarName, NewLocation):
ds1.ToNewMesh(ds2, "Ax", "EdgeX")
ds1.ToNewMesh(ds2, "Ay", "EdgeY")
ds2.Scalar("Az", "EdgeZ", ds2.Adummy, InitialVal=0.0)

ds1.ToNewMesh(ds2, "vx", "Nodes")
ds1.ToNewMesh(ds2, "vy", "Nodes")
ds1.ToNewMesh(ds2, "vz", "Nodes")

ds1.Write2HDF5("div_E_omega_2017-09-06T09-24-00_Big.h5", BaseAddress="/home/sigma/PLUTO/PLUTOTest/LowGammaFineMesh/")

del ds1

# CurlEdgeToFace(self, VarNameX, VarNameY, VarNameZ, NewVarX, NewVarY, NewVarZ):
ds2.CurlEdgeToFace("Ax", "Ay", "Az", "Bx", "By", "Bz")

# # Calculating Mean Molecular Weight(Mu)
H_MASS_FRAC = 0.7110  # Fraction of mass by Hydrogen
He_MASS_FRAC = 0.2741  # Fraction of mass by Helium
CONST_AH = 1.008  # Atomic weight of Hydrogen
CONST_AZ = 30.0  # Mean atomic weight of heavy elements
CONST_AHe = 4.004  # Atomic weight of Helium
Z_MASS_FRAC = (1.0 - H_MASS_FRAC - He_MASS_FRAC)
FRAC_He = (He_MASS_FRAC / CONST_AHe * CONST_AH / H_MASS_FRAC)
FRAC_Z = (Z_MASS_FRAC / CONST_AZ * CONST_AH / H_MASS_FRAC)
Mu = (CONST_AH + FRAC_He * CONST_AHe + FRAC_Z * CONST_AZ) / (2.0 + FRAC_He + FRAC_Z * (1.0 + CONST_AZ * 0.5))
print(u"\u03BC =", Mu)


params = ModelParameters()
params.r0 = Const.R_sun.si.value  # in meters
params.mean_molecular_mass = Mu * Const.m_p.si.value
params.m = params.mean_molecular_mass
params.T0 = 1.0e6  # 1 MK
params.MU0 = Const.mu0.si.value
params.VAlfven = 1.0e5  # m * s^{-1} ==> 100 km / s

ds2.Scalar('rho', 'Cells', ds2.Adummy, InitialVal=np.nan)
ds2.Scalar('P', 'Cells', ds2.Adummy, InitialVal=np.nan)

B2 = 0.25 * (ds2.vars['Bx']['val'][:-1, :, :] + ds2.vars['Bx']['val'][1:, :, :]) ** 2 + \
     0.25 * (ds2.vars['By']['val'][:, :-1, :] + ds2.vars['By']['val'][:, 1:, :]) ** 2 + \
     0.25 * (ds2.vars['Bz']['val'][:, :, :-1] + ds2.vars['Bz']['val'][:, :, 1:]) ** 2

ds2.vars['rho']['val'][:, :, :] = B2 / (params.MU0 * params.VAlfven ** 2)
ds2.vars['P']['val'][:, :, :] = ds2.vars['rho']['val'] * params.T0 * Const.k_B.si.value / params.m

for c in ds1.Direction:
    ds1.nodevect[c] *= 1.e-8
ds2.ToPluto(Address="/home/sigma/PLUTO/PLUTOTest/LowGammaFineMesh/")
ds2.Write2HDF5("div_E_omega_2017-09-06T09-24-00_Small.h5", BaseAddress="/home/sigma/PLUTO/PLUTOTest/LowGammaFineMesh/")
