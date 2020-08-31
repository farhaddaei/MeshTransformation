#!/usr/bin/env python
class ModelParameters(object): pass

from DataSet import DataSet as DS
import astropy.constants as Const
import numpy as np

BaseAddress = "/home/farhadda/div_E_Omega_2011-06-06T09-36-00/Daniel/"
ds1 = DS(SystemOfCoords="CAR", BaseAddress=BaseAddress,
         Pattern="div_E_omega_2017-09-06T09-24-00.npz",
         NBlocks=1, UseBlock=0)
# print("Number of cells in each direction : ", ds1.NCell)
ds1.LoadVTR(Pattern="div_E_omega_2017-09-06T09-24-00.vtr",
            BaseAddress=BaseAddress, Scale=1.e6)

## Calculation for   Mean Molecular Weight(Mu)
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

##
params = ModelParameters()
params.r0 = Const.R_sun.si.value  # in meters
params.mean_molecular_mass = Mu * Const.m_p.si.value
params.m = params.mean_molecular_mass
params.T0 = 1.0e6  # 1 MK
params.MU0 = Const.mu0.si.value
params.VAlfven = 1.0e5  # m * s^{-1} ==> 100 km / s

ds1.Scalar('rho', 'Cells', ds1.Adummy, InitialVal=np.nan)
ds1.Scalar('P', 'Cells', ds1.Adummy, InitialVal=np.nan)
# Note :: After importing ```vtr``` files, B[xyz] are located on the cell centers and magnetic fields
#         on cell faces are renamed to B[xyz]_on_Face[XYZ]
B2 = ds1.vars['Bx']['val'] ** 2 + ds1.vars['By']['val'] ** 2 + ds1.vars['Bz']['val'] ** 2
ds1.vars['rho']['val'][:, :, :] = B2 / (params.MU0 * params.VAlfven ** 2)
ds1.vars['P']['val'][:, :, :] = ds1.vars['rho']['val'] * params.T0 * Const.k_B.si.value / params.m

for c in ds1.Direction:
    ds1.nodevect[c] *= 1.e-8
ds1.ToPluto(Address="/home/local/farhadda/PLUTO/Try/")
ds1.Write2HDF5("div_E_omega_2017-09-06T09-24-00.h5", BaseAddress="/home/local/farhadda/PLUTO/Try/")
