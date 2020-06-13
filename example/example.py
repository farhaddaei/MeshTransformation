# /usr/bin/env python

from DataSet import DataSet as DS

BaseAddress = "/home/farhadda/div_E_Omega_2011-06-06T09-36-00/"
ds1 = DS(SystemOfCoords="CAR", BaseAddress=BaseAddress,
         Pattern="div_E_Omega_2011-06-02T05-36-00_00?.npz",
         NBlocks=10, UseBlock=0)
# print("Number of cells in each direction : ", ds1.NCell)
ds1.LoadVTR(Pattern="div_E_Omega_2011-06-06T09-36-00_00?.vtr",
            BaseAddress=BaseAddress, Scale=1.e6)

for l in ds1.LocationOnGrid:
    for c in ds1.Direction:
        ds1.__dict__[l][c] *= 1.e-6

ds1.ToPluto(Address="/home/local/farhadda/PLUTO/Try/")
ds1.Write2HDF5("OhOhZiad.h5", BaseAddress="/home/local/farhadda/PLUTO/Try/")