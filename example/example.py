# /usr/bin/env python

import numpy as np
from DataSet import DataSet as DS

BaseAddress = "/home/farhadda/"

file = BaseAddress+'div_E_omega_2017-09-06T11-24-00.npz'
data = np.load(file, allow_pickle=True, encoding="bytes")
ds1 = DS(SystemOfCoords="CAR", BaseAddress=BaseAddress, Pattern="div_E_omega_2017-09-06T11-24-00.npz",
         NBlocks=1, UseBlock=0)
for v in ds1.geo["vars"].keys():
    ds1.Scalar(v, ds1.geo["vars"][v]["Location"], ds1.Adummy)
    ds1.vars[v]["val"] = data[v]

del data
# Extracting Magnetic Vector Potential ($\vec{A}$) from Magnetic Field ($\vec{B}$), using $\vec{A}_{z} = 0$ as gauge
ds1.ExtractAFromBFace("Bx", "By", "Bz", "Ax", "Ay", "Az")
ds1.CurlEdgeToFace("Ax", "Ay", "Az", "BFx", "BFy", "BFz")

# ds1.CurlFaceToEdge("Bx", "By", "Bz", "Jx", "Jy", "Jz")

# ds1.AvgFaceToCell("Bx", "By", "Bz", "BXc", "BYc", "BZc")
# ds1.AvgEdgeToCell("Jx", "Jy", "Jz", "JXc", "JYc", "JZc")
# ds1.Cross("JXc", "JYc", "JZc", "BXc", "BYc", "BZc", "Fx", "Fy", "Fz")

# del ds1.vars["BXc"], ds1.vars["BYc"], ds1.vars["BZc"]
# del ds1.vars["JXc"], ds1.vars["JYc"], ds1.vars["JZc"]

# ds1.Write2HDF5(BaseAddress+"div_E_omega_2017-09-06T11-24-00.h5")
ds1.Write2HDF5(BaseAddress+"div_E_omega_2017-09-06T11-24-00.h5")

#%%

# ds2 = DS(SystemOfCoords="CAR", BaseAddress="/proj/farhadda/data/11226/Output_Sim_Omega/", Pattern="*.npz", NBlocks=20, UseBlock=0)
#
# ds1.ToNewMesh(ds2, "Ax", "EdgeX")
# ds1.ToNewMesh(ds2, "Ay", "EdgeY")
# ds2.Scalar("Az", "EdgeZ", ds2.Adummy)
