# The grain whose Friedel pairs were imaged was 113044.
# This is confirmed from the MIDAS spreadsheets.

import numpy as np
from argparse import Namespace

# Orientation matrix of the crystal
R = np.array( 
    [ 
        [ 0.649937, -0.733845, -0.197636 ], 
        [ 0.517676,  0.617869, -0.591824 ], 
        [ 0.556417,  0.28234 ,  0.781466 ]
    ]
)

# Strain tensor in lab frame (don't use EFab) 
EKen = np.array( 
    [ 
        [ 300.956808, 4.870705, 233.956324 ], 
        [ 4.870705, 328.57759, -113.937223 ], 
        [ 233.956324, -113.937223, -150.879237 ]
    ]
)

# Transformation from APS lab frame to FABLE frame
O = np.array( 
    [ 
        [ 0., 0., 1. ], 
        [ 1., 0., 0. ], 
        [ 0., 1., 0. ]
    ]
)

Q = Namespace( **sio.loadmat( '/home/smaddali/ANL/Manuscripts/HEBCDI/data/bases_FINAL.mat' ) ).Ghkl_111_A
Q = Q / np.linalg.norm( Q )
Qc = O @ Q
print( 
    'Strain component along <111> direction is %f'%( 
        ( Qc.T @ EKen @ Qc ).ravel()[0] 
    ) 
)
