#   Script to compute the Bragg components of elastic strains for all
#   measured grains. Makes use of LinearElasticity module.

import numpy as np
import scipy.io as sio
import LinearElasticity as elast
from tqdm import tqdm

dataRoot = '../data'
peaks = [ 
    [ 'stdSample', False ], 
    [ '111_A', False ], 
    [ '111_B', False ],
    [ '200_D', False ]
]
bases = sio.loadmat( '%s/bases_FINAL.mat'%dataRoot )

estimatedStrain = {}
for thisPeak in tqdm( peaks ):
    rho = sio.loadmat( '%s/%s_solution.mat'%( dataRoot, thisPeak[0] ) )[ 'rho' ]
    sup = sio.loadmat( '%s/%s_solution.mat'%( dataRoot, thisPeak[0] ) )[ 'support' ]
    if thisPeak[1]==True:
        rho = rho[ ::-1, ::-1, ::-1 ]
        rho = np.absolute( rho ) * np.exp( -1j * np.angle( rho ) )
        sup = sup[ ::-1, ::-1, ::-1 ]

    angles = np.angle( rho )
    unwrapThesePixels = np.where( 
        np.logical_and( 
            np.absolute( rho ) > 0., 
            np.angle( rho ) < 0.
        )
    )
    angles[ unwrapThesePixels ] = 2.*np.pi + angles[ unwrapThesePixels ]
    
    Breal = bases[ 'real_%s'%thisPeak[0] ]
    Q = bases[ 'Ghkl_%s'%thisPeak[0] ]
    Q = Q / 1.e9    # scaling to units of inverse nanometers
    strainComponent = elast.computeStrain( 
        phaseField=angles*sup, 
        Q=Q, 
        basis=Breal, 
        smooth=True
#        smooth=False
    )

    estimatedStrain[ 'rho_%s'%thisPeak[0] ] = rho
    estimatedStrain[ 'sup_%s'%thisPeak[0] ] = sup
    estimatedStrain[ 'strain_%s'%thisPeak[0] ] = strainComponent

sio.savemat( '%s/estimatedStrains_possibleError.mat'%dataRoot, estimatedStrain )
print( 'Done. ' )
