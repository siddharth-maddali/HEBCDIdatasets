import numpy as np
from scipy.ndimage.filters import gaussian_filter

def computeStrain( phaseField, Q, basis=np.eye( 3 ), smooth=False, sigma=1. ):
    """
    computeStrain:
    Computes the component of the elastic strain tensor in the 
    direction of a given reciprocal lattice vector.
    
    Syntax:
    grad = computeStrain( phaseField, Q, basis=np.eye( 3 ) )
    
    phaseField: angle array (radians)
    Q:          reciprocal lattice vector in the lab frame
    basis:      real-space sampling basis
    
    WARNING: assumes that the physical dimensions of Q and basis are true reciprocals
    of each other.
    """
    Qnorm = np.linalg.norm( Q )

    phaseScaled = phaseField / ( 2. * np.pi * Qnorm )
            # this quantity has physical dimensions of length

    if smooth==True:
        phaseScaled = gaussian_filter( phaseScaled, sigma=sigma )

    steps = np.sqrt( ( basis**2 ).sum( axis=0 ) )
            # this has physical dimensions of length

    grad = np.gradient( phaseScaled, -1.*steps[1], steps[0], steps[2] )
            # this little trick is necessary because np.gradient 
            # computes in matrix coordinates, not lab coordinates.
            
    basisDirections = basis / steps.reshape( 1, -1 ).repeat( 3, axis=0 )

    gradReal = np.linalg.inv( basisDirections.T ) @ np.concatenate( 
        ( 
            grad[1].reshape( 1, -1 ), 
            grad[0].reshape( 1, -1 ), 
            grad[2].reshape( 1, -1 ) 
        ), 
        axis=0 
    )

    Qnormed = Q / Qnorm
    gradProjection = ( Qnormed.T @ gradReal ).reshape( phaseScaled.shape )

    return gradProjection


