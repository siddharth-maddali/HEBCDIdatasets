import numpy as np


############################ USER EDIT ##################################

lam     = 1.377e-10        # wavelength, meters, corresp to 9 keV
#lam     = 2.3843e-11        # wavelength, meters, corresp to 52 keV
pix     = 55.e-6            # detector pixel size, meters
arm     = 0.6                 # sample-detector distance, meters
a0      = 4.078e-10         # lattice constant, meters
peak    = [ 1, 1, 1 ]       # Bragg peak of interest
eta     = -30.              # cone azimuth rotation, degrees, about downstream Z axis.
ki      = [ 0., 0., 1. ]    # incident wave vector
domega  = 0.01              # angular step in rocking curve, degrees
recipSpaceSteps = [ 258, 258, 70 ]  # first two are pixel widths of detector image, third is number of omegas.

#########################################################################

peak = np.array( peak )
ki = np.array( ki ).reshape( -1, 1 )
domega *= np.pi / 180.
theta =  np.arcsin( lam * np.linalg.norm( peak ) / ( 2. * a0 ) )
twotheta = 2. * theta
eta *= np.pi / 180.

Mx = np.array( 
    [ 
        [ 1., 0., 0. ], 
        [ 0.,  np.cos( twotheta ),  np.sin( twotheta ) ], 
        [ 0., -np.sin( twotheta ),  np.cos( twotheta ) ]
    ]
)
Mz = np.array( 
    [ 
        [ np.cos( eta ), -np.sin( eta ), 0. ], 
        [ np.sin( eta ),  np.cos( eta ), 0. ], 
        [ 0., 0., 1. ]
    ]
)
Momega = np.array( 
    [ 
        [  np.cos( domega ), 0., np.sin( domega ) ], 
        [ 0., 1., 0. ], 
        [ -np.sin( domega ), 0., np.cos( domega ) ]
    ]
)

kf = Mz @ Mx @ ki;
Ghat = kf - ki
Ghat *= ( 2. * np.pi * np.linalg.norm( peak ) / ( a0 * np.linalg.norm( Ghat ) ) )
dq = Momega.dot( Ghat ) - Ghat

det = np.array( 
    [ 
        [ 1., 0., 0. ], 
        [ 0., 1., 0. ]
    ]
).T  * 2. * np.pi * pix / ( lam * arm ) # detector basis
detTrans = Mz.dot( det )

detBasis = np.concatenate( ( detTrans, dq ), axis=1 )

#if np.linalg.det( detBasis ) < 0.:
#    detBasis = detBasis[ :, ::-1 ] # reverse order of vectors.

qScope = detBasis * np.array( recipSpaceSteps ).reshape( 1, -1 ).repeat(  3, axis=0 )

realBasis = 2*np.pi * np.cross( 
    qScope, np.roll( qScope, -1, axis=1 ) 
    ) / ( np.linalg.det( qScope ) ) 


print( 'Real space object sampling (meters):\n', realBasis )

print( '\n\nReal space resolution estimate (meters): \n', 
        np.power( np.linalg.det( realBasis ), 1./3. )
    )
