# Defines modules that resolve the scattering geometry of a 
# (partially) coherent scattering experiment from a crystalline 
# sample, performed at Beamline 1-ID of the Advanced Photon 
# Source. Determines sampling bases of real and reciprocal 
# space, given a fixed experimental configuration, which includes 
# scattering direction, detector pixel size and orientation,
# beam energy and sample-detector distance.
#
# Siddharth Maddali
# Argonne National Laboratory
# April 2018


import numpy as np

class ScatteringGeometry:
   
    def __init__( self, 
        #lam=8.2633e-11,        # wavelength, meters, corresp to 15 keV
        #lam= 1.377e-10,        # wavelength, meters, corresp to 9 keV
        lam= 2.3843e-11,        # wavelength, meters, corresp to 52 keV
        pix= 55.e-6,            # detector pixel size, meters
        arm= 6.,                # sample-detector distance, meters
        a0= 4.078e-10,          # lattice constant, meters
        peak= [ 1, 1, 1 ],      # Bragg peak of interest
        eta= -30.,              # cone azimuth rotation, degrees, about Z axis
        domega= 0.01,           # angular step in rocking curve, degrees
        recipSpaceSteps = [ 258, 258, 70 ]  # detector span + # of omegas
    ):
        self._lambda = lam
        self._pix = pix
        self._arm = arm
        self._a0 = a0
        self._peak = np.array( peak )
        self._eta = ( eta * np.pi / 180. )
        self._domega = (domega * np.pi / 180. )
        self._recip = np.array( recipSpaceSteps ).reshape( 1, -1 )

        self._ki = ( 1. / self._lambda ) * np.array( [ 0., 0., 1. ] ).reshape( -1, 1 )

        self._theta = np.arcsin( lam * np.linalg.norm( peak ) / ( 2. * self._a0 ) )
        self._twotheta = 2. * self._theta
        self._Mx = np.array( 
            [ 
                [ 1., 0., 0. ],  
                [ 0.,  np.cos( self._twotheta ), np.sin( self._twotheta ) ], 
                [ 0., -np.sin( self._twotheta ), np.cos( self._twotheta ) ]
            ]
        )
        self._Mz = np.array( 
            [ 
                [  np.cos( self._eta ), np.sin( self._eta ), 0. ], 
                [ -np.sin( self._eta ), np.cos( self._eta ), 0. ], 
                [ 0., 0., 1. ]
            ]
        )
        self._Momega = np.array( 
            [ 
                [  np.cos( self._domega ), 0., -np.sin( self._domega ) ], 
                [ 0., 1., 0. ],
                [ np.sin( self._domega ), 0.,   np.cos( self._domega ) ] 
            ]
        )

        self._kf = self._Mz @ self._Mx @ self._ki
        self._Q = self._kf - self._ki
#        self._Q *= ( 
#            2.*np.pi*np.linalg.norm(peak) / ( self._a0*np.linalg.norm(self._Q) )
#        )
        
        # dq = reciprocal space step due to sample rotation.
        self._dq = ( self._Momega @ self._Q ) - self._Q

        # det = reciprocal space steps in the detector plane.
        detTrans = ( self._pix / ( self._lambda * self._arm ) ) * self._Mz @\
            np.array( [ [ 1., 0., 0. ], [ 0., 1., 0. ] ] ).T
        
        self._recipSpaceBasis = np.concatenate( ( detTrans, -1.*self._dq ), axis=1 )

#        self._recipSpaceScope = self._recipSpaceBasis * self._recip.repeat( 3, axis=0 )
        Brange = np.diag( 1. / self._recip.ravel() )

        self._realSpaceBasis = np.linalg.inv( self._recipSpaceBasis ).T @ Brange
        
        
        # rescaling to nm for real space and nm^-1 for recip space
        self._realSpaceBasis    *= 1.e9
        self._recipSpaceBasis   *= 1.e-9

    def getSamplingBases( self ):
        return self._realSpaceBasis, self._recipSpaceBasis






            






