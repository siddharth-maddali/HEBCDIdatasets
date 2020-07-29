# Sparse recovery module for diffraction data collected during a high-resolution BDCI scan 
# at Sector 34-ID-C of the Advanced Photon Source. Input to the black box are the data set 
# itself, usually scanned at  theta steps ~1.e-3 deg, and the experimental geometry, which 
# is obtained from the corresponding .spec file.
#
# Siddharth Maddali, 2017
# Argonne National Laboratory
# smaddali@anl.gov

import numpy as np
import SuperSampling as s_samp
import gc

class imageSolver:

    def __init__( self, 
            data, lam=0.138942e-9,    
            gamma=45., delta=25.,          
            arm=0.62,  pixel=50, pixelPitch=55.e-6,
            numScans=24,
            theta=0.,  phi=-7.512, chi=90.,
            thetaInterval=0.01, phiInterval=0., chiInterval=0.
            binningFactor=4 
    ): 
# simple initializations
        self._data =            data
        self._lam =             lam
        self._gamma =           gamma * np.pi/180.
        self._delta =           delta * np.pi/180.
        self._arm =             arm
        self._pixel =           pixel
        self._pixelPitch =      pixelPitch
        self._numScans =        numScans
        self._theta =           theta * np.pi / 180.
        self._phi =             phi * np.pi/ 180.
        self._chi =             chi * np.pi / 180.
        self._thetaInteral =    thetaInterval * np.pi / 180.
        self._phiInterval =     phiInterval * np.pi / 180.
        self._chiInterval =     chiInterval * np.pi / 180.

        self._Gamma   = s_samp.generateRotationAboutPrincipalAxis( self._gamma, 'x' )
        self._Delta   = s_samp.generateRotationAboutPrincipalAxis( self._delta, 'z' )
        self._Phi     = s_samp.generateRotationAboutPrincipalAxis( self._phi, 'x' )
        self._Chi     = s_samp.generateRotationAboutPrincipalAxis( self._chi - np.pi/2., 'y' )
        self._Theta   = s_samp.generateRotationAboutPrincipalAxis( self._theta, 'z' )
        self._desiredImageSize = binningFactor * pixel

        self.initializeRotations()


    def initializeRotations():
        self._DG = Delta.dot( Gamma )             # this matrix rotates k_i to the expected k_f. Used for moving detector.
        self._R0 = Theta.dot( Chi ).dot( Phi )    # this rotation takes the recip lattice vector into the Bragg condition.
        
        self._thetaValues = np.linspace( -self._thetaInterval, self._thetaInterval, self._numScans )
        self._phiValues   = np.linspace( -self._phiInterval, self._phiInterval, self._numScans )
        self._chiValues   = np.linspace( -self._chiInterval, self._chiInterval, self._numScans ) - np.pi/2.

        self._ki = np.array( [ 0., -1., 0. ] ).reshape( -1, 1 )
        self._kf = self._DG.dot( ki )
        Ghkl = self._kf - self._ki
        Ghkl_crystalFrame = self._R0.T.dot( Ghkl )
        self._dq = np.zeros( ( 3, numScans ) )

        for i in list( range( numScans ) ):
            Phi_temp = ss.generateRotationAboutPrincipalAxis( self._phi + self._phiValues[i], 'x' )
            Theta_temp = ss.generateRotationAboutPrincipalAxis( self._theta + self._thetaValues[i], 'z' )
            Chi_temp = ss.generateRotationAboutPrincipalAxis( self._chi + self._chiValues[i], 'y' )
            R = Theta_temp.dot( Chi_temp ).dot( Phi_temp )
            self._q[:,i] = R.dot( Ghkl_crystalFrame )

        self._dq -= Ghkl.repeat( numScans, axis=1 )
        dq0 = np.diff( self._dq ).mean( axis=1 ).reshape( -1, 1 )
        basisInPlane = ( binningFactor * pixelPitch / arm ) * DG.dot( np.array( [ [ 1., 0., 0. ], [ 0., 0., 1. ] ] ).T )
            # basis is computed in units of pi / lam

        self._basis = np.concatenate( ( basisInPlane, dq0 ), axis=1 )
        self._basisInPlane = np.linalg.solve( DG, basis )
            # The last column of this matrix describes the step of each detector pixel, viewed in the detector plane.


#    def sparseRecovery( self ):


        

