#   Phase retriever class for low- and high-energy coherent diffraction data.
#   Modded tor GPU acceleration using Tensorflow.
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    2017-2018

import numpy as np
from numpy.fft import fftshift, fftn, ifftn
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

import GPUModule as accelerator

import SuperSampling as ss

###############################################################################################

#  The projector and reflector operators defined here are heavily used in the solver.

###############################################################################################

class PhaseRetriever:

    def __init__( self,
            modulus,
            support,            # MAKE SURE THAT SUPPORT ARRAY DIMS ARE EQUAL TO binning*modulus.shape
            beta=0.9, 
            binning=1,          # set this to the desired PBF for high-energy measurements
            averaging=True,
            gpu=False,
            num_gpus=0,         # Phase retrieval and partial coherence correction on different GPUs. How much does this affect performance?
            random_start=True, 
            initial_pcc_guess=[],
            pcc_learning_rate=1.e-3,    # default learning rate for the Gaussian partial coherence function optimization
            outlog = ''         # directory name for computational graph (GPU only).
            ):
        self._modulus           = fftshift( modulus )
        self._support           = support   # user should ensure this is approriate for binned data
        self._beta              = beta
        self._binning           = binning
        self._averaging         = averaging

        self._modulus_sum       = modulus.sum()
        self._support_comp      = 1. - support
        if random_start:
            self._cImage            = np.exp( 2.j * np.pi * np.random.rand( 
                                    binning*self._modulus.shape[0], 
                                    binning*self._modulus.shape[1], 
                                            self._modulus.shape[2]
                                    ) ) * self._support
        else:
            self._cImage            = 1. * support

        self._initial_pcc_guess = initial_pcc_guess
        self._pcc_learning_rate = pcc_learning_rate
        
        if self._binning > 1:
            self._binLeft, self._binRight = bng.fastInPlaneBinning( self._cImage[:,:,0], self._binning, self._binning )[1:]
            self._scale =   ( self._binLeft[ 0,:] > 1.e-6 ).astype( float ).sum() *\
                            ( self._binRight[0,:] > 1.e-6 ).astype( float ).sum()
            self._cImage_fft_mod = np.sqrt( 
                    ss.inPlaneBinning( 
                        np.absolute( fftn( self._cImage ) )**2, 
                        self._binning, 
                        self._binning 
                    ) 
                )
        else:
            self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )

        self._error             = []
        self._UpdateError()

        if gpu==True:
            self.gpusolver = accelerator.Solver( self.generateVariableDict(), num_gpus=num_gpus, outlog=outlog )

        if self._averaging:
            self._prefactor = 1.
        else:
            self._prefactor = 1. / ( self._binning**2 )

# Writer function to manually update support
    def UpdateSupport( self, support ):
        self._support = support
        self._support_comp = 1. - self._support
        return

# Writer function to manualy reset image
    def ImageRestart( self, cImg, reset_error=True ):
        self._cImage = cImg
        if reset_error:
            self._error = []
        return

# Reader function for the retrieved image
    def Image( self ):
        return self._cImage

# Reader function for the final computed modulus
    def Modulus( self ):
#        return np.absolute( fftshift( fftn( fftshift( self._cImage ) ) ) )
        return np.absolute( fftshift( fftn( self._cImage ) ) )

# Reader function for the error metric
    def Error( self ):
        return self._error

# Now, defining the phase retrieval algorithms based on the object metadata.

# Updating the error metric
    def _UpdateError( self ):
        self._error += [ ( ( self._cImage_fft_mod - self._modulus )**2 * self._modulus ).sum() / self._modulus_sum ]
        return

# Error reduction algorithm
    def ErrorReduction( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ) ):
            self._ModProject()
            self._cImage *= self._support
            self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )
            self._UpdateError()
        return

# Hybrid input/output algorithm
    def HybridIO( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ) ):
            origImage = self._cImage.copy() 
            self._ModProject()
            self._cImage = ( self._support * self._cImage ) + self._support_comp * ( origImage - self._beta * self._cImage )
            self._UpdateError()
        return

# Solvent flipping algorithm
    def SolventFlipping( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ) ):
            self._ModHatProject()
            self._ModProject()
            self._UpdateError()
        return

# High-energy error reduction
    def HEErrorReduction( self, num_iterations ):
        for i in tqdm( list( range(num_iterations ) ) ):
            self._HEModProject()
            self._cImage *= self._support
            self._UpdateError()
        return

# High-energy hybrid input/output algorithm
    def HEHybridIO( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ) ):
            origImage = self._cImage.copy() 
            self._HEModProject()
            self._cImage = ( self._support * self._cImage ) + self._support_comp * ( origImage - self._beta * self._cImage )
            self._UpdateError()
        return

# Basic shrinkwrap with gaussian blurring
    def ShrinkWrap( self, sigma, thresh ):
        result = gaussian_filter( np.absolute( self._cImage ), sigma, mode='constant', cval=0. )
        self._support = ( result > thresh*result.max() ).astype( float )
        self._support_comp = 1. - self._support
        return

# The projection operator into the modulus space of the FFT.
# This is a highly nonlinear operator.
    def _ModProject( self ):
        self._cImage = ifftn( self._modulus * np.exp( 1j * np.angle( fftn( self._cImage ) ) ) )
        return

# The reflection operator in the plane of the (linear)
# support operator. This operator is also linear.
    def _SupReflect( self ):
        self._cImage = 2.*( self._support * self._cImage ) - self._cImage
        return

# The projection operator into the 'mirror image' of the
# ModProject operator in the plane of the support projection
# operator. The involvement of the ModProject operator
# makes this also a highly nonlinear operator.
    def _ModHatProject( self ):
        self._SupReflect()
        self._ModProject()
        self._SupReflect()
        return

# High-energy modulus projection operator, analogous to _ModProject
    def _HEModProject( self ):
        fimg = fftn( self._cImage )
        thisAbs = np.absolute( fimg )
        self._cImage_fft_mod = np.sqrt( ss.inPlaneBinning( thisAbs**2, self._binning, self._binning, averaging=self._averaging ) )
        scl = np.divide( self._modulus, self._cImage_fft_mod )
#        self._cImage = ifftn( self._scale * ss.inPlaneExpansion( scl, self._binLeft, self._binRight, prefactor=self._prefactor ) * fimg )
        self._cImage = ifftn( ss.inPlaneExpansion( scl, self._binLeft, self._binRight, prefactor=self._prefactor ) * fimg )
        return

# Generates a package in the form of a dict for GPU module to read and generate tensors.
    def generateVariableDict( self ):
        mydict = {
            'modulus':self._modulus, 
            'support':fftshift( self._support ),
            'beta':self._beta, 
            'cImage':self._cImage, 
        }
        if self._binning > 1:       # used only in case of high-energy coherent scattering
            mydict[ 'bin_left' ] = self._binLeft
            mydict[ 'bin_right' ] = self._binRight
            mydict[ 'scale' ] = self._scale

        if len( self._initial_pcc_guess ) > 0:  # initial guess value for Gaussian partial coherence function.
            mydict[ 'initial_guess' ] = self._initial_pcc_guess
        mydict[ 'pcc_learning_rate' ] = self._pcc_learning_rate

        return mydict



