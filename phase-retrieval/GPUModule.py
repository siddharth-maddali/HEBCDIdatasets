##############################################################
#    GPU modding for FastPhaseRetriever. Always run in a 
#    virtualenv set up for tensorflow
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        2018
##############################################################

import numpy as np
import tensorflow as tf
import time
from numpy.fft import fftshift 

# Centering of image
import PostProcessing as post

# Class 'Solver' inherits methods from the mixins defined in the following modules.
import GPUModule_Base, GPUModule_ShrinkWrap
import GPUModule_ErrorReduction, GPUModule_HybridInputOutput
import GPUModule_GaussianPCF

class Solver( 
        GPUModule_Base.Mixin,
        GPUModule_ShrinkWrap.Mixin,
        GPUModule_ErrorReduction.Mixin,
        GPUModule_HybridInputOutput.Mixin,
        GPUModule_GaussianPCF.Mixin
    ):
    
    def __init__( self, varDict, num_gpus, outlog='' ):     # see FastPhaseRetriever for definition of varDict
        self.log_directory = outlog                         # directory for computational graph dump
        self.defineBaseVariables( varDict, gpu=0%num_gpus )
        self.defineShrinkwrap( varDict, gpu=0%num_gpus )
        self.defineER( gpu=0%num_gpus )
        self.defineHIO( varDict, gpu=0%num_gpus )
        self.initializeGaussianPCF( varDict, array_shape=self._probSize, gpu=1%num_gpus )
        
        self.initializeSession()
    
    def initializeSession( self ):
        config = tf.ConfigProto( allow_soft_placement=True, log_device_placement=True )
        config.gpu_options.allow_growth=True
        self.__sess__ = tf.Session( config=config )
        if len( self.log_directory ) > 0:
            writer = tf.summary.FileWriter( self.log_directory )
            writer.add_graph( self.__sess__.graph )
        self.__sess__.run( tf.global_variables_initializer() )
        return

    def Retrieve( self ):
        # thing to retrieve:
        # image (complex)
        # support (real)
        # partially coherent intensity (real)
        # Gaussian partial coherence function (real)
        # 3D Gaussian partial coherence parameters
        
        self.finalImage = fftshift( self._cImage.eval( session=self.__sess__ ) )
        self.finalSupport = fftshift( np.absolute( self._support.eval( session=self.__sess__ ) ) )
        self.finalImage, self.finalSupport = post.centerObject( self.finalImage, self.finalSupport )
        
        # partial coherence function in Fourier space
        self.finalPCC = fftshift( self._blurKernel.eval( session=self.__sess__ ) )
        self.finalPCSignal = fftshift( self._imgBlurred.eval( session=self.__sess__ ) )
        self.finalGaussPCCParams = self.__sess__.run( self._var_list )

        self.__sess__.close()
        return

   



