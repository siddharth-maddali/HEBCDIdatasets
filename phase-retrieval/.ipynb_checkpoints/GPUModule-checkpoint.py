#    GPU modding for FastPhaseRetriever. Always run in a 
#    virtualenv set up for tensorflow
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        2018

import numpy as np
import tensorflow as tf
import time
from tensorflow.python.client import timeline

class Solver:
##################################################################################
#    This constructor copies data over from the parent PhaseRetriever class.
##################################################################################
    def __init__( self, varDict ):
       
        # Tensorflow variables specified here.
        config = tf.ConfigProto( allow_soft_placement=True, log_device_placement=True )
        config.gpu_options.allow_growth=True

        self.__sess__ = tf.Session( config=config )
        self.__options__ = tf.RunOptions( trace_level=tf.RunOptions.FULL_TRACE )
        self.__run_metadata__ = tf.RunMetadata()

        self._modulus = tf.Variable( varDict[ 'modulus' ], dtype=tf.complex64, name='modulus' )
        self._support = tf.Variable( varDict[ 'support' ], dtype=tf.complex64, name='support' )
        self._cImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64, name='cImage' )
        self._cImage_fft_mod = tf.Variable( varDict[ 'cImage_fft_mod' ], dtype=tf.complex64, name='cImage_fft_mod' )
        self._beta = varDict[ 'beta' ]
        self._probSize = self._cImage.shape
	
        # Array coordinates specified here.
        x, y, z = np.meshgrid( 
            list( range( varDict[ 'cImage' ].shape[0] ) ),
            list( range( varDict[ 'cImage' ].shape[1] ) ),
            list( range( varDict[ 'cImage' ].shape[2] ) )
        )
        x = ( x - x.mean() ).reshape( -1, 1 )
        y = ( y - y.mean() ).reshape( -1, 1 )
        z = ( z - z.mean() ).reshape( -1, 1 )
        pts = np.concatenate( ( x, y, z ), axis=1 )
        self._domain = [ list( n ) for n in pts ]

        # Initialize everything.
        self.__sess__.run( tf.global_variables_initializer() )

###########################################################################################
#   Regular ER
###########################################################################################
    def ErrorReduction( self, num_iterations ):
        self.__count__ = num_iterations
        self.__count__, self._cImage, self._cImage_fft_mod = tf.while_loop( 
            self.__continue__, 
            self.__ERKernel__, 
            [ self.__count__, self._cImage, self._cImage_fft_mod ], 
            parallel_iterations=1
        )
        return

###########################################################################################
#   Regular HIO
###########################################################################################
    def HybridIO( self, num_iterations ):
        self.__count__ = num_iterations
        self.__count__, self._cImage, self._cImage_fft_mod = tf.while_loop( 
            self.__continue__, 
            self.__HIOKernel__, 
            [ self.__count__, self._cImage, self._cImage_fft_mod ], 
            parallel_iterations=1
        )
        return

###########################################################################################
#   Regular HIO
###########################################################################################
    def SolventFlipping( self, num_iterations ):
        self.__count__ = num_iterations
        self.__count__, self._cImage, self._cImage_fft_mod = tf.while_loop( 
            self.__continue__, 
            self.__SFKernel__, 
            [ self.__count__, self._cImage, self._cImage_fft_mod ], 
            parallel_iterations=1
        )
        return


###########################################################################################
#  Template for  
###########################################################################################


###########################################################################################
#   Shrinkwrap (Gaussian blurring followed by thresholding)
###########################################################################################
    def Shrinkwrap( self, sigma, thresh ):
        dist = tf.contrib.distributions.MultivariateNormalDiag( [ 0., 0., 0. ], [ sigma, sigma, sigma ] )
        kernelFFT = tf.fft3d( 
            tf.cast( 
                tf.reshape( dist.prob( self._domain ), self._probSize ), 
                tf.complex64 
            ) 
        )
        blurred = tf.abs( 
            tf.ifft3d( 
                tf.multiply( 
                    tf.fft3d( tf.cast( tf.abs( self._cImage ), tf.complex64 ) ), 
                    kernelFFT
                )
            )
        )
        blurred = tf.concat( ( blurred[ (self._probSize[0]//2):, :, : ], blurred[ :(self._probSize[0]//2), :, : ] ), axis=0 )
        blurred = tf.concat( ( blurred[ :, (self._probSize[1]//2):, : ], blurred[ :, :(self._probSize[1]//2), : ] ), axis=1 )
        blurred = tf.concat( ( blurred[ :, :, (self._probSize[2]//2): ], blurred[ :, :, :(self._probSize[2]//2) ] ), axis=2 )
        self._support = tf.cast( 
            blurred > thresh * tf.reduce_max( blurred ), 
            tf.complex64
        )
        self._support_comp = tf.cast( 
            blurred <= thresh * tf.reduce_max( blurred ), 
            tf.complex64
        )
        return

###########################################################################################
#   Method to clean up and extract values back to numpy.
###########################################################################################

    def Compute( self ):

        self.__sess__.run( 
            [ self._cImage, self._support ],
            options=self.__options__, 
            run_metadata=self.__run_metadata__
        )

        print( 'Extracting numpy arrays...' )
        start = time.time()
        self.finalImage = self._cImage.eval( session=self.__sess__ )
        self.finalSupport = np.absolute( self._support.eval( session=self.__sess__ ) )
        print( 'Time taken = %f sec'%( time.time() - start ) )
        
        print( 'Writing trace data...' )
        fetched_timeline = timeline.Timeline( self.__run_metadata__.step_stats )
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open( 'output.trace', 'w' ) as fid:
            fid.write( chrome_trace )

        print( 'Closing session...' )
        self.__sess__.close()

        print( 'Done.' )
   
###########################################################################################
#   Callable kernels representing single iterations of various phase retrieval algorithms. 
#   These are used in Tensorflow while loops for multiple iterations.
###########################################################################################

###########################################################################################
#   Conditional to continue iterating
###########################################################################################
    def __continue__( self, mycount, myimage, myimage_fft_mod ):
        return mycount > 0

###########################################################################################
#   Kernel representing a single iteration of error reduction
###########################################################################################
    def __ERKernel__( self, mycount, myimage, myimage_fft_mod ):
        myimage = tf.ifft3d( 
            tf.multiply( 
                self._modulus, 
                tf.exp( 
                    tf.complex( tf.zeros( myimage.shape ), tf.angle( tf.fft3d( myimage ) ) ) 
                ) 
            ) 
        )
        myimage = tf.multiply( myimage, self._support )
        myimage_fft_mod = tf.cast( tf.abs( tf.fft3d( myimage ) ), dtype=tf.complex64 )
        mycount -= 1
        return mycount, myimage, myimage_fft_mod

###########################################################################################
#   Kernel representing a single iteration of hybrid input/output
###########################################################################################
    def __HIOKernel__( self, mycount, myimage, myimage_fft_mod ):
        origImage = tf.identity( myimage )
        myimage = tf.ifft3d( 
            tf.multiply( 
                self._modulus, 
                tf.exp( 
                    tf.complex( tf.zeros( myimage.shape ), tf.angle( tf.fft3d( myimage ) ) ) 
                ) 
            ) 
        )
        myimage = tf.multiply( self._support, myimage ) + tf.multiply( self._support_comp, origImage - self._beta*myimage )
        mycount -= 1
        return mycount, myimage, myimage_fft_mod

###########################################################################################
#   Kernel representing a single iteration of solvent flipping.
###########################################################################################
    def __SFKernel__( self, mycount, myimage, myimage_fft_mod ):
        myimage = 2. *( self._support*myimage ) - myimage
        myimage = tf.ifft3d( 
            tf.multiply( 
                self._modulus, 
                tf.exp( 
                    tf.complex( tf.zeros( myimage.shape ), tf.angle( tf.fft3d( myimage ) ) ) 
                ) 
            ) 
        )
        myimage = 2. *( self._support*myimage ) - myimage
        mycount -= 1
        return mycount, myimage, myimage_fft_mod



