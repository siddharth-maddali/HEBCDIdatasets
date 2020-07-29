##############################################################
#
#	GPUModule_ErrorReduction:
#	    Contains a mixin for error reduction methods 
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import tensorflow as tf
from tqdm import tqdm

class Mixin:

    def defineER( self, gpu ): # symbolic ops for error reduction

        with tf.device( '/gpu:%d'%gpu ):
            with tf.name_scope( 'DiffractionPattern' ):
                self._getIntermediateFFT = tf.assign( 
                    self._intermedFFT, 
                    tf.fft3d( self._cImage ), 
                    name='intermedFFT'
                )
                self._getIntermediateInt = tf.assign( 
                    self._intermedInt, 
                    tf.square( tf.cast( tf.abs( self._intermedFFT ), dtype=tf.complex64 ) ), 
                    name='intermedInt'
                )
                

###########################################################################################
#   Performs <num_iterations> iterations of error reduction
###########################################################################################
    def ER( self, num_iterations, label=' ER: ' ):
        for n in tqdm( list( range( num_iterations ) ), desc=label ):
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._getIntermediateInt )
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._supproject )
        return

###########################################################################################
#   Performs <num_iterations> iterations of high-energy error reduction
###########################################################################################
    def HEER( self, num_iterations ):
        for n in list( range( num_iterations ) ):
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._binThis )
            self.__sess__.run( self._scaleThis )
            self.__sess__.run( self._expandThis )
            self.__sess__.run( self._HEImgUpdate )
            self.__sess__.run( self._supproject )
        return


