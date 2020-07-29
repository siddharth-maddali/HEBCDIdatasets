##############################################################
#
#	GPUModule_HybridInputOutput:
#	    Contains a mixin for hybrid input-output methods 
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

    def defineHIO( self, varDict, gpu ):

        with tf.device( '/gpu:%d'%gpu ):
            if 'bin_left' not in varDict:
                with tf.name_scope( 'HIO' ):
                    self._disrupt = tf.assign( 
                        self._cImage, 
                        ( self._support * self._cImage ) + self._support_comp * ( self._buffImage - self._beta*self._cImage ), 
                        name='disrupt'
                    )
            
            with tf.name_scope( 'bufferImage' ): # used both in HIO and HEHIO
                self._dumpimage = tf.assign( self._buffImage, self._cImage, name='dumpImage' )

        return

###########################################################################################
#   Performs <num_iterations> iterations of of hybrid input/output
###########################################################################################
    def HIO( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc='HIO' )
        else:
            allIterations = list( range( num_iterations ) )
        for n in allIterations:
            self.__sess__.run( self._dumpimage )
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._getIntermediateInt )
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._disrupt )
        return

###########################################################################################
#   Performs <num_iterations> iterations of of high-energy hybrid input/output
###########################################################################################
    def HEHIO( self, num_iterations ):
        for n in list( range( num_iterations ) ):
            self.__sess__.run( self._dumpimage )
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._binThis )
            self.__sess__.run( self._scaleThis )
            self.__sess__.run( self._expandThis )
            self.__sess__.run( self._HEImgUpdate )
            self.__sess__.run( self._HEImgCorrect )
        return

