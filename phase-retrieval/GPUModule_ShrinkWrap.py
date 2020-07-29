##############################################################
#
#	GPUModule_ShrinkWrap:
#	    Contains a mixin for the shrinkwrap algorithm 
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
import tensorflow as tf
from numpy.fft import fftshift

class Mixin:

    def defineShrinkwrap( self, varDict, gpu ):

        # Array coordinates
        x, y, z = np.meshgrid( 
            list( range( varDict[ 'cImage' ].shape[0] ) ),
            list( range( varDict[ 'cImage' ].shape[1] ) ),
            list( range( varDict[ 'cImage' ].shape[2] ) )
        )
        y = y.max() - y

        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()


        with tf.device( '/gpu:%d'%gpu ):
            # these are Tensorflow variables
            self._x = tf.constant( fftshift( x ), dtype=tf.float32, name='x' )
            self._y = tf.constant( fftshift( y ), dtype=tf.float32, name='y' )
            self._z = tf.constant( fftshift( z ), dtype=tf.float32, name='z' )
            self._blurred = tf.Variable( np.zeros( varDict[ 'support' ].shape ), dtype=tf.complex64, name='blurred' )
            self._dist = tf.Variable( tf.zeros( self._x.shape, dtype=tf.float32 ), name='dist' )

            # These are shrinkwrap-specific symbolic ops
            with tf.name_scope( 'Shrinkwrap' ):
                self._getNewDist = tf.assign( 
                    self._dist, 
                    tf.exp( 
                        self._neg * ( 
                            self._x*self._x + self._y*self._y + self._z*self._z 
                        ) / ( self._sigma * self._sigma )
                    ), 
                    name='getNewDist'
                )
#                self._copyDistToRollBuffer = tf.assign( self._rollBuffer, tf.cast( self._dist, dtype=tf.complex64 ), name='CopyDistToRollBuffer' )
#                self._retrieveDistFromRollBuffer = tf.assign( self._blurred, self._rollBuffer, name='retrieveDistFromRollBuffer' )
#                self._copyImageToRollBuffer = tf.assign( self._rollBuffer, tf.cast( tf.abs( self._cImage ), dtype=tf.complex64 ), name='copyImgToRollBuffer' )
#                self._convolveRollBufferWithBlur = tf.assign( self._rollBuffer, self._rollBuffer*self._blurred, name='convolve' )
#                self._retrieveBlurred = tf.assign( self._blurred, self._rollBuffer, name='retrieveBlurred' )
                self._blurShape = tf.assign( 
                    self._blurred, 
                    tf.ifft3d( 
                        tf.fft3d( tf.cast( self._dist, dtype=tf.complex64 ) ) *\
                        tf.fft3d( tf.cast( tf.abs( self._cImage ), dtype=tf.complex64 ) )
                    ), 
                    name='blurShape' 
                )
                self._updateSupport = tf.assign( 
                    self._support, 
                    tf.cast( tf.abs( self._blurred ) > self._thresh * tf.reduce_max( tf.abs( self._blurred ) ), tf.complex64 ), 
                    name='updateSup' 
                )
                self._updateSupComp = tf.assign( 
                    self._support_comp, 
                    tf.cast( tf.abs( self._blurred ) <= self._thresh * tf.reduce_max( tf.abs( self._blurred ) ), tf.complex64 ), 
                    name='updateSupComp' 
                )

###########################################################################################
#   Shrinkwrap (Gaussian blurring followed by thresholding)
###########################################################################################
    def Shrinkwrap( self, sigma, thresh ):
        self.__sess__.run( self._getNewDist, feed_dict={ self._sigma:sigma } )
        self.__sess__.run( self._blurShape )
        self.__sess__.run( self._updateSupport, feed_dict={ self._thresh:thresh } )
        self.__sess__.run( self._updateSupComp, feed_dict={ self._thresh:thresh } )
        self.__sess__.run( self._supproject )
        return
