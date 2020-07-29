##############################################################
#
#	GPUModule_Base:
#	    Contains a mixin for defining Tensorflow variables
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
import tensorflow as tf

class Mixin:

    def defineBaseVariables( self, varDict, gpu ):
        # Tensorflow variables specified here.
        with tf.device( '/gpu:%d'%gpu ):
            self._modulus = tf.constant( varDict[ 'modulus' ], dtype=tf.complex64, name='mod_measured' )
            self._support = tf.Variable( varDict[ 'support' ], dtype=tf.complex64, name='sup' )
            self._support_comp = tf.Variable( 1. - varDict[ 'support' ], dtype=tf.complex64, name='Support_comp' )
            self._cImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64, name='Image' )
            self._buffImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64, name='buffImage' )
            self._beta = tf.constant( varDict[ 'beta' ], dtype=tf.complex64, name='beta' )
            self._probSize = self._cImage.shape
            self._thresh = tf.placeholder( dtype=tf.float32, name='thresh' )
            self._sigma = tf.placeholder( dtype=tf.float32, name='sigma' )
            self._neg = tf.constant( -0.5, dtype=tf.float32 )
            self._intermedFFT = tf.Variable( tf.zeros( self._cImage.shape, dtype=tf.complex64 ), name='intermedFFT' )
            self._intermedInt = tf.Variable( tf.zeros( self._cImage.shape, dtype=tf.complex64 ), name='intermedInt' )

            with tf.name_scope( 'Support' ):
                self._supproject = tf.assign( 
                    self._cImage, 
                    self._cImage * self._support, 
                    name='supProject' 
                )
            # These are defined only if high-energy phasing is required.
            if 'bin_left' in varDict.keys():
                bL = varDict[ 'bin_left' ]
                sh = bL.shape
                self._binL = tf.constant( 
                    bL.reshape( sh[0], sh[1], 1 ).repeat( varDict[ 'modulus' ].shape[-1], axis=2 ), 
                    dtype=tf.complex64, 
                    name='binL'
                )
                self._binR = tf.constant( 
                    bL.T.reshape( sh[1], sh[0], 1 ).repeat( varDict[ 'modulus' ].shape[-1], axis=2 ), 
                    dtype=tf.complex64, 
                    name='binR'
                )
                self._scale = tf.constant( varDict[ 'scale' ], dtype=tf.complex64, name='scale' )
                self._binned = tf.Variable( tf.zeros( self._modulus.shape, dtype=tf.complex64 ), name='binned' )
                self._expanded = tf.Variable( tf.zeros( self._support.shape, dtype=tf.complex64 ), name='expanded' )
                self._scaled = tf.Variable( tf.zeros( self._modulus.shape, dtype=tf.complex64 ), name='scaled' )

                with tf.name_scope( 'highEnergy' ):
                    self._binThis = tf.assign( 
                        self._binned, 
                        tf.transpose( 
                            tf.matmul( 
                                tf.matmul( 
                                    tf.transpose( self._binL, [ 2, 0, 1 ] ), 
                                    tf.transpose( 
                                        tf.cast( tf.square( tf.abs( self._intermedFFT ) ), dtype=tf.complex64 ), 
                                        [ 2, 0, 1 ]
                                    )
                                ), 
                                tf.transpose( self._binR, [ 2, 0, 1 ] )
                            ), [ 1, 2, 0 ] 
                        ), 
                        name='Binning'
                    )
                    self._scaleThis = tf.assign( 
                        self._scaled,
                        tf.divide( self._modulus, tf.sqrt( self._binned ) ), 
                        name='Scaling'
                    )
                    self._expandThis = tf.assign( 
                        self._expanded, 
                        tf.transpose( 
                            tf.matmul( 
                                tf.matmul( 
                                    tf.transpose( self._binR, [ 2, 0, 1 ] ), 
                                    tf.transpose( self._scaled, [ 2, 0, 1 ] )
                                ), 
                                tf.transpose( self._binL, [ 2, 0, 1 ] )
                            ), [ 1, 2, 0 ] 
                        ), 
                        name='Expansion'
                    )
                    self._HEImgUpdate = tf.assign( 
                        self._cImage, 
                        tf.multiply( 
                            self._support, 
                            tf.ifft3d( self._scale * tf.multiply( self._expanded, self._intermedFFT ) ) 
                        ), 
                        name='HEImgUpdate'
                    )
                    self._HEImgCorrect = tf.assign( 
                        self._cImage, 
                        self._cImage + tf.multiply( self._support_comp, self._buffImage - self._beta*self._cImage ), 
                        name='HEImgCorrect' 
                    )
        
            else: # regular phasing 
                with tf.name_scope( 'ER' ):
                    self._modproject = tf.assign( 
                        self._cImage, 
                        tf.ifft3d( tf.divide( self._modulus, tf.sqrt( self._intermedInt ) ) * tf.fft3d( self._cImage ) ), 
                        name='modProject' 
                    )
        return


