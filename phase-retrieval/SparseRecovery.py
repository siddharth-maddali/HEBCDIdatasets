#######################################################################
#
# Module for sparse recovery of Bragg coherent diffraction signal 
# Reference: https://arxiv.org/pdf/1712.01108.pdf
# 
# Siddharth Maddali
# Argonne National Laboratory
# December 2017
#
#######################################################################

import numpy as np
import customTransform as trans
import scipy.fftpack as spfft
from sklearn.linear_model import Lasso

from tqdm import tqdm               # progress bar

class SparseRecoverySolver:

    def __init__( self, 
            data_stack, pixel_binning_factor, inplane_offset, pixel_size_mm, white_field=None, 
            cs_alpha=2.e-4, cs_tol=5.e-6, cs_max_iter=24000
        ):

        self._pbf        = pixel_binning_factor
        self._pitch      = pixel_size_mm
        
        self._inplane    = np.round( inplane_offset * pixel_binning_factor / pixel_size_mm ) # this is in physical coords
        self._inplane    = np.array( [ [ 0, -1 ], [ 1, 0 ] ] ).dot( self._inplane.T ).T      # conversion to image coords
        self._inplane    = self._inplane.astype( int )
        self._block_size = self._inplane.shape[0]

        if white_field != None:
            self._data = np.zeros( data_stack.shape )
            deadPixels = np.where( white_field == 0. )
            white_field[ deadPixels ] = 1.
            for n in list( range( self._data.shape[-1] ) ):
                temp = data_stack[:,:,n] / white_field
                temp[ deadPixels ] = 0.     # TODO: more sophisticated treatment of dead pixels
                self._data[:,:,n] = temp
        else:
            self._data = data_stack


        self._desiredSize = self._data.shape[0] * self._pbf
        self._Msensing = np.matrix( spfft.dct( np.identity( self._desiredSize ), norm='ortho' ).T )
        self._sensing_dict = {}

        self._csSolver = Lasso( 
                fit_intercept=False, 
                warm_start=True, 
                alpha=cs_alpha, 
                tol=cs_tol, 
                max_iter=cs_max_iter 
                )


    def fullRecover( self, threshold_to_zero=True ):
        numRecovImg = self._data.shape[-1] // self._block_size
        self._result = np.zeros( (
                self._desiredSize, 
                self._desiredSize, 
                numRecovImg
                ) )
        for n in tqdm( list( range( numRecovImg ) ) ):
            self._result[:,:,n] = self.recoverThis( n )

        if threshold_to_zero == True:
            self._result *= ( self._result > 0. ).astype( float )
        return

    def recoverThis( self, n ):
        A, b = self.__senseBlockInPlane__( 
                data_block=self._data[:,:,n*self._block_size:(n+1)*self._block_size],
                threshold=0. 
                )
        self._csSolver.fit( A, b )
        while self._csSolver.n_iter_ == self._csSolver.max_iter: # didn't converge; more iterations!
            self._csSolver.fit( A, b )

        return trans.inverseDCT2( 
                self._csSolver.coef_.reshape( self._desiredSize, self._desiredSize ) 
                ).astype( float )



    def __senseBlockInPlane__( self, data_block, threshold=1.e-7 ):
        

        A = np.zeros( ( np.prod( data_block.shape ) , self._desiredSize**2 ) )
        b = np.zeros( ( np.prod( data_block.shape ), 1 ) )
        row_count = 0
    
        [ j_grid, i_grid ] = np.meshgrid( 
                list( range( self._desiredSize ) ), 
                list( range( self._desiredSize ) ) 
                )

        rollingImage = np.zeros( ( self._desiredSize, self._desiredSize ) )
        rollingImage[ :self._pbf, :self._pbf ] = 1.
    
        Atemp = np.zeros( ( self._pbf**2, self._desiredSize**2 ) )
        Mavg = ( 1. / self._pbf )**2 * np.ones( ( 1, self._pbf**2 ) )
            
        for n in list( range( len( self._inplane ) ) ):
            thisDisp = self._inplane[n]
            i_grid_shifted, j_grid_shifted = i_grid + thisDisp[0], j_grid + thisDisp[1]
            for i in list( range( data_block.shape[0] ) ):
                for j in list( range( data_block.shape[1] ) ):
                    if data_block[:,:,n][i,j]/data_block[:,:,n].max() < threshold:    
                        # if below (normalized) noise level, ignore.
                        continue
                    rolledImage = np.roll( np.roll( rollingImage, self._pbf*i, axis=0 ), self._pbf*j, axis=1 )
                    i_patch, j_patch = \
                            i_grid_shifted[ np.where( rolledImage>0.5 ) ], \
                            j_grid_shifted[ np.where( rolledImage>0.5 ) ]
                    thisIsKosher = np.logical_and( 
                            np.logical_and( i_patch >=0, i_patch < self._desiredSize ), 
                            np.logical_and( j_patch >=0, j_patch < self._desiredSize )
                    ).all()
                    if thisIsKosher:
                        for k in list( range( len( i_patch ) ) ):
                            thisPixel = ( i_patch[k], j_patch[k] )
                            if thisPixel not in self._sensing_dict.keys():
                                self._sensing_dict[ thisPixel ] = np.array( \
                                        self._Msensing[:,thisPixel[0]] * self._Msensing[:,thisPixel[1]].T 
                                ).reshape( 1, -1 )
                            Atemp[k,:] = self._sensing_dict[ thisPixel ] 
                        A[ row_count, : ] = Mavg.dot( Atemp )
                        b[ row_count, : ] = data_block[:,:,n][i,j]
                        row_count += 1
    
        A = A[ :row_count, : ]  # final trim
        b = b[ :row_count, : ]
        return A, b
                
