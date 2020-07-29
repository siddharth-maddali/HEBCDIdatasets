import numpy as np
import scipy.sparse as sps
from customTransform import *
from tqdm import tqdm

###########################################################################################

def generateBinRanges( pixelsAlongAxis, dx ):
    lst1 = list(range( pixelsAlongAxis//2, pixelsAlongAxis, dx))
    lst2 = [ pixelsAlongAxis-(n+dx) for n in lst1 ][::-1]
    lstStart = lst2 + lst1
    if lstStart[0] < 0:
        lstStart = lstStart[1:][:-1]
    lstEnd = [ n+dx-1 for n in lstStart ]
    return lstStart, lstEnd

###########################################################################################

def inPlaneBinning( img, dx, dy, averaging=True ):
    if all( [ n%2==0 for n in img.shape ] ) == False:
        print( 'InPlaneBinning: Odd array dimension(s).' )
        return img
    listStart_x, listEnd_x = generateBinRanges( img.shape[0], dx )
    listStart_y, listEnd_y = generateBinRanges( img.shape[1], dy )
    binnedImg = np.zeros( ( len( listStart_x ), len( listStart_y ), img.shape[-1] ) )
    for i in range( len( listStart_x ) ):
        for j in range( len( listStart_y ) ):
            binnedImg[ i, j, : ] = img[ 
                listStart_x[i]:listEnd_x[i]+1, 
                listStart_y[j]:listEnd_y[j]+1, 
                :
            ].sum( axis=0 ).sum( axis=0 ).reshape( 1, 1, img.shape[-1] )
    if averaging:
        binnedImg /= ( dx*dy )
    return binnedImg

###########################################################################################

def inPlaneExpansion( img, binL, binR, prefactor=1. ):
#   N = len( np.where( binL[0,:] > 1.e-6 )[0] )
    N = ( binL[0,:] > 1.e-6 ).astype( int ).sum()
    imgExpanded = np.zeros( ( img.shape[0]*N, img.shape[1]*N, img.shape[2] ) )
    for n in list( range( imgExpanded.shape[-1] ) ):
        imgExpanded[:,:,n] = binR.T.dot( img[:,:,n] ).dot( binL )
    return prefactor * imgExpanded

###########################################################################################

def sliceMiniArray( origArray, inPlaneOrigin, inPlaneSpan ):
        return origArray[ 
            ( inPlaneOrigin[0]-inPlaneSpan[0] ):( inPlaneOrigin[0]+inPlaneSpan[0] ), 
            ( inPlaneOrigin[1]-inPlaneSpan[1] ):( inPlaneOrigin[1]+inPlaneSpan[1] ), 
            :
        ]
    
###########################################################################################

def generateRotationAboutPrincipalAxis( th, ax ):
    M = np.array( [ 
        [ 1., 0., 0. ], 
        [ 0., np.cos( th ), -np.sin( th ) ], 
        [ 0., np.sin( th ),  np.cos( th ) ]
    ])
    return np.roll( 
        np.roll( M, shift=ord(ax)-ord('x'), axis=0 ),
        shift=ord(ax)-ord('x'), 
        axis=1
    )

###########################################################################################

def basisSkew( basis ):
    return np.sqrt( 
        1. - ( np.absolute( basis[:,0].dot( np.cross( basis[:,1], basis[:,2] ) ) ) /\
        np.prod( np.sqrt( ( basis**2 ).sum( axis=0 ) ) ) )**2
    )
    
###########################################################################################

def createSuperGrid( binning_factor, detector_size ):
    sup = np.linspace( 0., 1., binning_factor * detector_size )
    mn = sup[:binning_factor ].mean()
    kernel = sup[:binning_factor] - mn

    mx = sup[-binning_factor:].mean()
    sub = mn + ( mx-mn )* np.linspace( 0., 1., detector_size )
    return sub, sup, kernel

###########################################################################################

def generateMatrixLaplacian( matrix_size=10 ):
    n = matrix_size**2
    L = sps.lil_matrix( ( n, n ) )
    y, x = np.meshgrid( list(range( matrix_size)), list(range( matrix_size)) )
    x = x.ravel()
    y = y.ravel()
    for i in range( matrix_size ):
        for j in range( matrix_size ):
            w = np.where( 
                np.logical_xor( 
                    np.logical_and( np.absolute( x-i ) == 1, np.absolute( y-j ) == 0 ), 
                    np.logical_and( np.absolute( x-i ) == 0, np.absolute( y-j ) == 1 ) 
                )
            )[0]
            for k in w:
                L[ i*matrix_size+j, x[k]*matrix_size+y[k] ] = -1.
                L[ i*matrix_size+j, i*matrix_size+j ] = len( w )
    return L

###########################################################################################

# Complete sensing routine for data block. Chooses default in-plane image as in the 
# center of the stack. Scans through image stack, and chooses sensing algorithm depending 
# on whether current image is central image or not. The latter requires sampling basis 
# of reciprocal space.

def senseBlock( data, binningFactor, recipBasis, sensingMatrix, sensingDict={}, n_central=None ):
    desiredImageSize = data.shape[0] * binningFactor
#   Some sanity checks and adjustments...
    if n_central==None:
        if len( data.shape==2 ):
            data = data.reshape( data.shape[0], data.shape[1], 1 )
            n_central = 0
        elif len( data.shape==3 ):
            n_central = data.shape[-1]//2; # the middle layer in stack
            if recipBasis==None and data.shape[-1] != 1:
                print( 'Error: Need reciprocal space sampling basis.\n' )
                return [], []
        else:
            print( 'Error: Incorrect input data shape.\n' )
            return [], []

    A = np.array( [] ).reshape( np.prod( data.shape ), desiredImageSize**2 )
    b = np.array( [] ).reshape( np.prod( data.shape ), 1 )

    subGrid, supGrid, kern = createSuperGrid( binningFactor, data.shape[0] )
    iD = np.matrix( inverseDCT2( np.identity( desiredImageSize ) ) )

    for n in tqdm( list( range( data.shape[-1] ) ) ):
        if n != n_central:          # this is a  neighbor to the central image
#            print( 'Non-central...\n' )
            dq_inplane = ( n - n_central )* recipBasis
            An, bn = getConstraintsFromNeighbor( dq_inplane, data[:,:,n], subGrid, supGrid, kern, sensingMatrix=iD, sensingDict=sensingDict )
        else:                       # this is the central image
#            print( 'Central...\n' )
            An, bn = getConstraintsInPlane( data[:,:,n], binningFactor, sensingMatrix=iD, sensingDict=sensingDict )

        A = np.concatenate( ( A, An ), axis=0 )
        b = np.concatenate( ( b, bn ), axis=0 )

    return A, b

###########################################################################################

# In-plane variation of the senseBlock function. This assumes the experiment was performed with 
# successive detector images in the data block being in the same plane in reciprocal space, but 
# displaced from the center of the original image. This is more similar to the kernel averaging 
# filter seems to be ideally suited to sparse recovery, as has been demonstrated in 1D. 

def senseBlockInPlane( data, binningFactor, inplaneDisp, sensingMatrix, sensingDict={}, threshold=1.e-7 ):
    
    desiredImageSize = data.shape[0] * binningFactor
    A = np.zeros( ( np.prod( data.shape ) , desiredImageSize**2 ) )
    b = np.zeros( ( np.prod( data.shape ), 1 ) )
    row_count = 0

    [ j_grid, i_grid ] = np.meshgrid( list( range( desiredImageSize ) ), list( range( desiredImageSize ) ) )

    rollingImage = np.zeros( ( desiredImageSize, desiredImageSize ) )
    rollingImage[ :binningFactor, :binningFactor ] = 1.

    Atemp = np.zeros( ( binningFactor**2, desiredImageSize**2 ) )
    Mavg = ( 1. / binningFactor )**2 * np.ones( ( 1, binningFactor**2 ) )
        
    for n in list( range( len( inplaneDisp ) ) ):
        thisDisp = inplaneDisp[n]
        i_grid_shifted, j_grid_shifted = i_grid + thisDisp[0], j_grid + thisDisp[1]
        for i in list( range( data.shape[0] ) ):
            for j in list( range( data.shape[1] ) ):
                if data[:,:,n][i,j]/data[:,:,n].max() < threshold:    # if below (normalized) noise level, ignore.
                    continue
                rolledImage = np.roll( np.roll( rollingImage, binningFactor*i, axis=0 ), binningFactor*j, axis=1 )
                i_patch, j_patch = i_grid_shifted[ np.where( rolledImage>0.5 ) ], j_grid_shifted[ np.where( rolledImage>0.5 ) ]
                thisIsKosher = np.logical_and( 
                        np.logical_and( i_patch >=0, i_patch < desiredImageSize ), 
                        np.logical_and( j_patch >=0, j_patch < desiredImageSize )
                ).all()
                if thisIsKosher:
                    for k in list( range( len( i_patch ) ) ):
                        thisPixel = ( i_patch[k], j_patch[k] )
                        if thisPixel not in sensingDict.keys():
                            sensingDict[ thisPixel ] = np.array( \
                                    sensingMatrix[:,thisPixel[0]] * sensingMatrix[:,thisPixel[1]].T 
                            ).reshape( 1, -1 )
                        Atemp[k,:] = sensingDict[ thisPixel ] 
                    A[ row_count, : ] = Mavg.dot( Atemp )
                    b[ row_count, : ] = data[:,:,n][i,j]
                    row_count += 1

    A = A[ :row_count, : ]  # final trim
    b = b[ :row_count, : ]
    return A, b

###########################################################################################

def getConstraintsFromNeighbor( dq_inPlane, neighboringImage, subGrid, supGrid, kernelGrid, sensingMatrix, sensingDict={} ):
    
# IMPORTANT NOTE: dq_inPlane should be in in-plane, final-image barycentric coordinates, not
# input image coordinates.
    flp = len( supGrid ) - 1
    
    
    An = np.zeros( ( neighboringImage.size, sensingMatrix.shape[0]**2  ) )
    bn = neighboringImage.reshape( -1, 1 )
    row_count = 0
    to_exclude_from_bn = []
    
    sensingAtom = np.zeros( ( 4, len( supGrid )**2 ) )
    sensingAtomAvg = np.zeros( ( 1, len( supGrid )**2 ) )
    
    db = np.diff( np.array( supGrid ) ).mean()
    sub_x = subGrid + dq_inPlane[0]
    sub_y = subGrid + dq_inPlane[1] # matrix-to-Cartesian coordinates conversion
        
    for i in list( range( neighboringImage.shape[0] ) ):
        for j in list( range( neighboringImage.shape[1] ) ):
            [ bx, by ] = np.meshgrid( sub_x[i]+kernelGrid, sub_y[j]+kernelGrid )
            by = by[::-1,:]
            bxn = ( bx // db ).astype( int ); byn = ( by // db ).astype( int )
            if np.any( bxn<0 ) or np.any( byn<0 ) or np.any( bxn>=len( supGrid )-1 ) or np.any( byn>=len( supGrid )-1 ):
                to_exclude_from_bn.extend( [ neighboringImage.shape[0]*i + j] )
                continue; # this is outside the scope of the current image.
            dbxn = (bx%db)/db; dbyn = (by%db)/db
            dbx = dbxn.mean()
            dby = dbyn.mean()
            baryCoeff = np.array( [ 1.-dby, dby ] ).reshape( -1, 1 ).dot( 
                np.array( [ 1.-dbx, dbx ] ).reshape( 1, -1 ) 
            ).reshape( 1, -1 )
            bxn = np.unique( bxn ); byn = np.unique( byn )
            sensingAtomAvg.fill( 0 )
            for p in list( range( 2 ) ):
                for q in list( range( 2 ) ):
                    bxt = bxn + p; byt = byn + q
                    byt = flp - byt
                    thisGuy = [ 
                        ( byt[0], bxt[0] ), 
                        ( byt[0], bxt[1] ), 
                        ( byt[1], bxt[0] ), 
                        ( byt[1], bxt[1] )
                    ]
                    for r in list( range( len( thisGuy ) ) ):
                        if thisGuy[r] not in sensingDict.keys():
                            sensingDict[ thisGuy[r] ] = np.array( sensingMatrix[:,thisGuy[r][0]] * sensingMatrix[:,thisGuy[r][1]].T ).reshape( 1, -1 )
                        sensingAtom[r,:] = sensingDict[ thisGuy[r] ]
                    sensingAtomAvg += baryCoeff.dot( sensingAtom ).reshape( 1, -1 )
            sensingAtomAvg *= 0.25
            An[ row_count, : ] = sensingAtomAvg
            row_count += 1
    An = An[ :row_count, : ]
    bn = bn[ [ i for i in list( range( bn.size ) ) if i not in to_exclude_from_bn ], : ]
    return An, bn

###########################################################################################

def getConstraintsInPlane( img, binningFactor, sensingMatrix, sensingDict={} ):
    desiredImageSize = img.shape[0] * binningFactor
    An = np.zeros( ( img.size, desiredImageSize**2 ) )
    bn = img.reshape( -1, 1 )
    Atemp = np.zeros( ( binningFactor**2, desiredImageSize**2 ) )
    Mavg = ( 1. / binningFactor )**2 * np.ones( ( 1, binningFactor**2 ) )

    rollingImage = np.zeros( ( desiredImageSize, desiredImageSize ) )
    rollingImage[ :binningFactor, :binningFactor ] = 1.

    for i in list( range( img.shape[0] ) ):
        for j in list( range( img.shape[1] ) ):
            rolledImage = np.roll( np.roll( rollingImage, binningFactor*i, axis=0 ), binningFactor*j, axis=1 )
            here = np.where( rolledImage > 0.5 )
            for k in list( range( len( here[0] ) ) ):
                thisGuy = ( here[0][k], here[1][k] )
                if thisGuy not in sensingDict.keys():
                    sensingDict[ thisGuy ] = np.array( sensingMatrix[ :, thisGuy[0] ] * sensingMatrix[ :, thisGuy[1] ].T ).reshape( 1, -1 )
                Atemp[ k, : ] = sensingDict[ thisGuy ]
            An[ i*img.shape[0] + j, : ] = Mavg.dot( Atemp ).reshape( 1, -1 )

    return An, bn

###########################################################################################

