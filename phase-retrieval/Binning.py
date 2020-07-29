import numpy as np

def GenerateBinRanges( pixelsAlongAxis, dx ):
    lst1 = range( pixelsAlongAxis/2, pixelsAlongAxis, dx )
    lst2 = [ pixelsAlongAxis-(n+dx) for n in lst1 ][::-1]
    lstStart = lst2 + lst1
    if lstStart[0] < 0:
        lstStart = lstStart[1:][:-1]
    lstEnd = [ n+dx-1 for n in lstStart ]
    return lstStart, lstEnd

def fastInPlaneBinning( img, dx, dy, averaging=False ):
    if any( [ n%2!=0 for n in img.shape ] ):
        print( 'fastInPlaneBinning: Odd array dimension(s).' )
        return img, [], []

    buf = ( ( img.shape[0]//2 )%dx, ( img.shape[1]//2 )%dy )
    imgCropped = img[ buf[0]:(img.shape[0]-buf[0]), buf[1]:(img.shape[1]-buf[1]) ]
    binL, binR = getBinningOperators( imgCropped.shape, dx, dy )
    if averaging:
        binL /= binL[0,:].sum()
        binR /= binR[0,:].sum()
    return binL.dot( imgCropped ).dot( binR.T ), binL, binR

def getBinningOperators( shp, dx, dy ):
    return bin1D( shp[0], dx ), bin1D( shp[1], dy )

def bin1D( dimen, dx ):       # averages by default
    arr0 = np.zeros( ( 1, dimen ) )
    arr0[ 0, :dx ] = 1.
    arr = np.array( [] ).reshape( 0, arr0.shape[-1] )
    for i in list( range( dimen//dx ) ):
        arr = np.concatenate( ( arr, np.roll( arr0, dx*i, axis=1 ) ), axis=0 )
    return arr
