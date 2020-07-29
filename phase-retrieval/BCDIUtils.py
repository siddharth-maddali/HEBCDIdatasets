# BCDIUtils.py -- general purpose tools for pre-processing stacks of BCDI data

import numpy as np

def centerDataOnBraggPeak( data, estimated_inplane_max, clip_radius=0, block_size=1 ):

    subArray = data[ :, :, list( range( 0, data.shape[-1], block_size ) ) ]

    if clip_radius > 0:
        subArray = subArray[ 
                ( estimated_inplane_max[0]-clip_radius ):( estimated_inplane_max[0]+clip_radius ), 
                ( estimated_inplane_max[1]-clip_radius ):( estimated_inplane_max[1]+clip_radius ), 
                :
                ]
        dataTemp = data[ 
                ( estimated_inplane_max[0]-clip_radius ):( estimated_inplane_max[0]+clip_radius ), 
                ( estimated_inplane_max[1]-clip_radius ):( estimated_inplane_max[1]+clip_radius ), 
                :
                ]
    else:
        dataTemp = data

    maxLocation = [ n[0] for n in np.where( subArray == subArray.max() ) ]



    circshift = ( 
            subArray.shape[0]//2-maxLocation[0], 
            subArray.shape[1]//2-maxLocation[1], 
            ( subArray.shape[2]//2-maxLocation[2] ) * block_size
            )
    dataOut = np.roll( dataTemp, shift=circshift, axis=( 0, 1, 2 ) )
            
    return dataOut
