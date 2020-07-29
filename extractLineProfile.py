import numpy as np

def extractLineProfile( img, pts, num_max=1000 ):
    """
    extractLineProfile:
    Extracts a line profile of the scalar field defined 
    in the 2D image 'img', between the 2D points pts[:,0] 
    and pts[:,1] within this range. The number of sample 
    points is 'num_max', defaults to 100.
    """
    xS, yS = tuple( 
        [
            pts[n,0] + ( pts[n,1]-pts[n],0 ) * np.linspace( 0., 1., num_max ).reshape( -1, 1 )
            for n in [ 0, 1 ]
        ]
    )
    ptsSample = np.unique( np.concatenate( ( xS, yS ), axis=1 ).round(), axis=0 )
    return ptsSample.astype( int )


if __name__=="__main__":
    import extractLineProfile as lp

    dat = Namespace( **sio.loadmat( '../data/estimatedStrains_possibleError.mat' ) ) 
    rho = np.absolute( dat.rho_111_A )
    snp = dat.strain_111_A
    snp[ np.where( rho > 0. ) ] = snp[ np.where( rho > 0. ) ] + 2.17e-4 
    img = snp[89:149,89:149,30]
    pts = np.array( [ [ 15., 15. ], [ 45., 45. ] ] )
    ptsSample = lp.extractLineProfile( img, pts )
