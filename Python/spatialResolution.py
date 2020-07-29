import numpy as np

from scipy.optimize import curve_fit
from scipy.special import erf

#=================================================================================================

def extractLineProfile( img, line=[ 1., 0. ] ):
    '''
    extractLineProfile( img, line ):
        Returns a line profile of a given image 'img' along the line=[ m, c ]
        where m is the slope and c is the y-intercept.

    Params: 
        img: 2D Numpy array
        line: list containing slope and y-intercept

    Returns:
        intens: 1D Numpy array of intensities along the line
    '''
    x, y = np.meshgrid( np.arange( img.shape[0] ), np.arange( img.shape[1] ) )
    y = np.flipud( y )
    here = np.where( np.absolute( y - ( line[0]*x + line[1] ) ) < 1. )
    return img[ here ], here

#=================================================================================================

def myEdge( x, mu, sig, amp=1. ):
    '''
    myEdge: 
        Models a rising edge by the following function:
        f(x) = \frac{A}{2} \left[ 1 + \text{erf}\left(\frac{x-\mu}{\sigma}\right) \right ]
    '''
    return( amp / 2. ) * ( 1. + erf( ( x - mu ) / sig ) )

#=================================================================================================

if __name__=='__main__':
    import spatialResolution as sr
    from argparse import Namespace
    import scipy.io as sio

    filename = '/home/smaddali/ANL/BeamRuns/Feb2018/reconstructions/stdSample_solution.mat'
    calfilename = '/home/smaddali/ANL/Manuscripts/HEBCDI/data/bases_FINAL.mat'
    dat = Namespace( **sio.loadmat( filename ) )
    cal = Namespace( **sio.loadmat( calfilename ) )
    print( 'Array shape = ', dat.rho.shape )
    intens, here = sr.extractLineProfile( 
        np.absolute( dat.rho[:,:,33] ), 
        line=[ 1., 0. ]
    )

    # fitting edge
    data = intens[50:66] 
    fspace_steps = cal.real_stdSample[:,0].reshape(-1,1)@here[0].reshape(1,-1) + cal.real_stdSample[:,1].reshape(-1,1)@here[1].reshape(1,-1)
    fspace_steps = fspace_steps - fspace_steps[:,0].reshape( -1, 1 ).repeat( fspace_steps.shape[1], axis=1 )
    my_x = np.sqrt( ( fspace_steps**2 ).sum( axis=0 ) )[50:66]
    seg = np.array(
        [ 
            [ here[1][50], here[1][66] ], 
            [ here[0][50], here[0][66] ]
        ]
    )
    popt, pcov = curve_fit( 
        myEdge, 
        my_x, data, 
        p0=[ 1750., 100., 0.09 ]
    )
    
    # plotting
    plt.figure( 1 )
    plt.clf()
    plt.imshow( np.absolute( dat.rho[:,:,33] ) )
    plt.xlim( [ 44, 84 ] )
    plt.ylim( [ 44, 84 ] )
    plt.colorbar()
    plt.plot( seg[0], seg[1], 'c' )
    plt.set_cmap( 'inferno' )
    plt.xticks( [] )
    plt.yticks( [] )
    
    plt.figure( 2 )
    plt.clf()
    plt.plot( my_x, data, '-o', label='Line profile' )
    plt.plot( my_x, myEdge( my_x, popt[0], popt[1], popt[2] ), label='Fitted edge' )
    plt.grid()
    plt.legend( loc='best' )
    plt.xlabel( 'nm', fontsize=18, fontweight='bold' )
    plt.ylabel( '$\\left|\\rho\\right|$', fontsize=18, fontweight='bold' )
