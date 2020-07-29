import autograd.numpy as np
from autograd import elementwise_grad as egrad

def radfun( 
        x, y, 
        a=1., mu=0.5, sigma=0.25, ph0=1., scl=1.,
        Q=np.array( [ 1., 0. ] ), 
        veclin=np.array( [ 1., 0. ] )
    ):
    '''
    a:      scale of tanh rising edge
    mu:     location of tanh rising edge
    sigma:  width of tanh rising edge
    ph0:    constant radfun
    veclin: 2D direction of projection
    dph:    slope of radfun ramp
    '''
    r = np.sqrt( x**2 + y**2 )
    fr = a * 0.5 * ( 1. + np.tanh( ( r - mu ) / sigma ) )
    xscl = x / r
    yscl = y / r
    dirscl = Q[0]*xscl + Q[1]*yscl
    framp = ph0 + scl * ( veclin[0]*x + veclin[1]*y )
    return fr * dirscl**2 * framp


######################################################################################

gbrad = 0.85
sigma=0.15
scl=0.

veclin = -1. + 2. * np.random.rand( 2 )
veclin = veclin / np.linalg.norm( veclin )
#veclin = np.array( [ 0., 0. ] )


Q = -1. + 2. * np.random.rand( 2 )
Q = Q / np.linalg.norm( Q )

lineslope = 2.

######################################################################################

x, y = np.meshgrid( 
    np.linspace( -1., 1., 500 ), 
    np.linspace( -1., 1., 500 )
)
y = np.flipud( y )
mask = ( np.sqrt( x**2 + y**2 ) < gbrad ).astype( float ) 

# gradients calculated here
dp_x = egrad( radfun, 0 )#( x, y, mu=gbrad, sigma=sigma, veclin=veclin )
dp_y = egrad( radfun, 1 )#( x, y, mu=gbrad, sigma=sigma, veclin=veclin )

linerange = np.linspace( -1., 1., 500 )
linecoords = Q.reshape( -1, 1 ) @ linerange.reshape( 1, -1 )
linevalues = radfun( linecoords[0,:], linecoords[1,:], mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin )
linegrad = Q[0]*dp_x( linecoords[0,:], linecoords[1,:], mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin ) + Q[1]*dp_y( linecoords[0,:], linecoords[1,:], mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin )

######################################################################################

plt.close()
plt.close()
plt.close()

plt.figure()
plt.imshow( 
    mask * radfun( x, y, mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin ), 
    extent=[ x.min(), x.max(), y.min(), y.max() ] 
)
plt.colorbar()
plt.set_cmap( 'viridis' )
plt.title( 'Phase' )
plt.quiver( [ 0. ], [ 0. ], [ Q[0] ], [ Q[1] ], linewidth=10., scale=10., color='r' )
plt.show()


plt.figure()
plt.imshow( 
    mask * ( 
        Q[0]*dp_x( x, y, mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin ) + 
        Q[1]*dp_y( x, y, mu=gbrad, sigma=sigma, scl=scl, Q=Q, veclin=veclin ) ),
    extent=[ x.min(), x.max(), y.min(), y.max() ] 
)
plt.colorbar()
plt.set_cmap( 'bwr' )
plt.title( 'Strain' )
plt.show()

plt.figure()
plt.plot( linerange, lineslope*linerange + linevalues, label='Line profile with physical ramp' )
plt.plot( linerange, linevalues, label='Line profile from phase retrieval (measured)' )
plt.plot( linerange, linegrad, label='Line gradient of measured line profile' )
plt.grid()
plt.legend( loc='best' )
plt.show()
