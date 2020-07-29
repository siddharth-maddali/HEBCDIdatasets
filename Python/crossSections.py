import TilePlot as tp

root = '/home/smaddali/ANL/BeamRuns/Feb2018/reconstructions'
datasets = [ 
    'stdSample', 
    '111-A', 
    '111-B'#, 
#    '200-D'
]

imgs = []
for snp in datasets:
    dat= sio.loadmat( '%s/%s_solution.mat'%( root, snp ) )
    reconst = np.absolute( dat[ 'rho' ] )
    x, y, z = reconst.shape
    nmin = min( reconst.shape ) # usually the 3rd axis
#    img1 = reconst[ x//2, ((y-z)//2):((y+z)//2), : ]
#    img2 = reconst[ ((x-z)//2):((x+z)//2), y//2, : ]
    img1 = dat[ 'data' ][:,:,z//2]
    img1[ np.where( img1==0. ) ] = np.unique( np.sort( img1.ravel() ) )[1]
    img1 = np.log10( img1 )
    img2 = np.log10( dat[ 'intensity' ][:,:,z//2] )
    img3 = reconst[ ((x-z)//2):((x+z)//2), ((y-z)//2):((y+z)//2), z//2 ]
    imgs.extend( [ img1, img2, img3 ] )

fig, ims, axes = tp.TilePlot( 
    imgs, 
    ( 3, 3 ), 
    color_scales=False
)

for n in list( range( len( axes ) ) ): 
    axes[n].set_xticks( [] )
    axes[n].set_yticks( [] )
    ims[n].set_cmap( 'inferno' )

plt.show()
    
