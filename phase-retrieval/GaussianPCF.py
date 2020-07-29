# Routines for computing the Gaussian partial coherence function 
# given a Bragg CDI diffraction data set and an estimate of the 
# coherent intensity pattern (usually obtained from phase retrieval).
# Takes into account the basis of sampling in reciprocal space. 
# 
# Siddharth Maddali
# Argonne National Laboratory
# 2018

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

def getGaussianPCF( 
        noisy_data, coherent_estimate, domain, 
        learning_rate=1.e-1, recip_basis=np.eye( 3 ), 
        min_iterations=200, max_iterations=10000, 
        iterations_per_checkpoint=50, tol=1.e-4 
    ):

    l1p, l2p, l3p = np.random.rand(), np.random.rand(), np.random.rand()
    psip, thetap, phip = 2.*np.pi*np.random.rand(), np.pi*np.random.rand(), 2.*np.pi*np.random.rand()

    with tf.device( '/gpu:0' ):

        roll_ = [ n // 2 for n in coherent_estimate.shape ]

        # defining constants
        with tf.name_scope( 'Constants' ):
            noisyData = tf.constant( noisy_data, dtype=tf.float32, name='noisyData' )
            coherentEstimate = tf.constant( coherent_estimate, dtype=tf.float32, name='coherentEstimate' )
            q = tf.constant( domain, dtype=tf.float32, name='domainPoints' )
    
            v0 = tf.constant( np.array( [ 1., 0., 0. ] ).reshape( -1, 1 ), dtype=tf.float32 )
            v1 = tf.constant( np.array( [ 0., 1., 0. ] ).reshape( -1, 1 ), dtype=tf.float32 )
            v2 = tf.constant( np.array( [ 0., 0., 1. ] ).reshape( -1, 1 ), dtype=tf.float32 )
            nskew0 = tf.constant( np.array( [ [ 0., 0., 0. ], [ 0., 0., -1. ], [ 0., 1., 0. ] ] ), dtype=tf.float32 )
            nskew1 = tf.constant( np.array( [ [ 0., 0., 1. ], [ 0., 0., 0. ], [ -1., 0., 0. ] ] ), dtype=tf.float32 )
            nskew2 = tf.constant( np.array( [ [ 0., -1., 0. ], [ 1., 0., 0. ], [ 0., 0., 0. ] ] ), dtype=tf.float32 )
            one = tf.constant( 1., dtype=tf.float32 )
            neg = tf.constant( -0.5, dtype=tf.float32 )
            twopi = tf.constant( ( 2 * np.pi )**( 3./2. ), dtype=tf.float32 )
            I = tf.eye( 3 )

        # defining the 6 optimization parameters
        with tf.name_scope( 'Parameters' ):
            l1 = tf.Variable( l1p, dtype=tf.float32, name='Lambda1' )                   #
            l2 = tf.Variable( l2p, dtype=tf.float32, name='Lambda2' )                   # Sqrt of eigenvalues of covariance matrix
            l3 = tf.Variable( l3p, dtype=tf.float32, name='Lambda3' )                   #
            psi = tf.Variable( psip, dtype=tf.float32, name='Psi' )                     # Rotation angle of eigenbasis
            theta = tf.Variable( thetap, dtype=tf.float32, name='Theta' )               # Polar angle of rotation axis
            phi = tf.Variable( phip, dtype=tf.float32, name='Phi' )                     # Azimuth angle of rotation axis

        # everything else
        with tf.name_scope( 'Auxiliary' ):
            mD = tf.diag( [ l1, l2, l3 ] )
            n0 = tf.sin( theta ) * tf.cos( phi )
            n1 = tf.sin( theta ) * tf.sin( phi )
            n2 = tf.cos( theta )
            n = n0*v0 + n1*v1 + n2*v2
            nskew = n0*nskew0 + n1*nskew1 + n2*nskew2
            R = tf.cos( psi )*I + tf.sin( psi )*nskew + ( one - tf.cos( psi ) )*tf.matmul( n, tf.transpose( n ) )
            C = tf.matmul( R, tf.matmul( tf.matmul( mD, mD ), tf.transpose( R ) ) )


        with tf.name_scope( 'Blurring' ):
            blurKernel = tf.reshape( 
                tf.exp( neg * tf.reduce_sum( q * tf.matmul( C, q ), axis=0 ) ), 
                shape=coherentEstimate.shape
            ) * (l1 * l2 * l3 ) / twopi
            
            tf_intens_f = tf.fft3d( tf.cast( coherentEstimate, dtype=tf.complex64 ) )
            tf_blur_f = tf.fft3d( tf.cast( blurKernel, dtype=tf.complex64 ) )
            tf_prod_f = tf_intens_f * tf_blur_f
            tf_prod_rolledIJK = tf.ifft3d( tf_prod_f )
            tf_prod_rolledJK = tf.concat( ( tf_prod_rolledIJK[ roll_[0]:, :, : ], tf_prod_rolledIJK[ :roll_[0], :, : ] ), axis=0 )
            tf_prod_rolledK = tf.concat( ( tf_prod_rolledJK[ :, roll_[1]:, : ], tf_prod_rolledJK[ :, :roll_[1], : ] ), axis=1 )
            imgBlurred = tf.abs( tf.concat( ( tf_prod_rolledK[ :, :, roll_[2]: ], tf_prod_rolledK[ :, :, :roll_[2] ] ), axis=2 ) )

#            imgBlurred = tf.reshape( 
#                tf.nn.convolution( 
#                    tf.reshape( coherentEstimate, [ coherentEstimate.shape[0], coherentEstimate.shape[1], coherentEstimate.shape[2], 1, 1 ] ), 
#                    tf.reshape( blurKernel, [ blurKernel.shape[0], blurKernel.shape[1], blurKernel.shape[2], 1, 1 ] ), 
#                    padding='SAME'
#                ), 
#                shape=coherentEstimate.shape
#            )

        with tf.name_scope( 'Optimizer' ):
            poissonNLL = tf.reduce_mean( imgBlurred - noisyData * tf.log( imgBlurred ) )
            var_list = [ l1, l2, l3, psi, theta, phi ]
#            poissonOptimizer = tf.train.MomentumOptimizer( learning_rate=1.e-5, momentum=0.99, use_nesterov=True, name='PoissonOptimize' )
            poissonOptimizer = tf.train.AdagradOptimizer( learning_rate=learning_rate, name='PoissonOptimize' )
            trainPoisson = poissonOptimizer.minimize( poissonNLL, var_list=var_list )
            currentGradients = [ n[0] for n in poissonOptimizer.compute_gradients( poissonNLL, var_list=var_list ) ]

    session = tf.Session()
    session.run( tf.global_variables_initializer() )


    progress = []
    
    this_grad = np.linalg.norm( np.array( session.run( currentGradients ) ) )
    normalizr = session.run( tf.reduce_sum( blurKernel ) )
    checkpoint = session.run( var_list )
    obj = session.run( poissonNLL )
    checkpoint.extend( [ obj, this_grad, normalizr ] )
    progress.append( checkpoint )

    n_iter = 1
    start = time.time()

#    for n_iter in tqdm( list( range( max_iter ) ) ):
    while n_iter < min_iterations or ( this_grad > tol and n_iter < max_iterations ):
        session.run( trainPoisson )
        this_grad = np.linalg.norm( np.array( session.run( currentGradients ) ) )
        if n_iter%iterations_per_checkpoint == 0:  # store progress every 'iterations_per_checkpoint' iterations.
            normalizr = session.run( tf.reduce_sum( blurKernel ) )
            checkpoint = session.run( var_list )
            obj = session.run( poissonNLL )
            checkpoint.extend( [ obj, this_grad, normalizr ] )
            progress.append( checkpoint )
        n_iter += 1
    
    normalizr = session.run( tf.reduce_sum( blurKernel ) )
    checkpoint = session.run( var_list )
    obj = session.run( poissonNLL )
    checkpoint.extend( [ obj, this_grad, normalizr ] )
    progress.append( checkpoint )
    stop = time.time()

    if n_iter>=max_iterations-1:
        print( 'Warning: Max. number of iterations reached (%f).'%max_iterations )
    else:
        print( 'Converged in %d iterations.'%n_iter )
    
    print( 'Time taken = %f sec'%( stop - start ) )
    
    blur_final, imgB_final, C_final = session.run( [ blurKernel, imgBlurred, C ] )
    return progress, blur_final, imgB_final, recip_basis.T @ C_final @ recip_basis




