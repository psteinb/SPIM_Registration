package mpicbg.spim.postprocessing.deconvolution2.mapg;

import ij.CompositeImage;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.util.ArrayList;
import java.util.Date;

import mpicbg.imglib.cursor.Cursor;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.image.display.imagej.ImageJFunctions;
import mpicbg.imglib.type.numeric.real.FloatType;
import mpicbg.spim.postprocessing.deconvolution2.AdjustInput;
import mpicbg.spim.postprocessing.deconvolution2.LRFFT;
import mpicbg.spim.postprocessing.deconvolution2.LRFFT.PSFTYPE;
import mpicbg.spim.postprocessing.deconvolution2.LRInput;

/**
 * Implementation of Verveer et al. MAPG algorithm based on the Python code. 
 * 
 * @author Stephan Preibisch (stephan.preibisch@gmx.de)
 *
 */
public class MAPG
{
	final int numViews, numDimensions;
	final LRInput views;
	final ArrayList<LRFFT> inputdata;
	final ArrayList< Image< FloatType > > images, kernels;
	final float avg;
	
	final Image< FloatType > estimation;
	final FourierConvolutionMAPG< FloatType, FloatType > blur;
	Image< FloatType > data, est_blurred, d, dblurred, tmp;
	
	double beta = 0.0, rtr = 0.0;
	boolean restart = true;
	
	public static boolean debug = true;
	public static int debugInterval = 1;
	final static float minValue = 0.0001f;

    ImageStack stack;
    CompositeImage ci;

	// current iteration
    int i = 0;
    
	public MAPG( final LRInput views, final int numIterations, final String name )
	{
		this.numViews = views.getNumViews();
		this.numDimensions = views.getViews().get( 0 ).getImage().getNumDimensions();
		this.inputdata = views.getViews();
		this.views = views;
		
		this.images = new ArrayList<Image<FloatType>>();
		this.kernels = new ArrayList<Image<FloatType>>();
		for ( int v = 0; v < numViews; ++v )
		{
			this.images.add( inputdata.get( v ).getImage() );
			this.kernels.add( inputdata.get( v ).getKernel1() );
		}
		
		this.estimation = inputdata.get( 0 ).getImage().createNewImage( "estimation (deconvolved image)" );
		this.tmp = inputdata.get( 0 ).getImage().createNewImage();
		
		
		views.init( PSFTYPE.MAPG );
		
		this.avg = (float)AdjustInput.normAllImages( inputdata );
		
		//
		// the real data image psi is initialized with the average 
		//	
		for ( final FloatType f : estimation )
			f.set( avg );
		
		this.blur = new FourierConvolutionMAPG<FloatType, FloatType>( images, kernels );
		
		// run the deconvolution
		while ( i < numIterations )
		{
			runIteration();
			
			if ( debug && (i-1) % debugInterval == 0 )
			{
				Image< FloatType > psi = getPsi();
				
				psi.getDisplay().setMinMax( 0, 1 );
				final ImagePlus tmp = ImageJFunctions.copyToImagePlus( psi );
				
				if ( this.stack == null )
				{
					this.stack = tmp.getImageStack();
					for ( int i = 0; i < psi.getDimension( 2 ); ++i )
						this.stack.setSliceLabel( "Iteration 1", i + 1 );
					
					tmp.setTitle( "debug view" );
					this.ci = new CompositeImage( tmp, CompositeImage.COMPOSITE );
					this.ci.setDimensions( 1, psi.getDimension( 2 ), 1 );
					this.ci.show();
				}
				else if ( stack.getSize() == psi.getDimension( 2 ) )
				{
					IJ.log( "Stack size = " + this.stack.getSize() );
					final ImageStack t = tmp.getImageStack();
					for ( int i = 0; i < psi.getDimension( 2 ); ++i )
						this.stack.addSlice( "Iteration 2", t.getProcessor( i + 1 ) );
					IJ.log( "Stack size = " + this.stack.getSize() );
					this.ci.hide();
					IJ.log( "Stack size = " + this.stack.getSize() );
					
					this.ci = new CompositeImage( new ImagePlus( "debug view", this.stack ), CompositeImage.COMPOSITE );
					this.ci.setDimensions( 1, psi.getDimension( 2 ), 2 );
					this.ci.show();
				}
				else
				{
					final ImageStack t = tmp.getImageStack();
					for ( int i = 0; i < psi.getDimension( 2 ); ++i )
						this.stack.addSlice( "Iteration " + i, t.getProcessor( i + 1 ) );

					this.ci.setStack( this.stack, 1, psi.getDimension( 2 ), stack.getSize() / psi.getDimension( 2 ) );	
				}
				/*
				Image<FloatType> psiCopy = psi.clone();
				//ViewDataBeads.normalizeImage( psiCopy );
				psiCopy.setName( "Iteration " + i + " l=" + lambda );
				psiCopy.getDisplay().setMinMax( 0, 1 );
				ImageJFunctions.copyToImagePlus( psiCopy ).show();
				psiCopy.close();
				psiCopy = null;*/
			}
		}
		
		IJ.log( "DONE (" + new Date(System.currentTimeMillis()) + ")." );		
	}
	
	public void runIteration() 
	{
		if ( i == 0 )
			runFirstIteration();
		else
			iterate();
		
		++i;
	}

	final private void runFirstIteration()
	{
		// multiply image with kernel (called only once)
		data = blur.matrix_transpose_multiply();
		
		//ImageJFunctions.show( data ).setTitle( "data" );
		
		// blur estimation with the kernel (or something like that)
		est_blurred = blur.matrix_multiply_twice( estimation );
		
		//ImageJFunctions.show( est_blurred ).setTitle( "est_blurred" );
				
		final Cursor< FloatType > cEstimation = estimation.createCursor();
		final Cursor< FloatType > cEst_blurred = est_blurred.createCursor();
		
		while ( cEstimation.hasNext() )
		{
			cEstimation.fwd();
			cEst_blurred.fwd();
			
			cEstimation.getType().set( (float)Math.sqrt( cEst_blurred.getType().get() ) );
		}
		
		cEst_blurred.close();
		cEstimation.close();

		//ImageJFunctions.show( estimation ).setTitle( "estimation" );
	}

	final private void iterate()
	{
		final Image< FloatType > lastEstimate = estimation.clone();
		
		//
		// compute temporary image
		// tmp = 4.0 * self.estimation * (self.data - self.est_blurred)
		//
		final Cursor< FloatType > cEstimation = estimation.createCursor();
		final Cursor< FloatType > cEst_blurred = est_blurred.createCursor();
		final Cursor< FloatType > cData = data.createCursor();
		final Cursor< FloatType > cTmp = tmp.createCursor();

		while ( cEstimation.hasNext() )
		{
			final float est = cEstimation.next().get();
			final float est_Blurred = cEst_blurred.next().get();
			final float data = cData.next().get();
			
			cTmp.next().set( 4.0f * est * ( data - est_Blurred ) );
		}

		cEst_blurred.close();
		cEstimation.close();
		cTmp.close();
		cData.close();
		
		//ImageJFunctions.show( tmp );
		
		final double rr = squaredSum( tmp );
		
		if ( restart )
		{
            this.restart = false;
            this.beta = 0.0;
            this.d = tmp.clone();
		}
        else
        {
            this.beta = rr / this.rtr;
            addMul( tmp, (float)beta, d );
            //self.d = tmp + self.beta * self.d
        }
		
        // remember rr
        this.rtr = rr;

        /*
        c1 = -inproduct(tmp, self.d)
	    tmp = self.d * self.d
	    d2Ax2 = inproduct(self.est_blurred, tmp)
	    d2b = inproduct(tmp, self.data)
	    dblurred = self.blur.matrix_multiply_twice(tmp)
	    c4 = inproduct(tmp, dblurred)
	    tmp = self.estimation * self.d
	    c3 = 4.0 * inproduct(tmp, dblurred)
	    tmp = self.blur.matrix_multiply_twice(tmp)     
	    xdAdx = product3_sum(tmp, self.estimation, self.d)
	    c2 = 4.0 * xdAdx + 2.0 * (d2Ax2 - d2b)
	        
	    ca = 0.75 * c3 / c4
	    cb = 0.50 * c2 / c4
	    cc = 0.25 * c1 / c4
		*/
        
        final double c1 = -inproduct( tmp, d );
		squareD( d, tmp );
		final double d2Ax2 = inproduct( est_blurred, tmp );
		final double d2b = inproduct( tmp, data );
	    dblurred = blur.matrix_multiply_twice( tmp );
	    final double c4 = inproduct( tmp, dblurred );
	    mul( estimation, d, tmp );	    	    	    
	    final double c3 = 4.0 * inproduct( tmp, dblurred );
	    tmp = blur.matrix_multiply_twice( tmp );     
	    final double xdAdx = product3_sum( tmp, estimation, d );
	    final double c2 = 4.0 * xdAdx + 2.0 * ( d2Ax2 - d2b );

	    final double ca = 0.75 * c3 / c4;
	    final double cb = 0.50 * c2 / c4;
	    final double cc = 0.25 * c1 / c4;

	    // self.alpha = polynomial3_first_root(ca, cb, cc)
	    
	    final double alpha = polynomial3_first_root( ca, cb, cc );
	    
	    //self.estimation += self.alpha * self.d
	    //self.est_blurred += self.alpha * self.alpha * dblurred
	    //self.est_blurred += 2.0 * self.alpha * tmp
	    
	    mulAdd( alpha, d, estimation );
	    mulAdd( alpha * alpha, dblurred, est_blurred );
	    mulAdd( 2.0 * alpha, tmp, est_blurred );
	    
	    //ImageJFunctions.show( estimation );
	    
		//SimpleMultiThreading.threadHaltUnClean();
	    
	    final Cursor< FloatType > last = lastEstimate.createCursor();
	    final Cursor< FloatType > next = estimation.createCursor();
	    
		double sumChange = 0;
		double maxChange = -1;

	    while ( last.hasNext() )
	    {	    
		    final float change = Math.abs( last.next().get() - next.next().get() );				
			sumChange += change;
			maxChange = Math.max( maxChange, change );
	    }
	    
	    IJ.log("iteration: " + i + " --- sum change: " + sumChange + " --- max change per pixel: " + maxChange );
	}

	final private static void mulAdd( final double alpha, final Image< FloatType > d, final Image< FloatType > target )
	{
		final Cursor< FloatType > cD = d.createCursor();
		final Cursor< FloatType > cT = target.createCursor();

		while ( cD.hasNext() )
		{
			cT.fwd();
			final double v = cD.next().get();
			
			
			cT.getType().set( cT.getType().get() + (float)( alpha * v ) );
		}

		cD.close();
		cT.close();
	}

	final private static double polynomial3_first_root( final double a, final double b, final double c )
	{
		final double Q = (a * a - 3.0 * b) / 9.0;
		final double R = (2.0 * a*a*a - 9.0 * a * b + 27.0 * c) / 54.0;

		final double Q3 = Q*Q*Q;
	    final double R2 = R * R;

	    int n;
	    double r1, r2 = 0, r3 = 0;
	    
	    if ( R2 < Q3 )
	    {
	    	n = 3;
	    	final double rQ = Math.sqrt(Q);
	    	final double t = Math.acos(R / Math.sqrt(Q3));
	    	r1 = -2.0 * rQ * Math.cos(t / 3.0) - a / 3.0;
	    	r2 = -2.0 * rQ * Math.cos((t + 2.0 * Math.PI) / 3.0) - a / 3.0;
	        r3 = -2.0 * rQ * Math.cos((t - 2.0 * Math.PI) / 3.0) - a / 3.0;
	    }
	    else
	    {
	    	n = 1;
	    	double A = Math.pow( (Math.abs(R) + Math.sqrt(R2 - Q3)), (1 / 3.0) );
	        if ( R >= 0.0 )
	            A = -A;
	        
	        final double B; 
	        if ( A != 0.0 )
	            B = Q / A;
	        else
	            B = 0.0;
	        
	        r1 = (A + B) - a / 3.0;
	        if ( R2 == Q3 && A != 0.0 )
	        {
	            r2 = -0.5 * (A + B) - a / 3.0;
	            n = 2;
	        }
	    }
	    
	    if ( n == 1 )
	        return r1;
	    
	    final double f1 = 0.25 * r1*r1*r1*r1 + a * r1*r1*r1 / 3.0 + 0.5 * b * r1*r1 + c * r1;
	    final double f2 = 0.25 * r2*r2*r2*r2 + a * r2*r2*r2 / 3.0 + 0.5 * b * r2*r2 + c * r2;

	    if ( n == 2 )
	    {
	    	if ( f1 < f2 )
	            return r1;
	        else
	            return r2;
	    }
	    
	    final double f3 = 0.25 * r3*r3*r3*r3 + a * r3*r3*r3 / 3.0 + 0.5 * b * r3*r3 + c * r3;

	    if ( f1 < f2 )
	    {
	        if ( f1 < f3 )
	            return r1;
	        else
	            return r3;
	    }
	    else
	    {
	        if ( f2 < f3 )
	            return r2;
	        else
	            return r3;
	    }
	}
	
	final private static void mul( final Image< FloatType > a, final Image< FloatType > b, final Image< FloatType > target )
	{
		final Cursor< FloatType > c1 = a.createCursor();
		final Cursor< FloatType > c2 = b.createCursor();
		final Cursor< FloatType > cT = target.createCursor();

		while ( c1.hasNext() )
		{
			final float v1 = c1.next().get();
			final float v2 = c2.next().get();
			
			cT.next().set( v1 * v2 );
		}

		c1.close();
		c2.close();
		cT.close();
	}

	final private static void squareD( final Image< FloatType > d, final Image< FloatType > tmp )
	{
		final Cursor< FloatType > cTmp = tmp.createCursor();
		final Cursor< FloatType > cD = d.createCursor();

		while ( cTmp.hasNext() )
		{
			final float value = cD.next().get();
			
			cTmp.next().set( value * value );
		}

		cD.close();
		cTmp.close();
		
	}

	final private static void addMul( final Image< FloatType > tmp, final float beta, final Image< FloatType > d  )
	{
		final Cursor< FloatType > cTmp = tmp.createCursor();
		final Cursor< FloatType > cD = d.createCursor();

		while ( cTmp.hasNext() )
		{
			final float tmpV = cTmp.next().get();
			final float dV = cD.next().get();
			
			
			cD.getType().set( tmpV + beta * dV );
		}

		cD.close();
		cTmp.close();
	}

	final private static double squaredSum( final Image< FloatType > input )
	{
		double ss = 0;
		
		for ( final FloatType f : input )
		{
			final double s = f.get();
			ss += s * s;
		}
		
		return ss;
	}
	
	final private static double inproduct( final Image< FloatType > input1, final Image< FloatType > input2 )
	{
		final Cursor< FloatType > c1 = input1.createCursor();
		final Cursor< FloatType > c2 = input2.createCursor();

		double sum = 0;
		
		while ( c1.hasNext() )
			sum += c1.next().get() * c2.next().get();
		
		c1.close();
		c2.close();
		
	    return sum;
	}

	final private static double product3_sum( final Image< FloatType > input1, final Image< FloatType > input2, final Image< FloatType > input3 )
	{
		final Cursor< FloatType > c1 = input1.createCursor();
		final Cursor< FloatType > c2 = input2.createCursor();
		final Cursor< FloatType > c3 = input3.createCursor();

		double sum = 0;
		
		while ( c1.hasNext() )
			sum += c1.next().get() * c2.next().get() * c3.next().get();
		
		c1.close();
		c2.close();
		c3.close();
		
	    return sum;
	}

	public Image<FloatType> getPsi()
	{
		final Image< FloatType > img = estimation.clone();
		
		for ( final FloatType t : img )
			t.set( t.get() * t.get() );
		
		return img; 
	}
}
