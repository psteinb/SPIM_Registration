/*
 * #%L
 * ImgLib: a general-purpose, multidimensional image processing library.
 * %%
 * Copyright (C) 2009 - 2013 Stephan Preibisch, Tobias Pietzsch, Barry DeZonia,
 * Stephan Saalfeld, Albert Cardona, Curtis Rueden, Christian Dietz, Jean-Yves
 * Tinevez, Johannes Schindelin, Lee Kamentsky, Larry Lindsey, Grant Harris,
 * Mark Hiner, Aivar Grislis, Martin Horn, Nick Perry, Michael Zinsmaier,
 * Steffen Jaensch, Jan Funke, Mark Longair, and Dimiter Prodanov.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 2 of the 
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public 
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-2.0.html>.
 * #L%
 */

package mpicbg.spim.postprocessing.deconvolution2.mapg;

import java.util.ArrayList;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import mpicbg.imglib.algorithm.Benchmark;
import mpicbg.imglib.algorithm.MultiThreaded;
import mpicbg.imglib.algorithm.fft.FourierTransform;
import mpicbg.imglib.algorithm.fft.FourierTransform.PreProcessing;
import mpicbg.imglib.algorithm.fft.FourierTransform.Rearrangement;
import mpicbg.imglib.algorithm.fft.InverseFourierTransform;
import mpicbg.imglib.cursor.Cursor;
import mpicbg.imglib.cursor.LocalizableByDimCursor;
import mpicbg.imglib.cursor.LocalizableCursor;
import mpicbg.imglib.image.Image;
import mpicbg.imglib.image.ImageFactory;
import mpicbg.imglib.image.display.imagej.ImageJFunctions;
import mpicbg.imglib.multithreading.Chunk;
import mpicbg.imglib.multithreading.SimpleMultiThreading;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyFactory;
import mpicbg.imglib.outofbounds.OutOfBoundsStrategyMirrorFactory;
import mpicbg.imglib.type.numeric.RealType;
import mpicbg.imglib.type.numeric.complex.ComplexFloatType;
import mpicbg.imglib.type.numeric.real.FloatType;

/**
 * Specialized implementation of the Fourier Convolution based on the python code of Verveer et al., Nature Methods (2007)
 *
 * @author Stephan Preibisch
 */
public class FourierConvolutionMAPG<T extends RealType<T>, S extends RealType<S>> implements MultiThreaded, Benchmark
{
	final int numDimensions, numViews;
	
	final ArrayList<Image<T>> images;
	final ArrayList<Image<S>> kernels;
	final ArrayList<Image<T>> weights;
	
	Image< ComplexFloatType > total_otf = null;
	
	final ArrayList<Image<ComplexFloatType>> kernelFFTs, imgFFTs; 
	
	OutOfBoundsStrategyFactory<T> strategy = new OutOfBoundsStrategyMirrorFactory<T>();
	
	final int[] kernelDim;

	String errorMessage = "";
	int numThreads;
	long processingTime;

	public FourierConvolutionMAPG( final ArrayList<Image<T>> images, final ArrayList<Image<S>> kernels, final ArrayList<Image<T>> weights )
	{
		if ( images == null || kernels == null || images.size() == 0 || images.size() != kernels.size() )
			throw new RuntimeException( "Same number (>0) of images and kernels required." );

		this.numDimensions = images.get( 0 ).getNumDimensions();
		this.numViews = images.size();
		
		final int[] dimImg = images.get( 0 ).getDimensions();
		this.kernelDim = kernels.get( 0 ).getDimensions();
		
		for ( int v = 0; v < numViews; ++v )
			for ( int d = 0; d < numDimensions; ++d )
				if ( images.get( v ).getDimension( d ) != dimImg[ d ] )
					throw new RuntimeException( "Image dimensions do not match." );

		for ( int v = 0; v < numViews; ++v )
			for ( int d = 0; d < numDimensions; ++d )
				if ( kernels.get( v ).getDimension( d ) != kernelDim[ d ] )
					throw new RuntimeException( "Kernel dimensions do not match." );

		this.images = images;
		this.kernels = kernels;
		this.weights = weights;
		
		this.kernelFFTs = null;
		this.imgFFTs = null;
		
		setNumThreads();
	}
	
	public void setImageOutOfBoundsStrategy( final OutOfBoundsStrategyFactory<T> strategy ) { this.strategy = strategy; }
	public OutOfBoundsStrategyFactory<T> getImageOutOfBoundsStrategy() { return this.strategy; }

	public Image< T > matrix_transpose_multiply() 
	{		
		final long startTime = System.currentTimeMillis();
		
		final Image< T > result = images.get( 0 ).createNewImage();
		final Image< T > weights = result.createNewImage();
		
		//
		// compute fft of the input image
		//		
		for ( int v = 0; v < numViews; ++v )
		{
			//
			// compute fft of image
			//
			final FourierTransform<T, ComplexFloatType> fftImg = new FourierTransform<T, ComplexFloatType>( images.get( v ), new ComplexFloatType() );
			fftImg.setNumThreads( this.getNumThreads() );
			
			// we do not rearrange the fft quadrants
			fftImg.setRearrangement( Rearrangement.UNCHANGED );
						
			// how to extend the input image out of its boundaries for computing the FFT,
			// we simply mirror the content at the borders
			//fftImage.setPreProcessing( PreProcessing.EXTEND_MIRROR );
			fftImg.setPreProcessing( PreProcessing.USE_GIVEN_OUTOFBOUNDSSTRATEGY );
			fftImg.setCustomOutOfBoundsStrategy( strategy );
		
			// the image has to be extended by the size of the kernel-1
			// as the kernel is always odd, e.g. if kernel size is 3, we need to add
			// one pixel out of bounds in each dimension (3-1=2 pixel all together) so that the
			// convolution works
			final int[] imageExtension = kernelDim.clone();		
			for ( int d = 0; d < numDimensions; ++d )
				--imageExtension[ d ];		
			fftImg.setImageExtension( imageExtension );
			
			if ( !fftImg.checkInput() || !fftImg.process() )
			{
				errorMessage = "FFT of image failed: " + fftImg.getErrorMessage();
				return null;			
			}
			
			//
			// Rearrange FFT of kernel
			//

			// get the size of the kernel image that will be fourier transformed,
			// it has the same size as the image
			final int kernelTemplateDim[] = fftImg.getResult().getDimensions();
			kernelTemplateDim[ 0 ] = ( fftImg.getResult().getDimension( 0 ) - 1 ) * 2;
			
			// instaniate real valued kernel template
			// which is of the same container type as the image
			// so that the computation is easy
			final ImageFactory<S> kernelTemplateFactory = new ImageFactory<S>( kernels.get( v ).createType(), images.get( v ).getContainer().getFactory() );
			final Image<S> kernelTemplate = kernelTemplateFactory.createImage( kernelTemplateDim );
			
			// copy the kernel into the kernelTemplate,
			// the key here is that the center pixel of the kernel (e.g. 13,13,13)
			// is located at (0,0,0)
			final LocalizableCursor<S> kernelCursor = kernels.get( v ).createLocalizableCursor();
			final LocalizableByDimCursor<S> kernelTemplateCursor = kernelTemplate.createLocalizableByDimCursor();
			
			final int[] position = new int[ numDimensions ];
			while ( kernelCursor.hasNext() )
			{
				kernelCursor.next();
				kernelCursor.getPosition( position );
				
				for ( int d = 0; d < numDimensions; ++d )
					position[ d ] = ( position[ d ] - kernelDim[ d ]/2 + kernelTemplateDim[ d ] ) % kernelTemplateDim[ d ];
				
				kernelTemplateCursor.setPosition( position );
				kernelTemplateCursor.getType().set( kernelCursor.getType() );
			}
			
			// 
			// compute FFT of kernel
			//
			final FourierTransform<S, ComplexFloatType> fftKernel = new FourierTransform<S, ComplexFloatType>( kernelTemplate, new ComplexFloatType() );
			fftKernel.setNumThreads( this.getNumThreads() );
			
			fftKernel.setPreProcessing( PreProcessing.NONE );		
			fftKernel.setRearrangement( fftImg.getRearrangement() );
			
			if ( !fftKernel.checkInput() || !fftKernel.process() )
			{
				errorMessage = "FFT of kernel failed: " + fftKernel.getErrorMessage();
				return null;			
			}		
			kernelTemplate.close();
							
			if ( total_otf == null )
				total_otf = fftImg.getResult().createNewImage();
			
			final Image< ComplexFloatType > otf = fftKernel.getResult();
			final Image< ComplexFloatType > cotf = fftKernel.getResult().clone();

			//
			// compute complex conjugate, multiply with imgfft and add to total
			//
			long numPixels = cotf.getDimension( 0 );
			for ( int d = 1; d < numDimensions; ++d )
				numPixels *= cotf.getDimension( d );

			final Vector< Chunk > threadChunks = SimpleMultiThreading.divideIntoChunks( numPixels, getNumThreads() );
			
			final AtomicInteger ai = new AtomicInteger(0);					
	        final Thread[] threads = SimpleMultiThreading.newThreads( numThreads );
	        for ( int ithread = 0; ithread < threads.length; ++ithread )
	            threads[ithread] = new Thread(new Runnable()
	            {
	                public void run()
	                {
	                	// get chunk of pixels to process
	                	final Chunk myChunk = threadChunks.get( ai.getAndIncrement() );
	                	
	                	// overwrite fftImg.getResult
	    				conjugateMultiply( myChunk.getStartPosition(), myChunk.getLoopSize(), cotf, fftImg.getResult() );
	                }
	            });
	        
	        SimpleMultiThreading.startAndJoin( threads );
	        
	        //
			// Compute inverse Fourier Transform of the multiplication
			//		
			final InverseFourierTransform<T, ComplexFloatType> invFFT = new InverseFourierTransform<T, ComplexFloatType>( fftImg.getResult(), fftImg );
			invFFT.setInPlaceTransform( true );
			invFFT.setNumThreads( this.getNumThreads() );

			if ( !invFFT.checkInput() || !invFFT.process() )
			{
				errorMessage = "InverseFFT of image failed: " + invFFT.getErrorMessage();
				return null;			
			}
						
			//
			// sum up with weights
			//
			final Cursor< T > weightsCursor = weights.createCursor();
			final Cursor< T > resultCursor = result.createCursor();
			
			final Cursor< T > invFFTCursor = invFFT.getResult().createCursor();
			final Cursor< T > weightInCursor = this.weights.get( v ).createCursor();
	        
			while( resultCursor.hasNext() )
			{
				weightsCursor.fwd();
				resultCursor.fwd();
				
				final double w = weightInCursor.next().getRealDouble();
				final double i = invFFTCursor.next().getRealDouble();
				
				resultCursor.getType().setReal( resultCursor.getType().getRealDouble() + i * w );
				weightsCursor.getType().setReal( weightsCursor.getType().getRealDouble() + w );
			}

	        //
	        // compute otf * cotf
	        //
	        ai.set( 0 );
	        for ( int ithread = 0; ithread < threads.length; ++ithread )
	            threads[ithread] = new Thread(new Runnable()
	            {
	                public void run()
	                {
	                	// get chunk of pixels to process
	                	final Chunk myChunk = threadChunks.get( ai.getAndIncrement() );
	                	
	    				sumMultiply( myChunk.getStartPosition(), myChunk.getLoopSize(), otf, cotf, total_otf );
	                }
	            });
	        
	        SimpleMultiThreading.startAndJoin( threads );
		}
		
		//
		// normalize
		//
		final Cursor< T > weightsCursor = weights.createCursor();
		final Cursor< T > resultCursor = result.createCursor();

		while( resultCursor.hasNext() )
		{
			weightsCursor.fwd();
			resultCursor.fwd();
			
			resultCursor.getType().setReal( resultCursor.getType().getRealDouble() / weightsCursor.getType().getRealDouble() );
			
		}
		
		weights.close();
		
		ImageJFunctions.show( result ).setTitle( "total" );

		for ( final ComplexFloatType t : total_otf )
			t.mul( 1.0 / (double)numViews );

		processingTime = System.currentTimeMillis() - startTime;
		
        return result;
	}
	
	public Image< T > matrix_multiply_twice( final Image< T > input )
	{
		/*
		 * input = self.forward_fft(input)
         * input = self.total_otf * input
         * return self.backward_fft(input, self.n)
		 */

		//
		// compute fft of image
		//
		final FourierTransform<T, ComplexFloatType> fftImg = new FourierTransform<T, ComplexFloatType>( input, new ComplexFloatType() );
		fftImg.setNumThreads( this.getNumThreads() );
		
		// we do not rearrange the fft quadrants
		fftImg.setRearrangement( Rearrangement.UNCHANGED );
					
		// how to extend the input image out of its boundaries for computing the FFT,
		// we simply mirror the content at the borders
		//fftImage.setPreProcessing( PreProcessing.EXTEND_MIRROR );
		fftImg.setPreProcessing( PreProcessing.USE_GIVEN_OUTOFBOUNDSSTRATEGY );
		fftImg.setCustomOutOfBoundsStrategy( strategy );
	
		// the image has to be extended by the size of the kernel-1
		// as the kernel is always odd, e.g. if kernel size is 3, we need to add
		// one pixel out of bounds in each dimension (3-1=2 pixel all together) so that the
		// convolution works
		final int[] imageExtension = kernelDim.clone();		
		for ( int d = 0; d < numDimensions; ++d )
			--imageExtension[ d ];		
		fftImg.setImageExtension( imageExtension );
		
		if ( !fftImg.checkInput() || !fftImg.process() )
		{
			errorMessage = "FFT of image failed: " + fftImg.getErrorMessage();
			return null;			
		}
	
		//
		// compute otf * fft( input )
		//
		long numPixels = fftImg.getResult().getDimension( 0 );
		for ( int d = 1; d < numDimensions; ++d )
			numPixels *= fftImg.getResult().getDimension( d );

		final Vector< Chunk > threadChunks = SimpleMultiThreading.divideIntoChunks( numPixels, getNumThreads() );
		
		final AtomicInteger ai = new AtomicInteger(0);					
        final Thread[] threads = SimpleMultiThreading.newThreads( numThreads );
        for ( int ithread = 0; ithread < threads.length; ++ithread )
            threads[ithread] = new Thread(new Runnable()
            {
                public void run()
                {
                	// get chunk of pixels to process
                	final Chunk myChunk = threadChunks.get( ai.getAndIncrement() );
                	
    				multiply( myChunk.getStartPosition(), myChunk.getLoopSize(), total_otf, fftImg.getResult() );
                }
            });
        
        SimpleMultiThreading.startAndJoin( threads );

        //
		// Compute inverse Fourier Transform
		//		
		final InverseFourierTransform<T, ComplexFloatType> invFFT = new InverseFourierTransform<T, ComplexFloatType>( fftImg.getResult(), fftImg );
		invFFT.setInPlaceTransform( true );
		invFFT.setNumThreads( this.getNumThreads() );

		if ( !invFFT.checkInput() || !invFFT.process() )
		{
			errorMessage = "InverseFFT of image failed: " + invFFT.getErrorMessage();
			return null;			
		}	
		
		return invFFT.getResult();
	}
	

	private final static void conjugateMultiply( final long start, final long loopSize, final Image< ComplexFloatType > fftkernel, final Image< ComplexFloatType > fftimg )
	{
		final Cursor<ComplexFloatType> cursorK = fftkernel.createCursor();
		final Cursor<ComplexFloatType> cursorI = fftimg.createCursor();
		
		cursorK.fwd( start );
		cursorI.fwd( start );
		final ComplexFloatType t = new ComplexFloatType();
		
		for ( long l = 0; l < loopSize; ++l )
		{
			cursorK.fwd();
			cursorI.fwd();
			
			// complex conjugate of kernel (permanent)
			cursorK.getType().complexConjugate();
						
			// multiply with image
			cursorI.getType().mul( cursorK.getType() );
		}
		
		cursorK.close();
		cursorI.close();
	}

	private final static void sumMultiply( final long start, final long loopSize, final Image< ComplexFloatType > fftOTF, final Image< ComplexFloatType > fftCOTF, final Image< ComplexFloatType > target )
	{
		final Cursor<ComplexFloatType> cursorC = fftCOTF.createCursor();
		final Cursor<ComplexFloatType> cursorO = fftOTF.createCursor();
		final Cursor<ComplexFloatType> cursorT = target.createCursor();
		
		cursorC.fwd( start );
		cursorO.fwd( start );
		cursorT.fwd( start );
		
		final ComplexFloatType t = new ComplexFloatType();
		
		for ( long l = 0; l < loopSize; ++l )
		{
			cursorC.fwd();
			cursorO.fwd();
			cursorT.fwd();
			
			t.set( cursorC.getType() );
			t.mul( cursorO.getType() );
			
			// add to target
			cursorT.getType().add( t );
		}
		
		cursorC.close();
		cursorO.close();
		cursorT.close();
	}

	private final static void multiply( final long start, final long loopSize, final Image< ComplexFloatType > fftOFT, final Image< ComplexFloatType > target )
	{
		final Cursor<ComplexFloatType> cursorO = fftOFT.createCursor();
		final Cursor<ComplexFloatType> cursorT = target.createCursor();
		
		cursorO.fwd( start );
		cursorT.fwd( start );
		
		for ( long l = 0; l < loopSize; ++l )
		{
			cursorO.fwd();
			cursorT.fwd();
			
			// add to target
			cursorT.getType().mul( cursorO.getType() );
		}
		
		cursorO.close();
		cursorT.close();
	}

	
	@Override
	public long getProcessingTime() { return processingTime; }
	
	@Override
	public void setNumThreads() { this.numThreads = Runtime.getRuntime().availableProcessors(); }

	@Override
	public void setNumThreads( final int numThreads ) { this.numThreads = numThreads; }

	@Override
	public int getNumThreads() { return numThreads; }	

}
