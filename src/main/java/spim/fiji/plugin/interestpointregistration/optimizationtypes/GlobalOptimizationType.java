package spim.fiji.plugin.interestpointregistration.optimizationtypes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import net.imglib2.realtransform.AffineTransform3D;

import spim.fiji.plugin.interestpointregistration.ChannelInterestPointList;
import spim.fiji.plugin.interestpointregistration.ChannelInterestPointListPair;
import spim.fiji.plugin.interestpointregistration.ChannelProcess;
import spim.fiji.plugin.interestpointregistration.Detection;
import spim.fiji.spimdata.SpimData2;
import spim.fiji.spimdata.interestpoints.CorrespondingInterestPoints;
import spim.fiji.spimdata.interestpoints.InterestPoint;
import spim.fiji.spimdata.interestpoints.InterestPointList;
import spim.fiji.spimdata.interestpoints.ViewInterestPointLists;
import spim.fiji.spimdata.interestpoints.ViewInterestPoints;

import mpicbg.spim.data.registration.ViewRegistration;
import mpicbg.spim.data.registration.ViewRegistrations;
import mpicbg.spim.data.registration.ViewTransform;
import mpicbg.spim.data.registration.ViewTransformAffine;
import mpicbg.spim.data.sequence.Angle;
import mpicbg.spim.data.sequence.Illumination;
import mpicbg.spim.data.sequence.TimePoint;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import mpicbg.spim.io.IOFunctions;
import mpicbg.spim.mpicbg.PointMatchGeneric;

/**
 * A certain type of global optimization, must be able to define all view pairs
 * that need to be matched and optimized inidivdually
 * 
 * @author Stephan Preibisch (stephan.preibisch@gmx.de)
 */
public abstract class GlobalOptimizationType
{
	protected boolean save;
	
	public GlobalOptimizationType( final boolean save ) { this.save = save; }
		
	public abstract List< GlobalOptimizationSubset > getAllViewPairs(
			final SpimData2 spimData,
			final ArrayList< Angle > anglesToProcess,
			final ArrayList< ChannelProcess > channelsToProcess,
			final ArrayList< Illumination > illumsToProcess,
			final ArrayList< TimePoint > timepointsToProcess,
			final int inputTransform,
			final double minResolution );

	/**
	 * @param viewId
	 * @param set
	 * @return - true if a certain tile is fixed for global optimization, otherwise false
	 */
	public abstract boolean isFixedTile( final ViewId viewId, final GlobalOptimizationSubset set );
	
	/** 
	 * @return - true if any of the data should be saved
	 */
	public boolean save() { return save; }

	/**
	 * Creates lists of input points for the registration, depending if the input is the current transformation or just the calibration
	 * 
	 * Note: this always duplicates the location array from the input List< InterestPoint > !!!
	 * 
	 * @param timepoint
	 */
	public HashMap< ViewId, ChannelInterestPointList > getInterestPoints(
			final SpimData2 spimData,
			final ArrayList< Angle > anglesToProcess,
			final ArrayList< ChannelProcess > channelsToProcess,
			final ArrayList< Illumination > illumsToProcess,
			final TimePoint timepoint,
			final int inputTransform,
			final double minResolution )
	{
		final HashMap< ViewId, ChannelInterestPointList > interestPoints = new HashMap< ViewId, ChannelInterestPointList >();
		final ViewRegistrations registrations = spimData.getViewRegistrations();
		final ViewInterestPoints interestpoints = spimData.getViewInterestPoints();
		
		for ( final Angle a : anglesToProcess )
			for ( final Illumination i : illumsToProcess )
				for ( final ChannelProcess c : channelsToProcess )
			{
				// bureaucracy
				final ViewId viewId = SpimData2.getViewId( spimData.getSequenceDescription(), timepoint, c.getChannel(), a, i );
				
				final ViewDescription< TimePoint, ViewSetup > viewDescription = spimData.getSequenceDescription().getViewDescription( 
						viewId.getTimePointId(), viewId.getViewSetupId() );

				if ( !viewDescription.isPresent() )
					continue;

				// update the registrations if required
				if ( inputTransform == 0 )
				{
					final ViewRegistration r = registrations.getViewRegistration( viewId );
					r.identity();
					
					final double calX = viewDescription.getViewSetup().getPixelWidth() / minResolution;
					final double calY = viewDescription.getViewSetup().getPixelHeight() / minResolution;
					final double calZ = viewDescription.getViewSetup().getPixelDepth() / minResolution;
					
					final AffineTransform3D m = new AffineTransform3D();
					m.set( calX, 0.0f, 0.0f, 0.0f, 
						   0.0f, calY, 0.0f, 0.0f,
						   0.0f, 0.0f, calZ, 0.0f );
					final ViewTransform vt = new ViewTransformAffine( "calibration", m );
					r.preconcatenateTransform( vt );
				}

				// assemble a new list
				final ArrayList< InterestPoint > list = new ArrayList< InterestPoint >();

				// check the existing lists of points
				final ViewInterestPointLists lists = interestpoints.getViewInterestPointLists( viewId );

				if ( !lists.contains( c.getLabel() ) )
				{
					IOFunctions.println( "Interest points for label '" + c.getLabel() + "' not found for timepoint: " + timepoint.getId() + " angle: " + 
							a.getId() + " channel: " + c.getChannel().getId() + " illum: " + i.getId() );
					
					continue;
				}
				
				if ( lists.getInterestPointList( c.getLabel() ).getInterestPoints().size() == 0 )
				{
					if ( !lists.getInterestPointList( c.getLabel() ).loadInterestPoints() )
					{
						IOFunctions.println( "Interest points for label '" + c.getLabel() + "' could not be loaded for timepoint: " + timepoint.getId() + " angle: " + 
								a.getId() + " channel: " + c.getChannel().getId() + " illum: " + i.getId() );
						
						continue;						
					}
				}
				
				final List< InterestPoint > ptList = lists.getInterestPointList( c.getLabel() ).getInterestPoints();
				
				final ViewRegistration r = registrations.getViewRegistration( viewId );
				final AffineTransform3D m = r.getModel();
				
				for ( final InterestPoint p : ptList )
				{
					final float[] l = new float[ 3 ];
					m.apply( p.getL(), l );
					
					list.add( new InterestPoint( p.getId(), l ) );
				}
				
				interestPoints.put( viewId, new ChannelInterestPointList( list, c ) );
			}
		
		return interestPoints;
	}

	/**
	 * Add all correspondences the list for those that are compared here
	 * 
	 * This method can be overwritten if saving, adding & clearing of correspondences is different for a certain type of registration
	 * 
	 * @param pairs
	 */
	public void addCorrespondences( final SpimData2 spimData, final ArrayList< ChannelInterestPointListPair > pairs )
	{
		for ( final ChannelInterestPointListPair pair : pairs )
		{
			final ArrayList< PointMatchGeneric< Detection > > correspondences = pair.getInliers();
			
			final String labelA = pair.getChannelProcessedA().getLabel();
			final String labelB = pair.getChannelProcessedB().getLabel();
			
			final ViewId viewA = pair.getViewIdA();
			final ViewId viewB = pair.getViewIdB();
			
			final InterestPointList listA = spimData.getViewInterestPoints().getViewInterestPointLists( viewA ).getInterestPointList( labelA );				
			final InterestPointList listB = spimData.getViewInterestPoints().getViewInterestPointLists( viewB ).getInterestPointList( labelB );
			
			final List< CorrespondingInterestPoints > corrListA = listA.getCorrespondingInterestPoints();
			final List< CorrespondingInterestPoints > corrListB = listB.getCorrespondingInterestPoints();
			
			for ( final PointMatchGeneric< Detection > d : correspondences )
			{
				final Detection dA = d.getPoint1();
				final Detection dB = d.getPoint2();
				
				final CorrespondingInterestPoints correspondingToA = new CorrespondingInterestPoints( dA.getId(), viewB, labelB, dB.getId() );
				final CorrespondingInterestPoints correspondingToB = new CorrespondingInterestPoints( dB.getId(), viewA, labelA, dA.getId() );
				
				corrListA.add( correspondingToA );
				corrListB.add( correspondingToB );
			}
		}		
	}
	
	/**
	 * Save all lists of existing correspondences for those that are compared here
	 * 
	 * This method can be overwritten if saving, adding & clearing of correspondences is different for a certain type of registration
	 *
	 * @param set
	 */
	public void saveCorrespondences( final SpimData2 spimData, final ArrayList< ChannelProcess > channelsToProcess, final GlobalOptimizationSubset set )
	{
		for ( final ViewId id : set.getViews() )
			for ( final ChannelProcess c : channelsToProcess )
				spimData.getViewInterestPoints().getViewInterestPointLists( id ).getInterestPointList( c.getLabel() ).saveCorrespondingInterestPoints();		
	}
	
	/**
	 * Clear all lists of existing correspondences for those that are compared here
	 * 
	 * This method can be overwritten if saving, adding & clearing of correspondences is different for a certain type of registration
	 *
	 * @param set
	 */
	public void clearExistingCorrespondences( final SpimData2 spimData, final ArrayList< ChannelProcess > channelsToProcess, final GlobalOptimizationSubset set )
	{
		for ( final ViewId id : set.getViews() )
			for ( final ChannelProcess c : channelsToProcess )
				spimData.getViewInterestPoints().getViewInterestPointLists( id ).getInterestPointList( c.getLabel() ).getCorrespondingInterestPoints().clear();		
	}

}