import sys
sys.path.append( "/projects/bridges2016099/gev_assessment" )

import analysis
import gevfit

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.axes as axes

from scipy import stats


def getMetaData( explist ):
    # Get DataFrame of All Experiments
    df_All = analysis.getAllExperiments()

    # Get Cori Runs
    df_Attaway = df_All.loc[ df_All[ 'Experiment' ].isin( explist ) ]

    # Create Meta DataFrame
    df_Attaway_Meta = pd.DataFrame( columns=[ 'Workload', 'Ranks', 'Stencil', 'Runtime'])

    # Loops Over All Cori Runs
    for df in [ df_Attaway ]:
        for run in range( 0, len( df ) ):
            currentRun = df.iloc[ run ]

            eid = currentRun['Experiment']
            workload = currentRun[ 'workload' ]
            ranks = currentRun[ 'processors' ]
            stencil = currentRun[ 'stencil_size' ]
            rab_work = currentRun[ 'rabbit_workload' ]
            rid = currentRun[ 'expid' ]

            currentPath = './mlruns/' + str(eid) + '/' + str(rid) + '/artifacts/bsp-trace.json'

            # Print Progress
            print( eid, currentPath )

            # Check that Rabbit is Off
            if ( rab_work == 0 ):
                # Get Data
                currentData = analysis.getData( currentPath )
                currentData = currentData[ currentData[ 'rank' ] == 0 ]

                # Calculate Total Runtime
                currentRunTime = currentData[ 'interval_max_usec' ].sum() / 1000000

                # Add to Metadata
                df_Attaway_Meta.loc[ len( df_Attaway_Meta ) ] = [ workload, int( ranks ), int( stencil ), currentRunTime ]

    return df_Attaway_Meta


def writeMetaData( fname, explist ):
    # Get Metadata
    df_Attaway_Meta = getMetaData( explist )

    # Write Metadata
    df_Attaway_Meta.to_csv( fname )


def normalizeMetaData( infile, outfile ):
    df_Meta = pd.read_csv( infile, index_col=0 )

    # Create Normalized Meta DataFrame
    df_Normalized_Meta = pd.DataFrame( columns=[ 'Workload', 'Ranks', 'Stencil', 'Normalized_Runtime'])

    workloads = df_Meta.Workload.unique().tolist()
    stencils = df_Meta.Stencil.unique().tolist()
    ranks = df_Meta.Ranks.unique().tolist()

    for workload in workloads:
        for rank in ranks:
            for stencil in stencils:
                avg_Runtime = 0
                total_Runtime = 0
                runCount = 0

                for run in range( 0, len( df_Meta ) ):
                    currentRun = df_Meta.iloc[ run ]

                    if ( ( currentRun[ 'Workload' ] == workload ) and ( currentRun[ 'Ranks' ] == rank ) and ( currentRun[ 'Stencil' ] == stencil ) ):
                        total_Runtime = total_Runtime + currentRun[ 'Runtime' ]
                        runCount = runCount + 1

                if ( runCount > 0 ):
                    avg_Runtime = total_Runtime / runCount

                    for run in range( 0, len( df_Meta ) ):
                        currentRun = df_Meta.iloc[ run ]

                        normalized_Runtime = currentRun[ 'Runtime' ] / avg_Runtime

                        if ( ( currentRun[ 'Workload' ] == workload ) and ( currentRun[ 'Ranks' ] == rank ) and ( currentRun[ 'Stencil' ] == stencil ) ):
                            df_Normalized_Meta.loc[ len( df_Normalized_Meta ) ] = [ workload, int( rank ), int( stencil ), normalized_Runtime ]

                    normalized_Runtimes = df_Normalized_Meta[ ( df_Normalized_Meta[ 'Workload' ] == workload ) & ( df_Normalized_Meta[ 'Ranks' ] == rank ) & ( df_Normalized_Meta[ 'Stencil' ] == stencil ) ].Normalized_Runtime.tolist()

                    print( workload, rank, stencil, runCount )
                    print( min( normalized_Runtimes ), max( normalized_Runtimes ), max( normalized_Runtimes ) - min( normalized_Runtimes ) )

    return df_Normalized_Meta


def writeNormalizedMetaData( infile, outfile ):
    df_Normalized_Meta = normalizeMetaData( infile, outfile )

    df_Normalized_Meta.to_csv( outfile )


def generateRuntimeFigure( fname, figfile ):
    attawayData = pd.read_csv( fname )

    fig, axs = plt.subplots(2, 3, figsize=(10,15))
    fig.suptitle( 'Atttaway Runtimes' )
    plt.subplots_adjust( wspace = .3, hspace =.3 )


    MILLION = 1000000
    yMin = 0
    yMax = 0
    count = 0

    for run in range( 0, len( attawayData ) ):
        label = ''
        workload = attawayData['Workload'][run]
        ranks = attawayData['Ranks'][run]
        stencil = attawayData['Stencil'][run]

        currentRunTime = attawayData['Runtime'][run]

        if int(stencil) > 0:
            label = label + 'Stencil'

        if label == '':
            if ( workload == 'sleep' ):
                axs[0][0].plot( int( ranks ), currentRunTime, 'o', color='red' ),
            elif ( workload == 'fwq' ):
                axs[0][1].plot( int( ranks ), currentRunTime, 'o', color='red' ),
            elif ( workload == 'dgemm' ):
                axs[0][2].plot( int( ranks ), currentRunTime, 'o', color='red' ),
            elif ( workload == 'spmv' ):
                axs[1][0].plot( int( ranks ), currentRunTime, 'o', color='red' ),
            elif ( workload == 'hpcg' ):
                axs[1][1].plot( int( ranks ), currentRunTime, 'o', color='red' ),
            elif ( workload == 'lammps' ):
                axs[1][2].plot( int( ranks ), currentRunTime, 'o', color='red' )

        elif label == 'Stencil':
            if ( workload == 'sleep' ):
                axs[0][0].plot( int( ranks ), currentRunTime, 'o', color='blue' ),
            elif ( workload == 'fwq' ):
                axs[0][1].plot( int( ranks ), currentRunTime, 'o', color='blue' ),
            elif ( workload == 'dgemm' ):
                axs[0][2].plot( int( ranks ), currentRunTime, 'o', color='blue' ),
            elif (workload == 'spmv' ):
                axs[1][0].plot( int( ranks ), currentRunTime, 'o', color='blue' ),
            elif ( workload == 'hpcg' ):
                axs[1][1].plot( int( ranks ), currentRunTime, 'o', color='blue' ),
            elif ( workload == 'lammps' ):
                axs[1][2].plot( int( ranks ), currentRunTime, 'o', color='blue' )

        if count == 0:
            yMin = currentRunTime
            yMax = currentRunTime
        else:
            if yMin > currentRunTime:
                yMin = currentRunTime
            elif yMax < currentRunTime:
                yMax = currentRunTime

        count = count + 1

    _ = axs[0][0].set_ylim( [64, 67] )
    _ = axs[0][0].set_title( 'FTQ' )
    _ = axs[0][0].set_xlabel( 'Ranks' )
    _ = axs[0][0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[0][1].set_ylim( [88, 93] )
    _ = axs[0][1].set_title( 'FWQ' )
    _ = axs[0][1].set_xlabel( 'Ranks' )
    _ = axs[0][1].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[0][2].set_ylim( [400, 450] )
    _ = axs[0][2].set_title( 'DGEMM' )
    _ = axs[0][2].set_xlabel( 'Ranks' )
    _ = axs[0][2].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][0].set_ylim( [550, 570] )
    _ = axs[1][0].set_title( 'SPMV' )
    _ = axs[1][0].set_xlabel( 'Ranks' )
    _ = axs[1][0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][1].set_ylim( [950, 1050] )
    _ = axs[1][1].set_title( 'HPCG' )
    _ = axs[1][1].set_xlabel( 'Ranks' )
    _ = axs[1][1].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][2].set_ylim( [1000, 1500] )
    _ = axs[1][2].set_title( 'LAMMPS' )
    _ = axs[1][2].set_xlabel( 'Ranks' )
    _ = axs[1][2].set_ylabel( 'Runtime (Seconds)' )

    plt.savefig( figfile, bbox_inches='tight' )


def generatePredictionFigure( df_All, workload, baseRanks, CI, fname ):
    print( 'Generating Prediction for: ' + workload )

    outfile = 'Figures/Attaway_' + workload + '_prediction'
    coriData = pd.read_csv( fname )
    workloadData = coriData[ ( coriData['Workload'] == workload ) & ( coriData['Ranks'] == baseRanks )]

    kvals = np.array( [1, 2, 4] )
    iterations = 1000

    MILLION = 1000000
    eps = 10 ** -5
    yMin = 0
    yMax = 0
    count = 0

    fig, axs = plt.subplots( 2, 1, figsize=( 10, 10 ) )
    fig.suptitle( 'Attaway Projection Confidence Interval vs. Actual Runtimes' )

    count = 0
    labels = ['No Stencil', 'Stencil']
    if workload == 'hpcg' or workload == 'lammps':
        df_Attaway_NoStencil = df_All[ ( df_All['machine'] == 'Attaway' ) & ( df_All['processors'] == baseRanks ) & ( df_All['workload'] == workload ) & ( df_All['stencil_size'] == 0 ) ]
        df_List = [df_Attaway_NoStencil]
    else:
        df_Attaway_NoStencil = df_All[ ( df_All['machine'] == 'Attaway' ) & ( df_All['processors'] == baseRanks ) & ( df_All['workload'] == workload ) & ( df_All['stencil_size'] == 0 ) ]
        df_Attaway_Stencil = df_All [ ( df_All['machine'] == 'Attaway' ) & ( df_All['processors'] == baseRanks ) & ( df_All['workload'] == workload ) & ( df_All['stencil_size'] != 0 ) ]
        df_List = [df_Attaway_NoStencil, df_Attaway_Stencil]

    for df in df_List:

        print( labels[count] )

        lowerBound = []
        middle = []
        upperBound = []

        if ( count == 0 ):
            runtimeData = workloadData[workloadData['Stencil'] == 0]['Runtime']
        else:
            runtimeData = workloadData[workloadData['Stencil'] != 0]['Runtime']

        minRuntime = min( runtimeData )
        medianRuntime = runtimeData.iloc[ int( len( runtimeData ) / 2  ) ]
        maxRuntime = max( runtimeData )

        for k in kvals:
            expListMin = []
            expListMed = []
            expListMax = []

            for run in range( 0, len( df ) ):
                currentRun = df.iloc[run]
                eid = currentRun['Experiment']
                rid = currentRun['expid']
                currentPath = './mlruns/' + str( eid ) + '/' + str( rid ) + '/artifacts/bsp-trace.json'
                currentData = analysis.getData( currentPath )
                currentData = currentData[currentData['rank'] == 0]
                currentRuntime = sum( currentData['interval_max_usec'] ) / MILLION

                if ( abs( currentRuntime - maxRuntime ) < eps ) or ( abs( currentRuntime - medianRuntime ) < eps ) or ( abs( currentRuntime - minRuntime ) < eps ):
                    for i in range( iterations ):
                        data = analysis.resample_project( currentData, len( currentData ), k, col='interval_max_usec' )
                        projectedRunTime = sum( data ) / MILLION
                        print( 'Run:', run + 1, '/', len( df ), '\tIteration:', i + 1, '/', iterations, '\tWorkload:', workload, '\tk:', k, '\tProjected Runtime:', projectedRunTime )

                        if ( abs( currentRuntime - minRuntime ) < eps ):
                            expListMin.append( projectedRunTime )
                        if ( abs( currentRuntime - medianRuntime ) < eps ):
                            expListMed.append( projectedRunTime )
                        if ( abs( currentRuntime - maxRuntime ) < eps ):
                            expListMax.append( projectedRunTime )

            expListMin.sort()
            expListMed.sort()
            expListMax.sort()

            meanMin = np.mean( np.array( expListMin ) )
            stvMin = stats.sem( np.array( expListMin ) )

            meanMed = np.mean( np.array( expListMed ) )
            stvMed = stats.sem( np.array( expListMed ) )

            meanMax = np.mean( np.array( expListMax ) )
            stvMax = stats.sem( np.array( expListMax ) )

            intervalMin = stats.norm.interval( CI, loc=meanMin, scale=stvMin )
            intervalMed = stats.norm.interval( CI, loc=meanMed, scale=stvMed )
            intervalMax = stats.norm.interval( CI, loc=meanMax, scale=stvMax )

            print( CI, '$ Confidence Interval:', intervalMin[0], intervalMax[1], '\tMedian:', ( intervalMed[0] + intervalMed[1] ) / 2 )

            lowerBound.append( intervalMin[0] )
            middle.append( ( intervalMed[0] + intervalMed[1] ) / 2 )
            upperBound.append( intervalMax[1] )

        if count == 0:
            _ = axs[count].plot( baseRanks * kvals, middle, color='red' )
            _ = axs[count].fill_between( baseRanks * kvals, lowerBound, upperBound, color='red', alpha=.1 )
        if count == 1:
            _ = axs[count].plot( baseRanks * kvals, middle, color='blue' )
            _ = axs[count].fill_between( baseRanks * kvals, lowerBound, upperBound, color='blue', alpha=.1)

        if count == 0:
            yMin = min( lowerBound)
            yMax = max( upperBound)
        else:
            if yMin > min( lowerBound ):
                yMin = min( lowerBound )
            if yMax < max( upperBound ):
                yMax = max( upperBound )

        count = count + 1

    attawayData = pd.read_csv( fname )

    for run in range( 0, len( attawayData ) ):
        label = ''
        myworkload = attawayData['Workload'][run]
        ranks = attawayData['Ranks'][run]
        stencil = attawayData['Stencil'][run]

        currentRunTime = attawayData['Runtime'][run]

        if int( stencil ) > 0:
            label = label + 'Stencil'

        if label == '':
            if myworkload == workload:
                axs[0].plot( int( ranks ), currentRunTime, 'o', color='red' )

        if not ( workload == 'hpcg' or workload == 'spmv' ):
            if label == 'Stencil':
                if myworkload == workload:
                    axs[1].plot( int( ranks ), currentRunTime, 'o', color='blue' )

    _ = axs[0].set_ylim( [0.95 * yMin, 1.05 * yMax] )
    _ = axs[0].set_title( 'No Interference' )
    _ = axs[0].set_xlabel( 'Ranks' )
    _ = axs[0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1].set_ylim( [0.95 * yMin, 1.05 * yMax] )
    _ = axs[1].set_title( 'Stencil' )
    _ = axs[1].set_xlabel( 'Ranks' )
    _ = axs[1].set_ylabel( 'Runtime (Seconds)' )

    plt.savefig( outfile, bboxinches='tight' )


def main():
    expRange = list( range( 235, 330 ) )
    # writeMetaData( 'Results/AttawayData.csv', list( expRange )

    # writeNormalizedMetaData( 'Results/AttawayData.csv', 'Results/AttawayNormalized.csv' )

    generateRuntimeFigure( 'Results/AttawayData.csv', 'Figures/AttawayRuntime.png' )

    workloads = ['sleep', 'fwq', 'dgemm', 'spmv', 'hpcg', 'lammps']

    df_All = analysis.getAllExperiments()
    df_All = df_All.loc[ df_All[ 'Experiment' ].isin( expRange ) ]

    CI = 0.95

    for workload in workloads:
        if workload == 'dgemm':
            baseRanks = 256
        else:
            baseRanks = 512

        generatePredictionFigure( df_All, workload, baseRanks, CI, 'Results/AttawayData.csv' )

main()
