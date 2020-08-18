import sys
sys.path.append( "/projects/bridges2016099/gev_assessment" )

import analysis
import gevfit

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.axes as axes


def getMetaData( explist ):
    # Get DataFrame of All Experiments
    df_All = analysis.getAllExperiments()

    # Get Cori Runs
    df_Cori = df_All.loc[ df_All[ 'Experiment' ].isin( explist ) ]

    # Create Meta DataFrame
    df_Cori_Meta = pd.DataFrame( columns=[ 'Workload', 'Ranks', 'Stencil', 'Runtime'])

    # Loops Over All Cori Runs
    for df in [ df_Cori ]:
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
                df_Cori_Meta.loc[ len( df_Cori_Meta ) ] = [ workload, int( ranks ), int( stencil ), currentRunTime ]

    return df_Cori_Meta


def writeMetaData( fname, explist ):
    # Get Metadata
    df_Cori_Meta = getMetaData( explist )

    # Write Metadata
    df_Cori_Meta.to_csv( fname )


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
    coriData = pd.read_csv( fname )

    fig, axs = plt.subplots(2, 3, figsize=(10,15))
    fig.suptitle( 'Cori Runtimes' )
    plt.subplots_adjust( wspace = .3, hspace =.3 )


    MILLION = 1000000
    yMin = 0
    yMax = 0
    count = 0

    for run in range( 0, len( coriData ) ):
        label = ''
        workload = coriData['Workload'][run]
        ranks = coriData['Ranks'][run]
        stencil = coriData['Stencil'][run]

        currentRunTime = coriData['Runtime'][run]

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

    _ = axs[0][0].set_ylim( [64, 69] )
    _ = axs[0][0].set_title( 'FTQ' )
    _ = axs[0][0].set_xlabel( 'Ranks' )
    _ = axs[0][0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[0][1].set_ylim( [63, 68] )
    _ = axs[0][1].set_title( 'FWQ' )
    _ = axs[0][1].set_xlabel( 'Ranks' )
    _ = axs[0][1].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[0][2].set_ylim( [360, 440] )
    _ = axs[0][2].set_title( 'DGEMM' )
    _ = axs[0][2].set_xlabel( 'Ranks' )
    _ = axs[0][2].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][0].set_ylim( [780, 860] )
    _ = axs[1][0].set_title( 'SPMV' )
    _ = axs[1][0].set_xlabel( 'Ranks' )
    _ = axs[1][0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][1].set_ylim( [1300, 1450] )
    _ = axs[1][1].set_title( 'HPCG' )
    _ = axs[1][1].set_xlabel( 'Ranks' )
    _ = axs[1][1].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1][2].set_ylim( [1400, 1700] )
    _ = axs[1][2].set_title( 'LAMMPS' )
    _ = axs[1][2].set_xlabel( 'Ranks' )
    _ = axs[1][2].set_ylabel( 'Runtime (Seconds)' )

    plt.savefig( figfile, bbox_inches='tight' )


def generatePredictionFigure( df_All, workload, baseRanks, fname ):
    print( 'Generating Prediction for: ' + workload )

    outfile = 'Figures/Cori_' + workload + '_prediction'

    df_Cori_NoStencil = df_All[ ( df_All['machine'] == 'Cori' ) & ( df_All['processors'] == baseRanks ) & ( df_All['workload'] == workload ) & ( df_All['stencil_size'] == 0 ) ]
    df_Cori_Stencil = df_All [ ( df_All['machine'] == 'Cori' ) & ( df_All['processors'] == baseRanks ) & ( df_All['workload'] == workload ) & ( df_All['stencil_size'] != 0 ) ]

    kvals = np.array( [1, 2, 4] )
    iterations = 50

    MILLION = 1000000
    yMin = 0
    yMax = 0
    count = 0

    fig, axs = plt.subplots( 2, 1, figsize=( 10, 10 ) )
    fig.suptitle( 'Re-Sampled Projections Confidence Interval vs. Actual Runtimes' )

    count = 0
    labels = ['No Stencil', 'Stencil']

    for df in [df_Cori_NoStencil, df_Cori_Stencil]:

        print( labels[count] )

        lowerBound = []
        median = []
        upperBound = []
        for k in kvals:
            expList = []

            for run in range( 0, len( df ) ):
                currentRun = df.iloc[run]
                eid = currentRun['Experiment']
                rid = currentRun['expid']
                currentPath = './mlruns/' + str(eid) + '/' + str(rid) + '/artifacts/bsp-trace.json'
                currentData = analysis.getData(currentPath)
                currentData = currentData[currentData['rank'] == 0]

                for i in range( iterations ):
                    data = analysis.resample_project( currentData, len( currentData ), k, col='interval_max_usec' )
                    projectedRunTime = sum( data ) / MILLION
                    expList.append( projectedRunTime )
                expList.sort()

            lowerBound.append( expList[0] )
            median.append( expList[int( len( expList ) / 2 )] )
            upperBound.append( expList[-1] )

        if count == 0:
            _ = axs[count].plot( baseranks * kvals, median, color='blue' )
            _ = axs[count].fill_between( baseranks * kvals, lowerBound, upperBound, color='blue', alpha=.1 )
        if count == 1:
            _ = axs[count].plot( baseranks * kvals, median, color='red' )
            _ = axs[count].fill_between( baseranks * kvals, lowerBound, upperBound, color='red', alpha=.1)

        if count == 0:
            yMin = min( lowerBound)
            yMax = max( upperBound)
        else:
            if yMin > min( lowerBound ):
                yMin = min( lowerBound )
            if yMax < max( upperBound ):
                yMax = max( upperBound )

        count = count + 1

    coriData = pd.read_csv( fname )

    for run in range( 0, len( coriData ) ):
        label = ''
        workload = coriData['Workload'][run]
        ranks = coriData['Ranks'][run]
        stencil = coriData['Stencil'][run]

        currentRunTime = coriData['Runtime'][run]

        if int( stencil ) > 0:
            label = label + 'Stencil'

        if label == '':
            axs[0].plot( int( ranks ), currentRunTime, 'o', color='red' )

        if label == 'Stencil':
            axs[1].plot( int( ranks ), currentRunTime, 'o', color='blue' )

    _ = axs[0].set_ylim( [0.5 * yMin, 1.5 * yMax] )
    _ = axs[0].set_title( 'No Interference' )
    _ = axs[0].set_xlabel( 'Ranks' )
    _ = axs[0].set_ylabel( 'Runtime (Seconds)' )

    _ = axs[1].set_ylim( [0.5 * yMin, 1.5 * yMax] )
    _ = axs[1].set_title( 'Stencil' )
    _ = axs[1].set_xlabel( 'Ranks' )
    _ = axs[1].set_ylabel( 'Runtime (Seconds)' )

    plt.savefig( outfile, bboxinches='tight' )


def main():
    # writeMetaData( 'Results/CoriData.csv', list( range( 85, 235 ) ) )

    # writeNormalizedMetaData( 'Results/CoriData.csv', 'Results/CoriNormalized.csv' )

    generateRuntimeFigure( 'Results/CoriData.csv', 'Figures/CoriRuntime.png' )

    workloads = ['sleep', 'fwq', 'dgemm', 'spmv', 'hpcg', 'lammps']

    df_All = analysis.getAllExperiments()
    baseRanks = 512

    for workload in workloads:
        generatePredictionFigure( df_All, workload, baseRanks, 'Results/CoriData.csv' )

main()
