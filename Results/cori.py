import sys
sys.path.append( ".." )

import analysis
import gevfit

import numpy as np
import pandas as pd

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

def main():
    # writeMetaData( 'CoriData.csv', list( range( 85, 235 ) ) )

    writeNormalizedMetaData( 'CoriData.csv', 'CoriNormalized.csv' )

main()
