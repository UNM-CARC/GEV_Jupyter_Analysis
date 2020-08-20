mlruns:
    1:  Wheeler, Verbose (Finished)
        Nodes: 1, 2, 4, 8, 16, 32, 48
        Cores per Node: 1, 2, 4, 8
        Iterations: 10000
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Workloads:
            sleep -a 100000 -b 1000
            sleep -a 100000 -b 10000
            dgemm -a 128 -b 64
            dgemm -a 256 -b 16

    2:  Wheeler, Verbose (Finished - removed some 32 and 48 node jobs)
        Nodes: 1, 2, 4, 8, 16, 32, 48
        Cores per Node: 1, 2, 4, 8
        Iterations: 10000
        Stencil: 0
        Inner Loops: 1
        Rabbit: False, True, False
        Osu: False, False, True
        Workloads:
            sleep -a 100000 -b 1000
            sleep -a 100000 -b 10000
            dgemm -a 128 -b 64
            dgemm -a 256 -b 16

    3.  Stampede2, KNL, Verbose (Finished - had issues on 16, 32, 48, etc nodes)
        Nodes: 1, 2, 4, 8
        Cores per Node: 8, 68
        Iterations: 1000
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False

        Workloads:
            sleep -a 100000 -b 10000
            dgemm -a 128 -b 64
            io -z 1024


    4:  Wheeler, Verbose (Finished - removed some 16 node osu jobs)
        Nodes: 1, 2, 4, 8, 16
        Cores per Node: 1, 2, 4, 8
        Iterations: 1000
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False, True, False
        Osu: False, False, True
        Workloads:
            sleep -a 100000 -b 1000
            sleep -a 100000 -b 10000
            dgemm -a 128 -b 64
            dgemm -a 256 -b 16
            dgemm -a 256 -b 64

    5:  Wheeler, Verbose (Finished - removed some 16 node osu jobs)
        Nodes: 1, 2, 4, 8, 16
        Cores per Node: 1, 2, 4, 8
        Iterations: 1000
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Workloads:
            sleep -a 100000 -b 1000
            sleep -a 100000 -b 10000
            dgemm -a 128 -b 64
            dgemm -a 256 -b 16
            dgemm -a 256 -b 64

    6:  Wheeler, Verbose (8 Total Runs: 2 Workloads each with and without stencil and with and without osu)
        Nodes: 16
        Cores per Node: 4
        Iterations: 1000
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False True False True
        Workloads:
            dgemm -a 128 -b 64
            dgemm -a 256 -b 64


    7:  Wheeler, Verbose (Ran 1 at a time - 8 Total Runs: 2 Workloads each with and without stencil and with and without osu)
        Nodes: 16
        Cores per Node: 4
        Iterations: 1000
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False True False True
        Workloads:
            dgemm -a 128 -b 64
            dgemm -a 256 -b 64

Experiments 8 - 18 were ran outside of the container. Used python scripts to call mlflow after the fact and track params and metrics

   8:  Stampede2, KNL, Verbose (4 Total Runs on the same allocation  with and without stencil and with and without rabbit)
        Nodes: 8
        Cores per Node: 64
        Iterations: 250
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: True False True False
        Osu: False
        Workloads:
            dgemm -a 256 -b 64

  9:  Stampede, Skylake, Verbose (4 Total Runs on the same allocation  with and without stencil and with and without rabbit)
        Nodes: 11
        Cores per Node: 46/47 (total 512)
        Iterations: 250
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: True False True False
        Osu: False
        Workloads:
            dgemm -a 256 -b 64


  10: Wheeler, Non-Verbose (4 Total Runs:  with and without stencil and with and without rabbit)
        Nodes: 32
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: True False True False
        Osu: False
        Workloads:
            dgemm -a 128 -b 64


  11: Wheeler repeat of exp 10


  12: Wheeler repeat of exp 10,11 Verbose


  13: Wheeler, Verbose (4 Total Runs :  with and without stencil and with and without rabbit)
        Nodes: 32
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0 0 32768 32768
        Inner Loops: 1
        Rabbit: True False True False
        Osu: False
        Workloads:
            dgemm -a 256 -b 64

  14: Stampede2, KNL,  Non-Verbose
        Nodes: 1, 2, 4, 8
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0 32768
        Inner Loops: 1
        Rabbit: False False
        Osu: False False
        Workloads:
            dgemm -a 256 -b 64

  15: Stampede2, KNL, Non-Verbose
        Nodes: 1, 2, 4, 8, 16, 32
        Cores per Node: 8, 64
        Iterations: 1000
        Stencil: 0 32768
        Inner Loops: 1
        Rabbit: False False
        Osu: False False
        Workloads:
            dgemm -a 256 -b 64

    16. Stampede 2, KNL Cache Mode (Same as all previous KNL), Non-Verbose
        Nodes: 1, 2, 4, 8, 16
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: True False
        Workloads:
            dgemm -a 256 -b 64

    17. Stampede 2, KNL Flat Mode (Using numactl on normal queue), Non-Verbose
        Nodes: 1, 2, 4, 8, 16
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: True False
        Workloads:
            dgemm -a 256 -b 64

    18. Stampede 2, Skylake, Non-Verbose
        Nodes: 1, 2, 4, 8, 16
        Cores per Node: 8
        Iterations: 1000
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: True False
        Workloads:
            dgemm -a 256 -b 64

    19. Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    20. Same as 19

    21. Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 250
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    22-26. Same as 21

    27. Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iterations: 500
        Stencil: 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False True True
        Workloads:
            dgemm -a 256 -b 64

    28-37: Stampede2, Skylake, Verbose
        Nodes: 11
        Cores per Node: 48
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    38-46: Stampede2, Skylake, Verbose
        Nodes: 6
        Cores per Node: 48
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    47-49: Stampede2, Skylake, Verbose
        Nodes: 22
        Cores per Node: 48
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    50-64: Wheeler, Verbose
        Nodes: 1
        Cores per Node: 7
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    65-79: Wheeler, Verbose
        Nodes: 4
        Cores per Node: 7
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    80-84: Wheeler, Verbose
        Nodes: 8
        Cores per Node: 7
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    85-104: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    105-114: Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768, 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False False True True
        Workloads:
            dgemm -a 256 -b 64

    115-117: Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            dgemm -a 256 -b 64

    118-127: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            sleep -a 100000 -b 10000

    128-132: Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            sleep -a 100000 -b 10000

    133-135: Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            sleep -a 100000 -b 10000

    136-155: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            fwq -a 100000 -b 10000

    156-165: Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            fwq -a 100000 -b 10000

    166-168: Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            fwq -a 100000 -b 10000

    169-188: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 200
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            spmv -a 114 -b 25

    189-198: Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 200
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            spmv -a 114 -b 25

    199-201: Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iterations: 200
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            spmv -a 114 -b 25

    202-211: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            hpcg -a 114 -b 25

    212-216: Cori, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            hpcg -a 114 -b 25

    217-219: Cori, Verbose
        Nodes: 64
        Cores per Node: 32
        Iteratiions: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            hpcg -a 114 -b 25

    220-229: Cori, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            lammps -a 64000 -b 250

    230-234: Cori, Verbose
        Nodes: 54
        cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            lammps -a 64000 -b 250

    235-254: Attaway, Verbose
        Nodes: 8
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Looops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            dgemm -a 256 -b 64

    255-264: Attaway, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            dgemm -a 256 -b 64

    265-274: Attaway, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            dgemm -a 256 -b 64

    275-279: Attaway, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            sleep -a 100000 -b 10000

    280-284: Attaway, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            sleep -a 100000 -b 10000

    285-289: Attaway, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            fwq -a 100000 -b 10000

    290-294: Attaway, Verbose
        Nodes: 32
        Cores per Node: 32
        Iterations: 500
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            fwq -a 100000 -b 10000

    295-304: Attaway, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 200
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            spmv -a 114 -b 25

    305-309: Attaway, Verbose (Waiting on)
        Nodes: 32
        Cores per Node: 32
        Iterations: 200
        Stencil: 0, 32768
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            spmv -a 114 -b 25

    310-314: Attaway, Verbose
        Nodes: 16
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            hpcg -a 114 -b 25

    315-319: Attaway, Verbose (Waiting on)
        Nodes: 32
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            hpcg -a 114 -b 25

    320-324: Attaway, Verbose (Waiting on)
        Nodes: 16
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            lammps -a 64000 -b 250

    325-329: Attaway, Verbose (Waiting on)
        Nodes: 54
        Cores per Node: 32
        Iterations: 100
        Stencil: 0
        Inner Loops: 1
        Rabbit: False
        Osu: False
        Rabbit_Workload: False
        Workloads:
            lammps -a 64000 -b 250

