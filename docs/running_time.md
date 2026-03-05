## Running time

Here's you'll find the approximate time it takes to run the pipeline on a single 16-hour-long recording:

| Model | Running time (CPU) | Running time (GPU) |
|:------|-------------------:|-------------------:|
| VTC 2.0 |          2.7 hours |         60 seconds |
| BabAR |          2.7 hours |         18 seconds |
| **Total** |      **5.4 hours** |     **78 seconds** |

Note that BabAR's running time depends on the number of KCHI utterances detected by VTC 2.0.
Recordings with more child speech will take a bit longer.

Running times were measured on an AMD EPYC 7453 28-Core processor (8 CPUs). The GPU run used an NVIDIA A40 (45 GB) with 12 CPUs.
