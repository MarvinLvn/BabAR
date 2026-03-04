## Running time

Here's you'll find the approximate time it takes to run the pipeline on a single 16-hour-long recording:

| Model     | CPU | GPU |
|-----------|-----|-----|
| VTC 2.0   | TBD | 60s |
| BabAR     | TBD | 18s |
| **Total** | **TBD** | **TBD** |

Note that BabAR's running time depends on the number of KCHI utterances detected by VTC.
Recordings with more child speech will take a bit longer.

Running times were measured on an AMD EPYC 7453 28-Core processor (8 CPUs). The GPU run used an NVIDIA A40 (45 GB) with 12 CPUs.
