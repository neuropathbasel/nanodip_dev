# nanodip_dev
Development repostitory of NanoDiP

This is the development branch of NanoDiP, supporting ONT's R10 pore chemistry as well as Illumina's EPIC V2 Human Methylation microarray. It has so far been tested on the **ORIN 32 GB developer kit** from Nvidia. The stable branch is https://github.com/neuropathbasel/nanodip, supporting R9 chemistry and 450K/EPIC V1 microarrays.

A demonstration VM image or the last stable release is available from https://www.epidip.org and https://epidip.usb.ch. NanoDiP supports ONT sequencing control through the MinKNOW API provided by ONT and also represents the computational backend of EpiDiP, https://www.epidip.org and https://epidip.usb.ch.

As a consequence of the basecaller "dorado" https://github.com/nanoporetech/dorado/ now requiring GPU support, NanoDiP requires a GPU to be present for operation on R9 and R10 fast5 data. Both the ORIN 32GB developer kit and cryptocurrency mining boards with 32 GB RAM and suitable CUDA-enabled GPU have been tested. There is currently no installer for the development branch, but a VM (without GPU) will be availale soon on www.epidip.org to provide a working system including (basecalled) demo data.

For the x86_64 platform, request the following binaries from ONT:
```
ubuntu 20.04 (Kernel 5.15)
Distribution:           23.07.5 (STABLE)
MinKNOW Core:           5.7.2
Bream:                  7.7.6
Protocol configuration: 5.7.8
Dorado (build):          7.0.5+b44bfcb66
Dorado (connected):      7.0.9+1d91537ff
```

For the aarch64 / ORIN 32GB version you need:
```
ubuntu 20.04 (L4T 34.1.1)
MinKNOW Core:           5.7.5
Bream:                  7.7.6
Protocol configuration: 5.7.11
Dorado (build):          7.0.5+b44bfcb66
Dorado (connected):      7.1.4+d7df870c0
```
You can (theoratically) use this for post-hoc analysis of fast5 data, but this has not been tested yet.
