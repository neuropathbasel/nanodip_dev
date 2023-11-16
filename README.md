# nanodip_dev
Development repostitory of NanoDiP

This is the development branch of NanoDiP, supporting ONT's R10 pore chemistry as well as Illumina's EPIC V2 Human Methylation microarray. It has so far been tested on the **ORIN 32 GB developer kit** from Nvidia. The stable branch is https://github.com/neuropathbasel/nanodip, supporting R9 chemistry and 450K/EPIC V1 microarrays.

A demonstration VM image or the last stable release is available from https://www.epidip.org and https://epidip.usb.ch. NanoDiP supports ONT sequencing control through the MinKNOW API provided by ONT and also represents the computational backend of EpiDiP, https://www.epidip.org and https://epidip.usb.ch.

As a consequence of the basecaller "dorado" https://github.com/nanoporetech/dorado/ now requiring GPU support, NanoDiP requires a GPU to be present for operation on R9 and R10 fast5 data. Both the ORIN 32GB developer kit and cryptocurrency mining boards with 32 GB RAM and suitable CUDA-enabled GPU have been tested. There is currently no installer for the development branch, but a VM (without GPU) will be availale soon on www.epidip.org to provide a working system including (basecalled) demo data.
