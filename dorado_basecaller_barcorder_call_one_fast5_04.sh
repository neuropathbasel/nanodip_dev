#!/bin/bash

# tested with dorado 0.4.2
# call one fast 5, then exit. Employ user-specified settings for kit and basecalling model
# usage: thisscript SAMPLENAME BCMODEL BARCODEKIT

# e.g.
# SAMPLENAME=MinKNOW_experiment_name
# BCMODEL=dna_r9.4.1_e8_hac@v3.3 / dna_r10.4.1_e8.2_400bps_hac@v4.2.0 / dna_r10.4.1_e8.2_260bps_hac@v4.1.0
# BARCODEKIT=SQK-RBK004 / SQK-RBK114-24

samplename=$1
bcmodel=$2
barkit=$3

#config
f5p=_fast5_pass
fqp=_fastq_pass
barcodes="barcode01 barcode02 barcode03 barcode04 barcode05 barcode06 barcode07 barcode08 barcode09 barcode10 barcode11 barcode12 barcode13 barcode14 barcode15 barcode16 barcode17 barcode18 barcode19 barcode20 barcode21 barcode22 barcode23 barcode24 unclassified"

# data pathes
inputdir=/data/$samplename
#outputdir=/data/nanodip_output/$samplename
basecalldir=/data/nanodip_bascalling
tmpdir=/data/dorado_tmp

# binaries & reference data
doradobin=/applications/ont_dorado_versions/dorado-0.4.2-linux-arm64/bin/dorado
bcmodelpath=/applications/ont_dorado_versions/dorado-0.4.2-models

mkdir -p $basecalldir

outputmk5d=`echo "/data/"$samplename"_ND/"`
mkdir -p $outputmk5d

numbercalled=0
numberfast5=`find $inputdir -name '*.fast5' | wc -l`
calledone="no"
for f5 in `find $inputdir -name '*.fast5'`; do
	echo "processing "$f5":"
	b5=$(basename -s .fast5 ${f5})
	echo "basename: "$b5
	d5=$basecalldir/$samplename/$bcmodel-$barkit/$b5
	mkdir -p $d5
	echo ""
	echo "linking "$f5" to "$d5/$b5.fast5
	ln -s $f5 $d5/$b5.fast5
	
	cd $d5
	bcd=$d5/basecalling
	mkdir -p $bcd
	cd $bcd
	allbam=$bcd/bam_all.bam
	if [ -f $allbam ]; then
  	echo "File "$allbam" exists, skipping basecalling."
	else
  	echo "Generating "$allq" by basecalling."
		$doradobin basecaller --device cuda:all --batchsize 256 $bcmodelpath/$bcmodel $d5 > $allbam
		calledone="yes"
	fi
	cd $d5
	bard=$d5/barcoding
	if [ -f $d5/$b5.fastq ]; then
		echo "File "$d5/$b5.fastq" exists. skipping barcoding."
	else
		cd $bard
		#$barcoderbin -i $bcd -s $bard --barcode_kits $barkit --fastq_out --device cuda:all
		$doradobin demux --output-dir $bard --kit-name $barkit --emit-fastq --threads 2 $allbam
		largest=0
		largestfq=""
		for fq in `find $bard/ -name *barcode*`; do
			si=`stat --printf="%s" $fq`
			echo "----- "$fq" = "$si" bytes"
			if [ $largest -lt $si ]; then
				largest=$si
				largestfq=$fq
			fi
		done
		ln -s $largestfq $d5/$b5.fastq
		largestbc="unclassified"
		for bc in $barcodes; do
			if [[ "$largestfq" == *"$bc"* ]]; then
				largestbc=$bc
			fi
		done
	fi

	# link only those files into the analysis directory which have been barcoded properly
	if [[ $largestbc != "" ]]; then
		bcpass5l=`echo $d5/$b5"_"$largestbc"_pass.fast5"`
		bcpassql=`echo $d5/$b5"_"$largestbc"_pass.fastq"`
		
		if [ -f $bcpass5l ]; then
			echo "link "$bcpass5l" exists, skipping."
		else
			ln -s $d5/$b5.fast5 $bcpass5l
		fi
		if [ -f $bcpassql ]; then
			echo "link "$bcpassql" exists, skipping."
		else
			ln -s $d5/$b5.fastq $bcpassql
		fi
		
		outputmk5d=`echo "/data/"$samplename"_ND/"$samplename"_ND/"$bcmodel-$barkit"/fast5_pass/"$largestbc`
		outputmkqd=`echo "/data/"$samplename"_ND/"$samplename"_ND/"$bcmodel-$barkit"/fastq_pass/"$largestbc`
		mkdir -p $outputmk5d
		mkdir -p $outputmkqd
		ln -s $bcpass5l $outputmk5d/
		ln -s $bcpassql $outputmkqd/
		
		numbercalled=$((numbercalled+1))
		echo ""
		echo "called one fast5, exiting"
		echo "####@####@####" # recognition string for output summary
		echo "basecalling completed for "$numbercalled"/"$numberfast5" fast5 files."
		if [[ $calledone == "yes" ]]; then
			break
		fi
	fi
done

