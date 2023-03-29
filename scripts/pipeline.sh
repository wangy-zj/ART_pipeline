#!/usr/bin/env bash

echo="echo -e"
trap ctrl_c INT

function ctrl_c() {
    $echo "** Trapped CTRL-C"
    cleanup
    exit
}

declare -a pids
declare -a keys

function cleanup {
    if [ ${#pids[@]} -gt 0 ]
    then
	$echo "existing pids are ${pids[@]}"
	for pid in "${pids[@]}"
	do
	    $echo "killing pid $pid"
	    kill -9 $pid
	done
    else
	$echo "we do not have any existing processes"
    fi
    
    if [ ${#keys[@]} -gt 0 ]
    then
	$echo "existing ring buffer keys are ${keys[@]}"
       for key in "${keys[@]}"
       do
    	   $echo "removing ring buffer $key"
    	   dada_db -k $key -d
       done
    else
	$echo "we do not have any existing ring buffers"
    fi
}

#$echo "This is a pipeline for beamforming at NAOC\n"

# There will be three ring buffers
# The first one is used to receive raw data
# The second one is used to receive beamformed data
# The last one is used to receive zoom fft data

# The first ring buffer need two readers as we need to read its data to GPU and write its data to disk
# The second ring buffer has only one reader
# The last ring buffer has only one reader as well

# setup command lines
WORK_ROOT=/home/hero/code
project_root=$WORK_ROOT/PAF_pipeline
hdr_root=$project_root/header
udp_command=$project_root/build/udp/udp2db
pipeline_command=$project_root/build/pipeline/pipeline_dada_beamform

$echo "Setting up ring buffers"

key_raw=a000
key_bmf=b000
key_zoo=c000

$echo "key_raw is: $key_raw"
$echo "key_bmf is: $key_bmf"
$echo "key_zoo is: $key_zoo\n"

nreader_raw=128
nreader_bmf=1
nreader_zoo=1

$echo "nreader_raw is: $nreader_raw"
$echo "nreader_bmf is: $nreader_bmf"
$echo "nreader_zoo is: $nreader_zoo\n"

pkt_dtsz=4096
nstream_gpu=24
nstream_bmf=1
npkt=2048
numa=0
bufsz=$(( pkt_dtsz*nstream_gpu*npkt ))
$echo "pkt_dtsz is:    $pkt_dtsz"
$echo "nstream_gpu is: $nstream_gpu"
$echo "nstream_bmf is: $nstream_bmf"
$echo "npkt is:        $npkt"
$echo "numa is:        $numa\n"

# it will be more flexible if we put equation here to calculate buffer size with some basic configrations
bufsz_raw=$bufsz # buffer block size to hold raw data in bytes, change it to real value later
bufsz_bmf=$(( bufsz*nstream_bmf/nstream_gpu )) # buffer block size to hold beamformed data in bytes, change it to real value later
bufsz_zoo=$bufsz_bmf # buffer block size to hold zoom data in bytes, change it to real value later

$echo "bufsz_raw is: $bufsz_raw"
$echo "bufsz_bmf is: $bufsz_bmf"
$echo "bufsz_zoo is: $bufsz_zoo"

numa=0 # numa node to use 
$echo "numa is: $numa\n"

$echo "Creating ring buffers"
dada_raw="dada_db -k $key_raw -b $bufsz_raw -p -w -c $numa -r 2"  #assign memory from NUMA node  [default: all nodes]
dada_bmf="dada_db -k $key_bmf -b $bufsz_bmf -p -w -c $numa"
dada_zoo="dada_db -k $key_zoo -b $bufsz_zoo -p -w -c $numa"

$echo "dada_raw is: $dada_raw"
$echo "dada_bmf is: $dada_bmf"
$echo "dada_zoo is: $dada_zoo\n"

$dada_raw & # should be unblock 
pids+=(`echo $! `)
keys+=(`echo $key_raw `)
$dada_bmf & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_bmf `)
$dada_zoo & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_zoo `)

sleep 1s # to wait all buffers are created 
$echo "created all ring buffers\n"

$echo "Setting file writers"
# different type of files should go to different directories
dir_raw=/home/hero/data/data_raw/ # change it to real value later
dir_bmf=/home/hero/data/data_bmf/ # change it to real value later
dir_zoo=/home/hero/data/data_zoom/ # change it to real value later

$echo "dir_raw is: $dir_raw"
$echo "dir_bmf is: $dir_bmf"
$echo "dir_zoo is: $dir_zoo\n"

# 设定写入数据的地址和数据来源ringbuffer
writer_raw="dada_dbdisk -D $dir_raw -k $key_raw -W" 
writer_bmf="dada_dbdisk -D $dir_bmf -k $key_bmf -W"
writer_zoo="dada_dbdisk -D $dir_zoo -k $key_zoo -W"

$echo "writer_raw is: $writer_raw"
$echo "writer_bmf is: $writer_bmf"
$echo "writer_zoo is: $writer_zoo\n"

$writer_raw & # should be unblock 
$writer_bmf & # should be unblock
$writer_zoo & # should be unblock

# now gpu pipeline
$echo "Starting process"
process="$pipeline_command -i $key_raw -o $key_bmf -z $key_zoo -n $nreader_raw -g 0" # need to add other configurations as well
$echo "process: $process\n"
$process & # should be unblock

# now udp2db
# udp2db is the only program which should be blocked (before cleanup),
# it mask ring buffer to tell other program to stop when it stops
# however with the current setup, dada_dbdisk does not stop with the signal
# we need to kill it in the end 
#$echo "Starting udp2db"
#udp2db="../build/udp/udp2db -k $key_raw -i 10.11.4.54 -p 12345 -f ../header/512MHz_1ant1pol_4096B.header -m 56400" # need to add more configuration
#udp2db="../build/udp/udp2db -k $key_raw -i 10.11.4.54 -p 12345 -f ../header/512MHz_beamform_4096B.header -m 56400" # need to add more configuration
#$echo "udp2db $udp2db\n"
#$udp2db
hdr_fname=$hdr_root/512MHz_beamform_4096B.header
nblock=1
nsecond=10
freq=1420

$echo "hdr_fname is: $hdr_fname"
$echo "nblock is:    $nblock"
$echo "nsecond is:   $nsecond"
$echo "freq is:      $freq\n"

# Start udp2db
$udp_command -f $hdr_fname -F $freq -n $nblock -N $nsecond -k $key_raw


sleep 1s # to wait all process finishes
$echo "done udp2db setup\n"

#$echo "Killing dada_dbdisk"
#writer_d="pkill -9 dada_dbdisk" #
#$echo "writer_d $writer_d\n"
#$writer_d # we should not block here 

#$echo "\nRemoving ring buffers"
#dada_raw_d="dada_db -k $key_raw -d"
#dada_bmf_d="dada_db -k $key_bmf -d"
#dada_zoo_d="dada_db -k $key_zoo -d"

#$echo "dada_raw_d is $dada_raw_d"
#$echo "dada_bmf_d is $dada_bmf_d"
#$echo "dada_zoo_d is $dada_zoo_d\n"

cleanup
#$dada_raw_d
#$dada_bmf_d
#$dada_zoo_d
