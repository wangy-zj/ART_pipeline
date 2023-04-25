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

#$echo "This is a pipeline for ART at NAOC\n"

# There will be three ring buffers
# The first one is used to receive raw data
# The second one is used to receive amplitude
# The last one is used to receive phase

# The first ring buffer need two readers as we need to read its data to GPU and write its data to disk
# The second ring buffer has only one reader
# The last ring buffer has only one reader as well

# setup command lines
WORK_ROOT=/home/hero/code
project_root=$WORK_ROOT/ART_pipeline
hdr_root=$project_root/header
udp_command=$project_root/build/udp/udp2db
pipeline_command=$project_root/build/pipeline/pipeline_dada_amplitude_phase

$echo "Setting up ring buffers"

key_raw=a000
key_amp=b000
key_phi=c000

$echo "key_raw is: $key_raw"
$echo "key_amp is: $key_amp"
$echo "key_phi is: $key_phi\n"

nreader_raw=128
nreader_amp=128
nreader_phi=128

$echo "nreader_raw is: $nreader_raw"
$echo "nreader_amp is: $nreader_amp"
$echo "nreader_phi is: $nreader_phi\n"

# udp包设置，变化之后要修改！！！
pkt_dtsz=8000
nstream=1
npkt=100
naverage=100

numa=0
bufsz=$(( pkt_dtsz*nstream*npkt ))
$echo "pkt_dtsz is:    $pkt_dtsz"
$echo "nstream_gpu is: $nstream"
$echo "npkt is:        $npkt"
$echo "naverage is:    $naverage"
$echo "numa is:        $numa\n"

# it will be more flexible if we put equation here to calculate buffer size with some basic configrations
bufsz_raw=$bufsz # buffer block size to hold raw data in bytes, change it to real value later
bufsz_amp=$((bufsz/naverage/2)) # buffer block size to hold amplitude data in bytes, change it to real value later
bufsz_phi=$bufsz_amp # buffer block size to hold phase data in bytes, change it to real value later

$echo "bufsz_raw is: $bufsz_raw"
$echo "bufsz_amp is: $bufsz_amp"
$echo "bufsz_phi is: $bufsz_phi"

numa=0 # numa node to use 
$echo "numa is: $numa\n"

$echo "Creating ring buffers"
dada_raw="dada_db -k $key_raw -b $bufsz_raw -n 20 -p -w -c $numa -r 2"  #assign memory from NUMA node  [default: all nodes]
dada_amp="dada_db -k $key_amp -b $bufsz_amp -n 20 -p -w -c $numa"
dada_phi="dada_db -k $key_phi -b $bufsz_phi -n 20 -p -w -c $numa"

$echo "dada_raw is: $dada_raw"
$echo "dada_amp is: $dada_amp"
$echo "dada_phi is: $dada_phi\n"

$dada_raw & # should be unblock 
pids+=(`echo $! `)
keys+=(`echo $key_raw `)
$dada_amp & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_amp `)
$dada_phi & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_phi `)

#$echo "existing pids are ${pids[@]}"

sleep 1s # to wait all buffers are created 
$echo "created all ring buffers\n"

$echo "Setting file writers"
# different type of files should go to different directories
dir_raw=/home/hero/data/data_raw/ # change it to real value later
dir_amp=/home/hero/data/data_amp/ # change it to real value later
dir_phi=/home/hero/data/data_phi/ # change it to real value later

$echo "dir_raw is: $dir_raw"
$echo "dir_amp is: $dir_amp"
$echo "dir_phi is: $dir_phi\n"

# 设定写入数据的地址和数据来源ringbuffer
writer_raw="dada_dbdisk -D $dir_raw -k $key_raw -W" 
writer_amp="dada_dbdisk -D $dir_amp -k $key_amp -W"
writer_phi="dada_dbdisk -D $dir_phi -k $key_phi -W"

$echo "writer_raw is: $writer_raw"
$echo "writer_amp is: $writer_amp"
$echo "writer_phi is: $writer_phi\n"

$writer_raw & # should be unblock 
pids+=(`echo $! `)
keys+=(`echo $write_raw `)
$writer_amp & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $write_amp `)
$writer_phi & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $write_phi `)

# now gpu pipeline
$echo "Starting process"
process="$pipeline_command -i $key_raw -a $key_amp -p $key_phi -n $nreader_raw -g 0" # need to add other configurations as well
$echo "process: $process\n"

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
hdr_fname=$hdr_root/art_test.header
nblock=100
nsecond=10
freq=1420

$echo "hdr_fname is: $hdr_fname"
$echo "nblock is:    $nblock"
$echo "nsecond is:   $nsecond"
$echo "freq is:      $freq\n"

# Start pipeline and udp2db
$process & # should be unblock
$udp_command -f $hdr_fname -F $freq -n $nblock -N $nsecond -k $key_raw

sleep 1s # to wait all process finishes

$echo "done processing\n"

cleanup
