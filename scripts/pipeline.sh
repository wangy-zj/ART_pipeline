#!/usr/bin/env bash

echo="echo -e"
trap ctrl_c INT

#程序停止时清除进程
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

# 设置各项路径
WORK_ROOT=/home/hero/code
project_root=$WORK_ROOT/ART_pipeline
hdr_root=$project_root/header
udp_command=$project_root/build/udp/udp2db
pipeline_command=$project_root/build/pipeline/pipeline_dada_amplitude_phase

$echo "Setting up ring buffers"

# 设置并创建环形缓冲区
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

# udp包的设置，用于计算缓冲区大小，要与udp.h中一致
pkt_dtsz=8000   #单个包数据大小
nstream=5       #包含数据流数目
npkt=1000       #单个block包含的包数目
naverage=100    #积分长度

numa=0
bufsz=$(( pkt_dtsz*nstream*npkt )) 
$echo "pkt_dtsz is:    $pkt_dtsz"
$echo "nstream_gpu is: $nstream"
$echo "npkt is:        $npkt"
$echo "naverage is:    $naverage"
$echo "numa is:        $numa\n"

# 计算设置相应缓冲区大小
bufsz_raw=$bufsz # buffer block size to hold raw data in bytes, change it to real value later
bufsz_amp=$((bufsz/naverage/2)) # buffer block size to hold amplitude data in bytes, change it to real value later
bufsz_phi=$bufsz_amp # buffer block size to hold phase data in bytes, change it to real value later

$echo "bufsz_raw is: $bufsz_raw"
$echo "bufsz_amp is: $bufsz_amp"
$echo "bufsz_phi is: $bufsz_phi"

# 创建缓冲区
$echo "Creating ring buffers"
dada_raw="dada_db -k $key_raw -b $bufsz_raw -n 100 -p -w -c $numa -r 2"  #assign memory from NUMA node  [default: all nodes]
dada_amp="dada_db -k $key_amp -b $bufsz_amp -n 100 -p -w -c $numa"
dada_phi="dada_db -k $key_phi -b $bufsz_phi -n 100 -p -w -c $numa"

$echo "dada_raw is: $dada_raw"
$echo "dada_amp is: $dada_amp"
$echo "dada_phi is: $dada_phi\n"

# 加入进程，程序结束后清除
$dada_raw & # should be unblock 
pids+=(`echo $! `)
keys+=(`echo $key_raw `)
$dada_amp & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_amp `)
$dada_phi & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $key_phi `)

sleep 1s # to wait all buffers are created 
$echo "created all ring buffers\n"

$echo "Setting file writers"
# 设置需要保存文件的路径
dir_raw=/home/hero/workspace/data/data_raw/ 
dir_amp=/home/hero/workspace/data/data_amp/ 
dir_phi=/home/hero/workspace/data/data_phi/ 

# 设定写入数据的地址和数据来源缓冲区
writer_raw="dada_dbdisk -D $dir_raw -k $key_raw -W" 
writer_amp="dada_dbdisk -D $dir_amp -k $key_amp -W"
writer_phi="dada_dbdisk -D $dir_phi -k $key_phi -W"

$echo "writer_raw is: $writer_raw"
$echo "writer_amp is: $writer_amp"
$echo "writer_phi is: $writer_phi\n"

# 加入进程，程序结束后清除
$writer_raw & # should be unblock 
pids+=(`echo $! `)
keys+=(`echo $write_raw `)
$writer_amp & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $write_amp `)
$writer_phi & # should be unblock
pids+=(`echo $! `)
keys+=(`echo $write_phi `)

# 设置数据处理程序
$echo "Starting process"
process="$pipeline_command -i $key_raw -a $key_amp -p $key_phi -n $nreader_raw -g 0" # need to add other configurations as well
$echo "process: $process\n"

# 设置udp2db,从端口读取udp数据包并写入输出缓冲区
$echo "Starting udp2db"
hdr_fname=$hdr_root/art_test.header
nblock=100     #每隔nblock报告状态
nsecond=50     #接收数据时常，单位s
freq=1420      #数据中心频率，暂时没用

# 开启处理和读取程序
$process & # should be unblock
$udp_command -f $hdr_fname -F $freq -n $nblock -N $nsecond -k $key_raw

sleep 1s # to wait all process finishes

$echo "done processing\n"

# 程序结束后清除各进程
cleanup
