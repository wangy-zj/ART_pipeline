# (Astronomical Read Out) ART_pipeline

## 模拟udp包的发送和接收

### 1.配置udp/udp.h文件

  udp包的数据大小：PKT_DTSZ  =  8000
  
  udp包头大小：PKT_HDRSZ  =  8
  
  udp包大小：PKTSZ  =    (PKT_HDRSZ+PKT_DTSZ)
  
  udp包含的通道数：PKT_NCHAN  =  1000
  
  每个通道带宽：PKT_CHAN_WIDTH  =  0.031250f(MHz)
  
  每个数据点的采样宽度：PKT_SAMPSZ  =  8(Bytes)  4 real and 4 imag
  
  每个数据包中的采样点数目：PKT_NTIME   =   (PKT_DTSZ/(PKT_NCHAN*PKT_SAMPSZ))
  
  数据采样间隔：TSAMP     =     (1/PKT_CHAN_WIDTH) 
  
  单个数据包时间：PKT_DURATION = (PKT_NTIME*TSAMP)

  数据发送ip： IP_SEND  =  "10.11.4.54"   需根据实际情况修改
  
  数据发送port： PORT_SEND  =  60000
  
  数据接收ip： IP_RECV  =  "10.11.4.54"  
  
  数据接收port： PORT_RECV  =  10000

  数据流数目（接收单元数目）： NSTREAM_UDP =  1
  
  幅度和相位积分长度： NAVERAGE = 100

### 2.启动udp2db

  修改udp/udp2db.sh文件
  
  WORK_ROOT：代码所在路径
  
  pkt_dtsz   与udp.h保持一致，用来计算ringbuffer大小
  
  nstream_gpu    与udp.h保持一致
  
  npkt    2048   ringbuffer中每个block的包数目，用来计算和配置block大小
  
  运行udp2db.sh

### 3.启动udpgen，进行数据流仿真

  编译之后运行：/build/udp/udpgen

## 从FPGA上接收udp包并进行处理和存储

  ### 1. 仅存储原始数据
  
  修改udp/udp2db.sh文件
  
  WORK_ROOT：代码所在路径
  
  pkt_dtsz   与udp.h保持一致，用来计算ringbuffer大小
  
  nstream_gpu    与udp.h保持一致
  
  npkt    2048   ringbuffer中每个block的包数目，用来计算和配置block大小
  
  存储文件大小修改：udp/udp2db.cpp 中设置nblocksave参数，之后重新编译。
  
  运行udp/udp2db.sh
  
  ### 2. 存储原始数据，在GPU上计算幅度和相位并存储
  
![ARP_pipeline](https://user-images.githubusercontent.com/110006648/234159154-f6134435-2ea4-4909-8ce7-7ab56f214e60.png)  

   配置scripts/pipeline.sh文件
   
   配置目录：project_root(代码路径)，hdr_root(dada header文件路径)，udp_command(udp包接收程序)，pipeline_command(GPU pipeline程序)
   
   配置ringbuffer: 设置相关参数用于创建输入和输出ringbuffer 
   
   $key_raw 输入ringbuffer key，$key_amp 输出幅度ringbuffer key，$key_phi 输出相位ringbuffer key
   
   配置存储目录：dir_raw(原始udp包数据目录)，dir_amp(幅度数据存储目录)，dir_phi(相位数据存储目录)
   
   配置pipeline启动行：$pipeline_command -i $key_raw -a $key_amp -p $key_phi -n $nreader_raw -g 0
   
   配置udp2db启动行：$udp_command -f $hdr_fname -F $freq -n $nblock -N $nsecond -k $key_raw
   
   运行scripts/pipeline.sh
   
