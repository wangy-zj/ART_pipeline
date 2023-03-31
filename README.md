# ART_pipeline
A-li read out

## 模拟udp包的发送和接收

1.配置udp.h文件：
  udp包的数据大小：PKT_DTSZ    8192
  udp包头大小：PKT_HDRSZ    8
  udp包含的通道数：PKT_NCHAN    1024
  每个通道带宽：PKT_CHAN_WIDTH    0.015625f(MHz)
  每个数据点的采样宽度：PKT_SAMPSZ    8(Bytes)  4 real and 4 imag

  数据发送ip： IP_SEND    "10.11.4.54"   需根据实际情况修改
  数据发送port： PORT_SEND    60000
  数据接收ip： IP_RECV    "10.11.4.54"  
  数据接收port： PORT_RECV    10000

  数据流数目（接收单元数目）： NSTREAM_UDP    1

2.启动udp2db:
  修改udp/udp2db.sh文件
  WORK_ROOT：代码所在路径
  pkt_dtsz  8192  与udp.h保持一致，用来计算ringbuffer大小
  nstream_gpu   1  
  npkt    2048   每个ringbuffer的包数目，用来计算ringbuffer大小
  
  运行udp2db.sh

3.启动udpgen
  编译之后运行：/build/udp/udpgen
