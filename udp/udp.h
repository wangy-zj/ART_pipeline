#ifndef _UDP_H
#define _UDP_H

#include <math.h>
#include <signal.h>

#include "ipcio.h"
#include "futils.h"
#include "ipcbuf.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "dada_hdu.h"
#include "multilog.h"

#include "dada_def.h"

#define MJD1970 40587.0f ///< MJD days in 1970.01.01
#define SECDAY  86400.0f ///< Seconds in a days

#define STR_BUFLEN 1024

// 设置udp包的大小，变化之后要修改！！
// define UDP packet size 
#define PKT_DTSZ   8000
#define PKT_HDRSZ  8
#define PKTSZ      (PKT_HDRSZ+PKT_DTSZ)

// define packet contents
// 1 channels per packet, each channel has 0.5 MHz,
// each sample has 2 bytes (1 for real and 1 for image)
#define PKT_NCHAN      1000
#define PKT_CHAN_WIDTH 0.031250f // MHz
#define PKT_SAMPSZ     8 // 8 bytes, 4 real and 4 imag
#define NAVERAGE       100

#define PKT_NTIME      (PKT_DTSZ/(PKT_NCHAN*PKT_SAMPSZ))  //number of time stamps per packet
#define TSAMP          (1/PKT_CHAN_WIDTH) // microseconds
#define PKT_DURATION   (PKT_NTIME*TSAMP)  // microseconds


// define network interfaces
//#define IP_SEND     "192.168.2.20"
//#define IP_RECV     "192.168.2.90"
#define IP_SEND     "10.11.153.134"

#define IP_RECV     "10.11.153.134"
//#define IP_SEND     "10.11.4.54"
//#define IP_RECV     "10.11.4.54"
#define PORT_SEND   10000 
#define PORT_RECV   60000
#define AD0         96

#define NSTREAM_UDP 1

const unsigned window_bytes = 64*pow(1000.0, 2); ///< 64 MB in bytes
const int reuse = 1; ///< reust address 

const struct timeval tout  = {0, 0};
const uint32_t nsleep_dead = 45;  ///< Estimated dead time of sleep
const int nsecond_report   = 1;   ///< Report traffic status every second

// udp 包头，包含counter和flag（AD序号），根据实际情况修改
typedef struct packet_header_t{
  //uint64_t flag : 0;
  uint64_t counter : 64; // for 512 MHz bandwidth, it could cover about 800 days???
}packet_header_t;

void catch_int(int sig_num);

uint64_t gregorian_calendar_to_jd(int y, int m, int d);
uint64_t gregorian_calendar_to_mjd(int y, int m, int d);

#endif
