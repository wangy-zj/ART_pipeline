#ifndef __DADA_HEADER_H
#define __DADA_HEADER_H

#define DADA_STRLEN 1024

#ifdef __cplusplus
extern "C" {
#endif

#include "inttypes.h"

  typedef struct dada_header_t{
    double mjd_start;
    char utc_start;
    int file_size;
    int npkt;
    int pkt_nchan;
    int pkt_ntime;
    double pkt_tsamp;
    double freq;
    int naverage;
    int nstream;
  }dada_header_t;

  int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header);

  int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer);

  int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header);

  int write_dada_header_to_file(const dada_header_t dada_header,const char *dada_header_file_name);

#ifdef __cplusplus
}
#endif

#endif
