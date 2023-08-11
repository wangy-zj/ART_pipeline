#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "futils.h"
#include "dada_def.h"
#include "ascii_header.h"

#include "../include/dada_header.h"

#include <stdlib.h>
#include <stdio.h>

#include <string.h>

int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header){

  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  fileread(dada_header_file_name, dada_header_buffer, DADA_DEFAULT_HEADER_SIZE);
  read_dada_header(dada_header_buffer, dada_header);

  free(dada_header_buffer);

  return EXIT_SUCCESS;
}

int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header){


  if (ascii_header_get(dada_header_buffer, "MJD_START", "%lf", &dada_header->mjd_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "UTC_START", "%lf", &dada_header->utc_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting UTC_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "FILE_SIZE", "%lf", &dada_header->file_size) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting FILE_SIZE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NPKT", "%d", &dada_header->npkt) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NCHAN", "%d", &dada_header->pkt_nchan) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NTIME", "%d", &dada_header->pkt_ntime) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_NTIME, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_TSAMP", "%lf", &dada_header->pkt_tsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "FREQ", "%lf", &dada_header->freq) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting FREQ, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NAVERAGE", "%d", &dada_header->naverage) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NAVERAGE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NSTREAM", "%d", &dada_header->nstream) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NSTREAM, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

int write_dada_header_to_file(const dada_header_t dada_header, const char *dada_header_file_name){

  FILE *fp = fopen(dada_header_file_name, "w");
  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  sprintf(dada_header_buffer, "HDR_VERSION  1.0\nHDR_SIZE     4096\n");
  write_dada_header(dada_header, dada_header_buffer);
  fprintf(fp, "%s\n", dada_header_buffer);

  free(dada_header_buffer);
  fclose(fp);

  return EXIT_SUCCESS;
}

int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer){

  if (ascii_header_set(dada_header_buffer, "MJD_START", "%f", dada_header.mjd_start) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "UTC_START", "%f", dada_header.utc_start) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting UTC_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "FILE_SIZE", "%f", dada_header.file_size) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting FILE_SIZE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NPKT", "%d", dada_header.npkt) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NCHAN", "%d", dada_header.pkt_nchan) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NTIME", "%f", dada_header.pkt_ntime) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NTIME, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_TSAMP", "%f", dada_header.pkt_tsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NAVERAGE", "%d", dada_header.naverage) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NAVERAGE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
