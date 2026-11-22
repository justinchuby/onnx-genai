#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void) {
 uint32_t state=0x7124abcd; size_t cells=0, values=0, different=0; double max_error=0;
 int ms[]={1,2,3,7,17,33,65}, ks[]={1,3,16,33,127,257}, ns[]={1,3,9,33,129};
 for(int mi=0;mi<7;mi++)for(int ki=0;ki<6;ki++)for(int ni=0;ni<5;ni++) {
  int m=ms[mi],k=ks[ki],n=ns[ni];
  float *a=malloc(m*k*4),*b=malloc(k*n*4),*bt=malloc(k*n*4),*nn=calloc(m*n,4),*nt=calloc(m*n,4);
  for(int i=0;i<m*k;i++){state=1664525*state+1013904223;a[i]=((int)(state>>8)-8388608)/16777216.f;}
  for(int i=0;i<k*n;i++){state=1664525*state+1013904223;b[i]=((int)(state>>8)-8388608)/16777216.f;}
  for(int p=0;p<k;p++)for(int j=0;j<n;j++)bt[j*k+p]=b[p*n+j];
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,k,b,n,0,nn,n);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1,a,k,bt,k,0,nt,n);
  for(int i=0;i<m;i++)for(int j=0;j<n;j++){
   double ref=0;for(int p=0;p<k;p++)ref+=(double)a[i*k+p]*bt[j*k+p];
   if(memcmp(nn+i*n+j,nt+i*n+j,4))different++;
   double err=fabs(nt[i*n+j]-ref);if(err>max_error)max_error=err;values++;
  }
  free(a);free(b);free(bt);free(nn);free(nt);cells++;
 }
 printf("{\"cells\":%zu,\"values\":%zu,\"nn_nt_bit_differences\":%zu,\"max_nt_abs_error_vs_f64\":%.9g}\n",cells,values,different,max_error);
}
