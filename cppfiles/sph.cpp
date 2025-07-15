#include "sph.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <errno.h>

#define M_PI 3.14159265358979323846

// グローバル変数実体定義
int    nxg, nyg, N;
double m, h;
double ps, Lx, Ly, dt;
double rho0, k, visc, gacc, eps;
double barrier_x_min, barrier_x_max, hole_y_min, hole_y_max;
int    steps, step_per_frame;

double *x, *y, *vxh, *vyh, *dens, *ax, *ay;
int    *head, *linked;

inline double W2D(double r){
    double q = r/h;
    double sigma = 10.0/(7.0*M_PI*h*h);
    if(q < 1.0)      return sigma*(1-1.5*q*q+0.75*q*q*q);
    else if(q < 2.0) return sigma*0.25*pow(2.0-q,3);
    return 0.0;
}

inline void gradW2D(double dx,double dy,double r,double *rx,double *ry){
    if(r < eps || r >= 2*h){ *rx = *ry = 0.0; return; }
    double q = r/h, sigma = 10.0/(7.0*M_PI*h*h), dw;
    if(q < 1.0) dw = sigma*(-3*q + 2.25*q*q)/h;
    else        dw = -sigma*0.75*pow(2.0-q,2)/h;
    *rx = dw * dx/r;
    *ry = dw * dy/r;
}

void build_grid(){
    int G = nxg*nyg;
    for(int i=0;i<G;i++) head[i] = -1;
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        int cx = (int)(x[i]/Lx * nxg);
        if(cx<0) cx=0; if(cx>=nxg) cx=nxg-1;
        int cy = (int)(y[i]/Ly * nyg);
        if(cy<0) cy=0; if(cy>=nyg) cy=nyg-1;
        int ci = cy*nxg + cx;
        #pragma omp critical
        {
            linked[i] = head[ci];
            head[ci]  = i;
        }
    }
}

void calc_density(){
    build_grid();
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        double xi = x[i], yi = y[i], sum = 0.0;
        int cx = (int)(xi/Lx * nxg), cy = (int)(yi/Ly * nyg);
        for(int dyc=-1; dyc<=1; dyc++){
        for(int dxc=-1; dxc<=1; dxc++){
            int cx2 = cx+dxc, cy2 = cy+dyc;
            if(cx2<0||cx2>=nxg||cy2<0||cy2>=nyg) continue;
            int j = head[cy2*nxg + cx2];
            while(j!=-1){
                double dx = xi - x[j], dy = yi - y[j];
                sum += m * W2D(hypot(dx,dy));
                j = linked[j];
            }
        }}
        dens[i] = sum>eps ? sum : eps;
    }
}

void calc_acc(){
    build_grid();
    static double *p = NULL;
    if(!p) p = (double*)malloc(sizeof(double)*N);

    #pragma omp parallel for
    for(int i=0;i<N;i++){
        double pi = k*(dens[i] - rho0);
        p[i] = pi>0 ? pi : 0;
        ax[i] = ay[i] = 0.0;
    }

    #pragma omp parallel for
    for(int i=0;i<N;i++){
        double xi=x[i], yi=y[i];
        int cx=(int)(xi/Lx*nxg), cy=(int)(yi/Ly*nyg);
        for(int dyc=-1; dyc<=1; dyc++){
        for(int dxc=-1; dxc<=1; dxc++){
            int cx2 = cx+dxc, cy2 = cy+dyc;
            if(cx2<0||cx2>=nxg||cy2<0||cy2>=nyg) continue;
            int j = head[cy2*nxg + cx2];
            while(j!=-1){
                if(j>i){
                    double dx=xi-x[j], dy=yi-y[j];
                    double r=hypot(dx,dy), rx,ry;
                    gradW2D(dx,dy,r,&rx,&ry);
                    double denom = dens[i]*dens[i] + dens[j]*dens[j] + eps;
                    double aij = -(p[i]+p[j])*m/denom;
                    double aix = aij * rx, aiy = aij * ry;

                    #pragma omp atomic
                    ax[i] += aix;
                    #pragma omp atomic
                    ay[i] += aiy;
                    #pragma omp atomic
                    ax[j] -= aix;
                    #pragma omp atomic
                    ay[j] -= aiy;

                    if(r<2*h){
                        double lap   = 20.0/(7.0*M_PI*h*h)*(2.0-r/h);
                        double dvx   = vxh[j]-vxh[i];
                        double dvy   = vyh[j]-vyh[i];
                        double coef  = visc * m / (dens[j] + eps);
                        double ax2   = coef * dvx * lap;
                        double ay2   = coef * dvy * lap;

                        #pragma omp atomic
                        ax[i] += ax2;
                        #pragma omp atomic
                        ay[i] += ay2;
                        #pragma omp atomic
                        ax[j] -= ax2;
                        #pragma omp atomic
                        ay[j] -= ay2;
                    }
                }
                j = linked[j];
            }
        }}
        ay[i] -= gacc;
    }
}

void xsph_correction(){
    build_grid();
    const double xsph_eps = 0.1;
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        double xi=x[i], yi=y[i], vxc=0.0, vyc=0.0;
        int cx=(int)(xi/Lx*nxg), cy=(int)(yi/Ly*nyg);
        for(int dyc=-1; dyc<=1; dyc++){
        for(int dxc=-1; dxc<=1; dxc++){
            int cx2=cx+dxc, cy2=cy+dyc;
            if(cx2<0||cx2>=nxg||cy2<0||cy2>=nyg) continue;
            int j = head[cy2*nxg+cx2];
            while(j!=-1){
                if(j!=i){
                    double dx=xi-x[j], dy=yi-y[j];
                    double r = hypot(dx,dy);
                    if(r<2*h){
                        double w = W2D(r);
                        vxc += (vxh[j]-vxh[i])*w*m/dens[j];
                        vyc += (vyh[j]-vyh[i])*w*m/dens[j];
                    }
                }
                j = linked[j];
            }
        }}
        vxh[i] += xsph_eps * vxc;
        vyh[i] += xsph_eps * vyc;
    }
}

void write_ppm(const char *dir, int frame){
    char fname[256];
    sprintf(fname, "%s/frame%04d.ppm", dir, frame);
    FILE *fp = fopen(fname, "wb");
    if(!fp){ perror(fname); return; }
    fprintf(fp, "P6\n%d %d\n255\n", IMG_W, IMG_H);
    size_t sz = 3*(size_t)IMG_W*IMG_H;
    unsigned char *img = (unsigned char*)malloc(sz);
    memset(img, 255, sz);

    const int radius = 2;
    for(int i=0;i<N;i++){
        int ix = (int)(x[i]/Lx*IMG_W);
        int iy = IMG_H-1 - (int)(y[i]/Ly*IMG_H);
        for(int dy=-radius;dy<=radius;dy++){
        for(int dx=-radius;dx<=radius;dx++){
            int xx = ix+dx, yy = iy+dy;
            if(xx<0||xx>=IMG_W||yy<0||yy>=IMG_H) continue;
            int idx = 3*(yy*IMG_W + xx);
            img[idx+0]=70; img[idx+1]=130; img[idx+2]=180;
        }}
    }
    fwrite(img,1,sz,fp);
    free(img);
    fclose(fp);
}