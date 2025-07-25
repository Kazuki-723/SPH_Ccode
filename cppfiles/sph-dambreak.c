#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
  #include <direct.h>
#endif
#include "sph.h"

int main(){
  // 1) シミュレーションパラメタ設定
  ps               = 0.001;
  Lx               = 1.0;  Ly = 0.5;
  dt               = 1e-4;
  steps            = 10000;
  step_per_frame   = 10;
  rho0             = 1000.0;
  k                = 100.0;
  visc             = 100.0;
  gacc             = 9.8;
  eps              = 1e-6;
  barrier_x_min    = 0.48;
  barrier_x_max    = 0.52;
  hole_y_min       = 0.20;
  hole_y_max       = 0.30;

  // 2) 格子＆粒子数設定
  h   = 1.3 * ps;
  m   = rho0 * ps * ps;
  nxg = (int)(Lx/h) + 1;
  nyg = (int)(Ly/h) + 1;

  // 左側流体だけ初期配置
  int nx1 = (int)(barrier_x_min/ps), ny1 = (int)(Ly/ps);
  N = nx1 * ny1;

  // 3) メモリ確保
  x      = malloc(sizeof(double)*N);
  y      = malloc(sizeof(double)*N);
  vxh    = calloc(N, sizeof(double));
  vyh    = calloc(N, sizeof(double));
  dens   = malloc(sizeof(double)*N);
  ax     = calloc(N, sizeof(double));
  ay     = calloc(N, sizeof(double));
  head   = malloc(sizeof(int)*nxg*nyg);
  linked = malloc(sizeof(int)*N);

  // 4) 初期位置割当
  for(int j=0, idx=0; j<ny1; j++){
    for(int i=0; i<nx1; i++, idx++){
      x[idx] = (i+0.5)*ps;
      y[idx] = (j+0.5)*ps;
    }
  }

  // 5) Leap‐Frog 初期化
  calc_density();
  calc_acc();
  #pragma omp parallel for
  for(int i=0;i<N;i++){
    vxh[i] += 0.5*dt*ax[i];
    vyh[i] += 0.5*dt*ay[i];
  }

  // 6) temp フォルダ作成
  #ifdef _WIN32
    if(_mkdir("temp")!=0 && errno!=EEXIST){ perror("mkdir"); return 1; }
  #else
    if(mkdir("temp",0755)!=0 && errno!=EEXIST){ perror("mkdir"); return 1; }
  #endif

  // 7) 時間発展ループ
  int frame = 0;
  for(int t=0; t<steps; t++){
    #pragma omp parallel for
    for(int i=0;i<N;i++){
      // 位置更新＋外壁反射
      x[i] += vxh[i]*dt;
      y[i] += vyh[i]*dt;
      if(x[i]<0){ x[i]=0;      vxh[i]*=-0.5; }
      if(x[i]>Lx){ x[i]=Lx;    vxh[i]*=-0.5; }
      if(y[i]<0){ y[i]=0;      vyh[i]*=-0.5; }
      if(y[i]>Ly){ y[i]=Ly;    vyh[i]*=-0.5; }
    }

    calc_density();
    calc_acc();
    #pragma omp parallel for
    for(int i=0;i<N;i++){
      vxh[i] += dt*ax[i];
      vyh[i] += dt*ay[i];
    }

    xsph_correction();

    if((t+1)%step_per_frame==0){
      write_ppm("temp", frame++);
      if(frame%100==0) fprintf(stderr,"Frame %d\n",frame);
    }
  }

  // 8) GIF 化
  system("ffmpeg -y -f image2 -framerate 60 -i temp/frame%04d.ppm -loop 0 dambreak.gif");
  printf("Saved -> hole.gif\n");
  return 0;
}
