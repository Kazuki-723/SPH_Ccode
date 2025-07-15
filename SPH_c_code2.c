#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
  #include <direct.h>
#endif

#define M_PI 3.14159265358979323846

// シミュレーションパラメータ
const double ps             = 0.001;
const double Lx             = 1.0, Ly = 0.5;
const double dt             = 5e-5;
const int    steps          = 10000;
const double rho0           = 1000.0;
const double k              = 100.0;
const double visc           = 1e-3;
const double gacc           = 9.8;
const double eps            = 1e-6;
const int    step_per_frame = 10;
// 中央壁定義
const double barrier_x_min = 0.48;
const double barrier_x_max = 0.52;
const double hole_y_min    = 0.20;
const double hole_y_max    = 0.30;


/// 画面サイズ
const int IMG_W = 800, IMG_H = 400;

/// 動的変数
int nxg, nyg, N;
int Nf, Nb;
double m, h;
double *x, *y, *vxh, *vyh, *dens, *ax, *ay;
int    *head, *linked;

/// SPHカーネル
inline double W2D(double r){
    double q = r/h;
    double sigma = 10.0 / (7.0 * M_PI * h * h);
    if(q < 1.0)      return sigma * (1 - 1.5 * q * q + 0.75 * q * q * q);
    else if(q < 2.0) return sigma * 0.25 * pow(2.0 - q, 3);
    return 0.0;
}

inline void gradW2D(double dx,double dy,double r,double *rx,double *ry){
    if(r < eps || r >= 2*h){
        *rx = *ry = 0.0;
        return;
    }
    double q     = r/h;
    double sigma = 10.0/(7.0*M_PI*h*h);
    double dw;
    if(q < 1.0)      dw = sigma * (-3.0*q + 2.25*q*q) / h;
    else              dw = -sigma * 0.75 * pow(2.0 - q, 2) / h;
    *rx = dw * dx / r;
    *ry = dw * dy / r;
}

// セルリスト構築
void build_grid(){
    int G = nxg * nyg;
    for(int i = 0; i < G; i++) head[i] = -1;
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        int cx = fmin((int)(x[i]/Lx * nxg), nxg-1);
        int cy = fmin((int)(y[i]/Ly * nyg), nyg-1);
        int ci = cy * nxg + cx;
        #pragma omp critical
        {
            linked[i] = head[ci];
            head[ci]  = i;
        }
    }
}

// 壁粒子を数えるユーティリティ
int count_barrier(){
    int Ny = (int)(Ly/ps);
    int cnt = 0;
    for(int j=0; j<Ny; j++){
        double yy = (j+0.5)*ps;
        if(yy>hole_y_min && yy<hole_y_max) continue;
        cnt++;
    }
    return cnt * 2;  // 左側と右側
}

// 密度計算
void calc_density(){
    build_grid();
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        double xi = x[i], yi = y[i];
        double sum = 0.0;
        int cx = fmin((int)(xi/Lx * nxg), nxg-1);
        int cy = fmin((int)(yi/Ly * nyg), nyg-1);
        for(int dyc = -1; dyc <= 1; dyc++){
            for(int dxc = -1; dxc <= 1; dxc++){
                int cx2 = cx + dxc, cy2 = cy + dyc;
                if(cx2 < 0 || cx2 >= nxg || cy2 < 0 || cy2 >= nyg) 
                    continue;
                int j = head[cy2 * nxg + cx2];
                while(j != -1){
                    double dx = xi - x[j], dy = yi - y[j];
                    sum += m * W2D(hypot(dx, dy));
                    j = linked[j];
                }
            }
        }
        dens[i] = sum > eps ? sum : eps;
    }
}

// 加速度計算
void calc_acc(){
    build_grid();  
    static double *p = NULL;
    if(!p) p = malloc(sizeof(double) * N);

    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        double pi = k * (dens[i] - rho0);
        p[i] = pi > 0.0 ? pi : 0.0;
        ax[i] = ay[i] = 0.0;
    }

    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        double xi = x[i], yi = y[i];
        int cx = fmin((int)(xi/Lx * nxg), nxg-1);
        int cy = fmin((int)(yi/Ly * nyg), nyg-1);
        for(int dyc = -1; dyc <= 1; dyc++){
            for(int dxc = -1; dxc <= 1; dxc++){
                int cx2 = cx + dxc, cy2 = cy + dyc;
                if(cx2 < 0 || cx2 >= nxg || cy2 < 0 || cy2 >= nyg) 
                    continue;
                int j = head[cy2 * nxg + cx2];
                while(j != -1){
                    if(j > i){
                        double dx = xi - x[j], dy = yi - y[j];
                        double r = hypot(dx, dy), rx, ry;
                        gradW2D(dx, dy, r, &rx, &ry);

                        double denom = dens[i]*dens[i] + dens[j]*dens[j] + eps;
                        double aij   = -(p[i] + p[j]) * m / denom;
                        double aix   = aij * rx, aiy = aij * ry;

                        #pragma omp atomic
                        ax[i] += aix;
                        #pragma omp atomic
                        ay[i] += aiy;
                        #pragma omp atomic
                        ax[j] -= aix;
                        #pragma omp atomic
                        ay[j] -= aiy;

                        if(r < 2*h){
                            double lap  = 20.0/(7.0*M_PI*h*h)*(2.0 - r/h);
                            double dvx  = vxh[j] - vxh[i];
                            double dvy  = vyh[j] - vyh[i];
                            double coef = visc * m / (dens[j] + eps);
                            double ax2  = coef * dvx * lap;
                            double ay2  = coef * dvy * lap;

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
            }
        }
        ay[i] -= gacc;
    }
}


// XSPH 速度平滑化
void xsph_correction(){
    const double xsph_eps = 0.1;

    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        double xi = x[i], yi = y[i];
        double vxc = 0.0, vyc = 0.0;
        int cx = fmin((int)(xi/Lx * nxg), nxg-1);
        int cy = fmin((int)(yi/Ly * nyg), nyg-1);

        for(int dyc = -1; dyc <= 1; dyc++){
            for(int dxc = -1; dxc <= 1; dxc++){
                int cx2 = cx + dxc, cy2 = cy + dyc;
                if(cx2 < 0 || cx2 >= nxg || cy2 < 0 || cy2 >= nyg) 
                    continue;
                int j = head[cy2 * nxg + cx2];
                while(j != -1){
                    if(j != i){
                        double dx = xi - x[j], dy = yi - y[j];
                        double r  = hypot(dx, dy);
                        if(r < 2*h){
                            double w = W2D(r);
                            vxc += (vxh[j] - vxh[i]) * w * m / dens[j];
                            vyc += (vyh[j] - vyh[i]) * w * m / dens[j];
                        }
                    }
                    j = linked[j];
                }
            }
        }

        vxh[i] += xsph_eps * vxc;
        vyh[i] += xsph_eps * vyc;
    }
}

// PPM 出力
void write_ppm(int frame){
    char fname[64];
    sprintf(fname, "temp/frame%04d.ppm", frame);

    FILE *fp = fopen(fname, "wb");
    if(!fp){
        fprintf(stderr, "[write_ppm] fopen(%s) failed: %s\n",
                fname, strerror(errno));
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", IMG_W, IMG_H);

    size_t sz = 3 * (size_t)IMG_W * IMG_H;
    unsigned char *img = calloc(sz, 1);
    memset(img, 255, sz);

    const int radius = 2;
    for(int p = 0; p < N; p++){
        int ix = (int)(x[p]/Lx * IMG_W);
        int iy = (int)(y[p]/Ly * IMG_H);
        iy   = IMG_H - 1 - iy;

        for(int dy = -radius; dy <= radius; dy++){
            int yy = iy + dy;
            if(yy<0 || yy>=IMG_H) continue;
            for(int dx = -radius; dx <= radius; dx++){
                int xx = ix + dx;
                if(xx<0 || xx>=IMG_W) continue;
                int idx = 3*(yy*IMG_W + xx);
                img[idx+0] = 70;
                img[idx+1] = 130;
                img[idx+2] = 180;
            }
        }
    }

    fwrite(img, 1, sz, fp);
    free(img);
    fclose(fp);
}

/// メイン関数
int main(){
  #ifdef _WIN32
    if(_mkdir("temp") != 0 && errno != EEXIST){
        perror("mkdir temp failed");
        return 1;
    }
  #else
    if(mkdir("temp", 0755) != 0 && errno != EEXIST){
        perror("mkdir temp failed");
        return 1;
    }
  #endif
    // 壁粒子数を先にカウント
    Nb = count_barrier();

    // 流体領域（仕切りの左側）の粒子数
    int nx1 = (int)(barrier_x_min/ps), ny1 = (int)(Ly/ps);
    Nf = nx1 * ny1;

    // 総粒子数
    N  = Nf + Nb;

    // SPHパラメタ
    h  = 1.3 * ps;
    m  = rho0 * ps * ps;
    nxg= (int)(Lx/h) + 1;
    nyg= (int)(Ly/h) + 1;

    // 配列確保（Nf+Nb 粒子分）
    x      = malloc(sizeof(double)*N);
    y      = malloc(sizeof(double)*N);
    vxh    = calloc(N, sizeof(double));
    vyh    = calloc(N, sizeof(double));
    dens   = malloc(sizeof(double)*N);
    ax     = calloc(N, sizeof(double));
    ay     = calloc(N, sizeof(double));
    head   = malloc(sizeof(int)*nxg*nyg);
    linked = malloc(sizeof(int)*N);

    // ① 流体粒子の初期配置
    int idx = 0;
    for(int j=0; j<ny1; j++){
      for(int i=0; i<nx1; i++){
        x[idx] = (i+0.5)*ps;
        y[idx] = (j+0.5)*ps;
        idx++;
      }
    }
    // idx == Nf

    // ② 境界粒子（中央仕切り）を追加
    int Ny = (int)(Ly/ps);
    for(int side=0; side<2; side++){
      double bx = (side==0 ? barrier_x_min : barrier_x_max);
      for(int j=0; j<Ny; j++){
        double yy = (j+0.5)*ps;
        if(yy>hole_y_min && yy<hole_y_max) continue;  // 穴はスキップ
        x[idx] = bx;
        y[idx] = yy;
        // velocity, acceleration は 0 のまま
        idx++;
      }
    }
    // 最終チェック
    if(idx != N){
      fprintf(stderr,"ERROR: idx(%d) != N(%d)\n", idx, N);
      return 1;
    }

    // 初期密度・加速・Leap-Frog 初期化
    calc_density();
    calc_acc();
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        vxh[i] += 0.5 * dt * ax[i];
        vyh[i] += 0.5 * dt * ay[i];
    }

    int frame = 0;
    for(int t = 0; t < steps; t++){
        #pragma omp parallel for
        for(int i=0; i<Nf; i++){
            // 通常の位置更新
            x[i] += vxh[i] * dt;
            y[i] += vyh[i] * dt;

            // 左右・上下の容器境界反射
            if(x[i] < 0){ x[i]=0;      vxh[i]*=-0.5; }
            if(x[i] > Lx){ x[i]=Lx;    vxh[i]*=-0.5; }
            if(y[i] < 0){ y[i]=0;      vyh[i]*=-0.5; }
            if(y[i] > Ly){ y[i]=Ly;    vyh[i]*=-0.5; }

            // 中央仕切りの壁処理
            // x が仕切り範囲内、かつ穴の外なら反射
            if(x[i] > barrier_x_min && x[i] < barrier_x_max){
                if(y[i] < hole_y_min || y[i] > hole_y_max){
                    // どちらの面に衝突したかで位置を押し戻し
                    if(vxh[i] > 0) x[i] = barrier_x_min;
                    else           x[i] = barrier_x_max;
                    vxh[i] *= -0.5;
                }
            }
        }


        // SPH 演算
        calc_density();
        calc_acc();
        #pragma omp parallel for
        for(int i = 0; i < Nf; i++){
            vxh[i] += dt * ax[i];
            vyh[i] += dt * ay[i];
        }

        // XSPH 速度平滑化
        xsph_correction();

        // フレーム出力
        if((t + 1) % step_per_frame == 0){
            write_ppm(frame++);
            if(frame % 100 == 0)
                fprintf(stderr, "Frame %d / %d\n",
                        frame, steps/step_per_frame);
        }
    }

    system("ffmpeg -y -f image2 -framerate 60 -i temp/frame%04d.ppm -loop 0 hole_with_boundary.gif");
    printf("Saved -> hole_with_boundary.gif\n");
    return 0;
}