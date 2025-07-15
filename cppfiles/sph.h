#ifndef SPH_H
#define SPH_H

#ifdef __cplusplus
extern "C" {
#endif

// grid / particle
extern int    nxg, nyg, N;
extern double m, h;

// simulation domain & parameters (to be defined in main.c)
extern double ps, Lx, Ly, dt;
extern double rho0, k, visc, gacc, eps;
extern double barrier_x_min, barrier_x_max, hole_y_min, hole_y_max;
extern int    steps, step_per_frame;

// particle arrays
extern double *x, *y, *vxh, *vyh, *dens, *ax, *ay;
extern int    *head, *linked;

// rendering constants
#define IMG_W 800
#define IMG_H 400

// core SPH routines
void build_grid();
void calc_density();
void calc_acc();
void xsph_correction();

// output PPM under directory 'dir' with frame number
void write_ppm(const char *dir, int frame);

#ifdef __cplusplus
}
#endif

#endif  // SPH_H
