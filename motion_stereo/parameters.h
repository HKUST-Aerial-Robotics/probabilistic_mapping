#ifndef _parameters_h
#define _parameters_h

#include <cstring>

#define FLOAT_EPS 1e-5

const int WIDTH = 640 ;
const int HEIGHT = 480 ;
//const int WIDTH = 752/2 ;
//const int HEIGHT = 480/2 ;

extern int CASE;

const int DEP_CNT = 64 ;
const float SAD_WEIGHT = 1.0f;

const int MARGIN = 2;
const int PATCH_SIZE = (2 * MARGIN + 1) * (2 * MARGIN + 1);

extern float pi1;
extern float pi2;
extern float tau_so;
extern float sgm_q1;
extern float sgm_q2;
extern float var_scale;

const float DEP_INF_1 = 2000.0f;  // non-overlap
const float DEP_INF = 1000.0f;
const float COST_INF = 0.0f;

#endif
