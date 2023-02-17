#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
       case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

/*void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}
*/

/* vectorized activate array*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    long gvl;
    float beta=0.1,ALPHA=0, ONE=1;
   switch(a){
        case LEAKY:
               for(i = 0; i < n; )
               {
                gvl = __builtin_epi_vsetvl(((long)n - (long)i), __epi_e32, __epi_m1);
                __epi_2xf32 XERO = __builtin_epi_vfmv_v_f_2xf32(ALPHA, gvl);
                __epi_2xf32 BETA = __builtin_epi_vfmv_v_f_2xf32(beta, gvl);
                __epi_2xf32 x_vec = __builtin_epi_vload_2xf32(&x[i+0], gvl);
                __epi_2xi1 mask = __builtin_epi_vmflt_2xf32(x_vec, XERO, gvl);
                x_vec =  __builtin_epi_vfmul_2xf32_mask(x_vec, x_vec,BETA, mask,gvl);
                __builtin_epi_vstore_2xf32(&x[i+0], x_vec,   gvl);
                //__builtin_epi_vstore_2xf32(&x[i+0], alpha,    gvl);
                i += gvl;

                }
                break;
        case LINEAR:
               for(i = 0; i < n; )
               {
                gvl = __builtin_epi_vsetvl(((long)n - (long)i), __epi_e32, __epi_m1);
                __epi_2xf32 x_vec = __builtin_epi_vload_2xf32(&x[i+0], gvl);
                __builtin_epi_vstore_2xf32(&x[i+0], x_vec,   gvl);
                i += gvl;

                }
                break;
	case RELU:
	        for(i=0;i<n;)
		{
                gvl = __builtin_epi_vsetvl(((long)n - (long)i), __epi_e32, __epi_m1);
                __epi_2xf32 XERO = __builtin_epi_vfmv_v_f_2xf32(ALPHA, gvl);
                __epi_2xf32 one = __builtin_epi_vfmv_v_f_2xf32(ONE, gvl);
                __epi_2xf32 x_vec = __builtin_epi_vload_2xf32(&x[i+0], gvl);
                __epi_2xi1 mask = __builtin_epi_vmflt_2xf32(x_vec, XERO, gvl);
                x_vec =  __builtin_epi_vfmul_2xf32_mask(x_vec, x_vec,XERO, mask,gvl);
                __builtin_epi_vstore_2xf32(&x[i+0], x_vec,   gvl);
		i += gvl;
		//x[i] = x[i]*(x[i]>0);
		}
		break;
         default:
               for(i = 0; i < n; ++i){
                   x[i] = activate(x[i], a);
                }
                break;
      }
}
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

