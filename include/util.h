#ifndef UTIL_H
#define UTIL_H

#include "mynngen.h"
#include "host_matrix.h"
using namespace std;

typedef host_matrix<float> mat;
extern myNnGen gn;

extern void rand_init(mat& w,float range=1);
extern void rand_norm(mat& w,myNnGen& ran=gn);

extern void pushOne(mat& in);
extern void parseDim(string str, vector<size_t>& dim);
/*
extern void getBias(mat& out,const mat& w);
extern void replaceBias(mat& w,const mat& bias);
*/
#endif
