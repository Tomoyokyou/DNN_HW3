#include <string>
#include <vector>
#include <string>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <mynngen.h>
#include "host_matrix.h"
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXf;
using namespace Eigen;

typedef host_matrix<float> mat;

myNnGen gn(0,1);

void rand_init(mat& w,float range){
	MatrixXf* h_data=w.getData();
	*h_data=range*(MatrixXf::Random(w.getRows(),w.getCols())-MatrixXf::Constant(w.getRows(),w.getCols(),0.5));
}
void rand_norm(mat& w,myNnGen& ran){
	MatrixXf* h_data=w.getData();
	for(size_t x=0;x<w.getCols();++x){
		for(size_t y=0;y<w.getRows();++y)
			(*h_data)(y,x)=ran();
	}
}
void pushOne(mat& in){
	MatrixXf* tmp=in.getData();
	MatrixXf next=MatrixXf::Ones(in.getRows()+1,in.getCols());
	int r=tmp->rows(),c=tmp->cols();
	next.block(0,0,r,c)=(*tmp).block(0,0,r,c);
	*tmp=next;
}
void parseDim(string str,vector<size_t>& dim){
	size_t begin=str.find_first_not_of(' '),end;
	string hold;
	while(begin!=string::npos){
		end=str.find_first_of('-',begin);
		if(end==string::npos)
			hold=str.substr(begin);
		else
			hold=str.substr(begin,end-begin);
		if(!hold.empty())
			dim.push_back(atoi(hold.c_str()));
		begin=str.find_first_not_of('-',end);
	}
}
