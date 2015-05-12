#include "host_matrix.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;
typedef host_matrix<float> hmat;
void pushOne(hmat& in){
	hmat tmp(~in);
	float next[tmp.size()+tmp.getRows()],*ww=tmp.getData();
	for(size_t t=0;t<tmp.size();++t)
		next[t]=ww[t];
	for(size_t t=0;t<tmp.getRows();++t)
		next[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	ww=tmp.getData();
	for(size_t t=0;t<tmp.size();++t)
		ww[t]=next[t];
	in=~tmp;
}
void substractMaxPerCol(hmat& z){
	float* h_data=z.getData();
	size_t temp_max_idx;
	float max;
	for(size_t t=0;t<z.getCols();++t){
		temp_max_idx=z.getRows()*t;
		for(size_t k=1;k<z.getRows();++k){
			if(h_data[temp_max_idx]<h_data[t*z.getRows()+k]){
				temp_max_idx=t*z.getRows()+k;
			}
		}
		max=h_data[temp_max_idx];
		for(size_t l=0;l<z.getRows();++l)
			h_data[t*z.getRows()+l]-=max;
	}
}
void init(hmat& w){
	float* data=w.getData();
	for(size_t t=0;t<w.size();++t)
		data[t]=rand()/(float)RAND_MAX;
}

int main(){
	srand(time(NULL));
	hmat A(5,6,2);
	hmat B;
	B=~A;
	A.print();
	pushOne(A);
	cout<<endl;
	A.print();
	cout<<endl;

	init(B);
	B.print();
	cout<<endl;
	substractMaxPerCol(B);
	cout<<"after substract"<<endl;
	B.print();
	cout<<endl;

	return 0;
}

