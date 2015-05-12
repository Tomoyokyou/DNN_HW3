#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include "host_matrix.h"
#include "host_math.h"
#include "util.h"
#include "transforms.h"

using namespace std;

typedef host_matrix<float> mat;


/********************TRANSFORM**********************/

Transforms::Transforms(const Transforms& t):_w(t._w),_i(t._i),_pw(t._pw){}

Transforms::Transforms(const mat& w,const mat& b){
	assert(b.getRows()==1 || b.getCols()==1);
	size_t r=b.getRows(),c=b.getCols();
	if(r==1){r=c;c=1;}
	assert(w.getRows()==r);
	_w.resize(w.getRows(),w.getCols()+1);
	MatrixXf* wptr=_w.getData();
	wptr->block(0,0,w.getRows(),w.getCols())=*w.getData();
	wptr->block(w.getCols(),0,b.getRows(),1)=*b.getData();
	_pw.resize(_w.getRows(),_w.getCols(),0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,float range){
	_w.resize(outputdim,inputdim+1);
	rand_init(_w,range); // uniform distribution
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim+1,0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,myNnGen& ran){
	_w.resize(outputdim,inputdim+1);
	rand_norm(_w,ran);  // default variance = 0.2 , to change varance head to include/util.h
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim+1,0);
}
size_t Transforms::getInputDim()const{
	return _w.getCols();
}
size_t Transforms::getOutputDim()const{
	return _w.getRows();
}

void Transforms::print(ofstream& out){

	MatrixXf* h_data = _w.getData();
	out<<fixed<<setprecision(6);
    for(size_t t=0;t<_w.getRows();++t){
		for(size_t k=0;k<_w.getCols()-1;++k)
			out<<setw(9)<<(*h_data)(t,k);
		out<<endl;
	}
    out<<"<bias> "<<_w.getRows()<<endl;
    for(size_t t=0;t<_w.getRows();++t)
                out<<setw(9)<<(*h_data)(t,_w.getCols()-1);
	out << endl;
}

/****************************************************/
/********************SIGMOID*************************/
Sigmoid::Sigmoid(const Sigmoid& s): Transforms(s){
}
Sigmoid::Sigmoid(const mat& w, const mat& bias): Transforms(w,bias){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Sigmoid::forward(mat& out,const mat& in,bool train){
	mat _inp(in);
	pushOne(_inp);
	out=sigmoid(_w * _inp);
	if(train){
		_i=in;
	}
}
void Sigmoid::backPropagate(mat& out,const mat& delta, float rate,float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==_i.getCols()) );

	mat sigdiff=_i & ((float)1.0-_i);
	MatrixXf* optr=out.getData(),*dptr=delta.getData(),*sdptr=sigdiff.getData();

	MatrixXf wbias=_w.getData()->block(0,0,_w.getRows(),_w.getCols()-1);
	*optr = sdptr->cwiseProduct(wbias.transpose() * (*dptr));
	// update weight
	mat _inp(_i);
	pushOne(_inp);
	_pw= delta * ~_inp + _pw * momentum;
	rate/=(float)_i.getCols();
	_w -= _pw * rate;

}
void Sigmoid::write(ofstream& out){
	out<<"<sigmoid> "<<_w.getRows()<<" "<<_w.getCols()-1<<endl;
	print(out);
}

/*****************************************************/
/***********************SOFTMAX***********************/
Softmax::Softmax(const Softmax& s): Transforms(s){
}
Softmax::Softmax(const mat& w, const mat& bias):Transforms(w,bias){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Softmax::forward(mat& out,const mat& in,bool train){
	mat inp=in;
	pushOne(inp);
	mat z=_w * inp;
	out=softmax(z);
	if(train){
		_i=in;
	}
}

void Softmax::backPropagate(mat& out,const mat& delta,float rate, float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==_i.getCols()) );

	mat sigdiff=_i & ((float)1.0-_i);
	MatrixXf wbias=_w.getData()->block(0,0,_w.getRows(),_w.getCols()-1);
	MatrixXf *optr=out.getData(),*dptr=delta.getData(),*sdptr=sigdiff.getData();
	*optr=sdptr->cwiseProduct(wbias.transpose() * (*dptr));

	//update weight
	mat inp(_i);
	pushOne(inp);	
	_pw=delta * ~inp + _pw * momentum;
	rate/=(float)_i.getCols();
	_w-= _pw * rate;
}
void Softmax::write(ofstream& out){
	out<<"<softmax> "<<_w.getRows()<<" "<<_w.getCols()-1<<endl;
	print(out);
}
/*****************************************************/
