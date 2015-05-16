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

Transforms::Transforms(const Transforms& t):_w(t._w),_pw(t._pw){}

//Transforms::Transforms(const mat& w,const mat& b){ RNN
Transforms::Transforms(const mat& w){
	_w=w;
	_pw.resize(_w.getRows(),_w.getCols(),0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,float range){
	_w.resize(outputdim,inputdim);
	rand_init(_w,range); // uniform distribution
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim,0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,myNnGen& ran){
	_w.resize(outputdim,inputdim);
	rand_norm(_w,ran);  // default variance = 0.1 , to change varance head to include/util.h
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim);
}
size_t Transforms::getInputDim()const{
	return _w.getCols();
}
size_t Transforms::getOutputDim()const{
	return _w.getRows();
}
mat Transforms::getWeight()const{return _w;}
mat Transforms::getGradient()const{return _pw;}
void Transforms::print(ofstream& out){

	MatrixXf* h_data = _w.getData();
	out<<fixed<<setprecision(6);
    for(size_t t=0;t<_w.getRows();++t){
		//for(size_t k=0;k<_w.getCols()-1;++k)
		for(size_t k=0;k<_w.getCols();++k)
			out<<setw(9)<<(*h_data)(t,k);
		out<<endl;
	}
}

/****************************************************/
/********************SIGMOID*************************/
Sigmoid::Sigmoid(const Sigmoid& s): Transforms(s){
}
Sigmoid::Sigmoid(const mat& w): Transforms(w){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Sigmoid::forward(mat& out,const mat& in){
	out=sigmoid(_w * in);
}
void Sigmoid::backPropagate(const mat& fin,const mat& delta, float rate,float momentum,float regularization){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );

	//mat sigdiff=_i & ((float)1.0-_i);
	//mat sigdiff =  fin & ((float)1.0-fin);
	//MatrixXf* optr=out.getData(),*dptr=delta.getData(),*sdptr=sigdiff.getData();

	/*MatrixXf wbias=_w.getData()->block(0,0,_w.getRows(),_w.getCols()-1);
	*optr = sdptr->cwiseProduct(wbias.transpose() * (*dptr));*/
	//MatrixXf* wptr=_w.getData();
	//*optr = sdptr->cwiseProduct(wptr->transpose() * (*dptr));
	// update weight
	/*mat _inp(_i);
	pushOne(_inp);
	_pw= delta * ~_inp + _pw * momentum;*/
	rate/=(float)fin.getCols();
	_pw= (delta * ~fin) * rate + _w * regularization + _pw * momentum;
	_w -= _pw;

}
void Sigmoid::write(ofstream& out){
	out<<"<sigmoid> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
}

/*****************************************************/
/***********************SOFTMAX***********************/
Softmax::Softmax(const Softmax& s): Transforms(s){
}
//Softmax::Softmax(const mat& w, const mat& bias):Transforms(w,bias){
Softmax::Softmax(const mat& w):Transforms(w){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Softmax::forward(mat& out,const mat& in){
	out=softmax(_w * in);
}

void Softmax::backPropagate(const mat& fin,const mat& delta,float rate, float momentum, float regularization=0.0){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );

	//mat sigdiff=fin & ((float)1.0-fin);
	//MatrixXf wbias=_w.getData()->block(0,0,_w.getRows(),_w.getCols()-1);
	//MatrixXf *optr=out.getData(),*dptr=delta.getData(),*sdptr=sigdiff.getData();
	//*optr=sdptr->cwiseProduct(wbias.transpose() * (*dptr));
	//MatrixXf* wptr=_w.getData();
	//*optr=sdptr->cwiseProduct(wptr->transpose() * (*dptr));

	//update weight
	rate/=(float)fin.getCols();
	_pw= (delta* ~fin) * rate + _w * regularization + _pw * momentum;
	_w-= _pw;
}
void Softmax::write(ofstream& out){
	out<<"<softmax> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
}
/*****************************************************/
/********************RECURSIVE************************/

Recursive::Recursive(const Recursive& s): Transforms(s),_step(s._step),_counter(0),_h(s._h){	
	_mem.resize(s._w.getRows(),1,0);
}

Recursive::Recursive(const mat& w,const mat& h,int step): Transforms(w),_step(step),_h(h),_counter(0){
	_mem.resize(w.getRows(),1,0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,float range,int step): Transforms(inputdim,outputdim,range),_step(step),_counter(0){
	_h.resize(outputdim,outputdim);
	rand_init(_h,range);
	_h/=sqrt((float)outputdim);
	_mem.resize(outputdim,1,0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,myNnGen& ran,int step): Transforms(inputdim,outputdim,ran),_step(step),_counter(0){
	_h.resize(outputdim,outputdim);
	rand_norm(_h,ran);
	_h/=sqrt((float)outputdim);
	_mem.resize(outputdim,1,0);
}
void Recursive::forward(mat& out,const mat& in){
	out=sigmoid(_w*in+_h*_mem);
	_mem=out;
	if(_counter<_step){
		_history.push_back(in);
		_counter++;
	}
	else{
		_history.erase(_history.begin());
		_history.push_back(in);
	}
}

void Recursive::backPropagate(const mat& fin,const mat& delta,float rate,float momentum,float regularization){
	bptt(delta,rate,regularization);
}

void Recursive::write(ofstream& out){
	out<<"<recursive> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
	MatrixXf* hptr=_h.getData();
	out<<"<memory> "<<_h.getRows()<<" "<<_h.getCols()<<endl;
	out<<fixed<<setprecision(6);
	for(size_t t=0;t<_h.getRows();++t){
		for(size_t k=0;k<_h.getCols();++k)
			out<<setw(9)<<(*hptr)(t,k);
		out<<endl;
	}
}

void bptt(const mat& delta,float rate,float regularization){
	
}

/**************************
	HUI BPTT
***************************/
/*void bptt(size_t step,const vector<mat>& x,const mat& xw,const mat& mem,const mat& ans){
	assert(step==x.size());  // # of feat. must match 
	vector<mat> feat;
	//   UNFOLD COMPLETE

	mat memInit(mem.getCols(),1,0);  // initialize memory with 0
	mat lastout;
	vector<mat> outList(step);  // output of activations except Softmax
	vector<mat> memList(step);
	outList[0]=sigmoid(mem*memInit + xw * x[0]); // this may explode !!!!
	for(size_t t=1;t<step;++t){
		outList[t]=sigmoid(mem*outList[t-1]+xw*x[t]);
	}

	//TODO

}	
*/

/******************************************************/
