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
	//cout<<"forward:\n";
	//cout<<in.getRows()<<" "<<in.getCols()<<endl;
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

void Recursive::bptt(const mat& delta,float rate,float regularization){
	int num=(_counter<_step)? _counter : _step;
	vector<mat> outset(num);
	//for H mat  NOTE: can be a mat only, announce a vector for debug.
	vector<mat> deltaset(num);
	mat graW;
	mat graH;
	outset[0]=sigmoid(_w*_history[0]); //init=0;
	//feed forward  for unfold DNN
	for(size_t t=1;t<num;++t){
		outset[t]=sigmoid(_w*_history[t]+_h*outset[t-1]);
	}
	//back propagation of unfold DNN
	graW=(delta* ~_history[num-1]) *rate + _w * regularization;
	graH=(delta* ~outset[num-1]) *rate + _h * regularization;
	deltaset[num-1]=(outset[num-1]&((float)1.0-outset[num-1]) & (~_h * delta));
	for(int j=num-2;j>=0;j--){
		graW=(deltaset[j+1]* ~ _history[j]) * rate + _w *regularization;
		graH=(deltaset[j+1]* ~ outset[j]) * rate + _h * regularization;
		deltaset[j]=(outset[j]&((float)1.0-outset[j])& (~_h * deltaset[j+1]));
	}
	//update weight
	_w-= graW/(float)num;
	_h-= graH/(float)num;
}

/**************************
	HUI BPTT
***************************/

/******************************************************/
