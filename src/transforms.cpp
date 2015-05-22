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

Transforms::Transforms(const Transforms& t):_w(t._w){}

//Transforms::Transforms(const mat& w,const mat& b){ RNN
Transforms::Transforms(const mat& w){
	_w=w;
}

Transforms::Transforms(size_t inputdim,size_t outputdim,float range){
	_w.resize(outputdim,inputdim);
	rand_init(_w,range); // uniform distribution
	_w/=sqrt((float)inputdim);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,myNnGen& ran){
	_w.resize(outputdim,inputdim);
	rand_norm(_w,ran);  // default variance = 0.1 , to change varance head to include/util.h
	_w/=sqrt((float)inputdim);
}
size_t Transforms::getInputDim()const{
	return _w.getCols();
}
size_t Transforms::getOutputDim()const{
	return _w.getRows();
}
mat Transforms::getWeight()const{return _w;}
void Transforms::print(ofstream& out){

	MatrixXf* h_data = _w.getData();
	out<<fixed<<setprecision(6);
    for(size_t t=0;t<_w.getRows();++t){
		//for(size_t k=0;k<_w.getCols()-1;++k)
		for(size_t k=0;k<_w.getCols();++k)
			out<<" "<<(*h_data)(t,k);
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
void Sigmoid::backPropagate(const mat& fin,const mat& delta, float rate,float regularization){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );

	rate/=(float)fin.getCols();
	_w-= (delta * ~fin) * rate + _w * regularization;

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

void Softmax::backPropagate(const mat& fin,const mat& delta,float rate, float regularization=0.0){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );

	rate/=(float)fin.getCols();
	_w-= (delta* ~fin) * rate + _w * regularization;
}
void Softmax::write(ofstream& out){
	out<<"<softmax> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
}

/*****************************************************/
/********************RECURSIVE************************/

Recursive::Recursive(const Recursive& s): Transforms(s),_step(s._step),_h(s._h){	
	_history.push_back(mat(s._h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
}

Recursive::Recursive(const mat& w,const mat& h,int step): Transforms(w),_step(step),_h(h){
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,float range,int step): Transforms(inputdim,outputdim,range),_step(step){
	_h.resize(outputdim,outputdim);
	rand_init(_h,range);
	_h/=sqrt((float)outputdim);
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,myNnGen& ran,int step): Transforms(inputdim,outputdim,ran),_step(step){
	_h.resize(outputdim,outputdim);
	rand_norm(_h,ran);
	_h/=sqrt((float)outputdim);
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
}
void Recursive::forward(mat& out,const mat& in){
	_input.push_back(in);	
	out=sigmoid(_w*in+_h*_history.back());
	_history.push_back(out);
}

void Recursive::backPropagate(const mat& fin,const mat& delta,float rate,float regularization){
	//_history.pop_back();//last history is useless
	//_history.pop_back();
	mat gra=delta;
	bptt(gra,rate,regularization);
}

void Recursive::write(ofstream& out){
	out<<"<recursive> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
	MatrixXf* hptr=_h.getData();
	out<<"<memory> "<<_h.getRows()<<" "<<_h.getCols()<<" "<<_step<<endl;
	out<<fixed<<setprecision(6);
	for(size_t t=0;t<_h.getRows();++t){
		for(size_t k=0;k<_h.getCols();++k)
			out<<" "<<(*hptr)(t,k);
		out<<endl;
	}
}

void Recursive::bptt(mat& gra,float rate,float regularization){
	int iidx=_input.size()-1,hidx=_history.size()-2;
	if(iidx>=0&&hidx>=0){
	for(int count=_step;count>0;count--){
		_wmem+=(gra* ~_input[iidx]) * rate + _w * regularization;
		_hmem+=(gra* ~_history[hidx]) * rate + _h * regularization;
		iidx--;hidx--;
		if(iidx<0||hidx<0)
			break;
		gra=_history.back()&((float)1.0-_history.back())&(~_h * gra);	
	}
	_input.pop_back();_history.pop_back();
	}
	else{cout<<"no input/history to update recursive weight\n";}
/*
	if(_graHis.size()==_step){
		_graHis.erase(_graHis.begin());
	}
	//for H mat  NOTE: can be a mat only, announce a vector for debug.
	int hsize=_history.size(); //hsize-1 = idx of latest memory
	mat graH=(delta* ~_history[hsize-2]) *rate + _h * regularization;
	//back propagation of unfold DNN
	int num=_graHis.size();
	for(int j=0;j<_graHis.size();++j){
		graH+=(_graHis[num-1-j]* ~ _history[hsize-3-j]) * rate + _h * regularization;
	}
	_graHis.push_back(delta); //delta back
	//update weight
	_h-= graH/(float)_graHis.size();
*/	
}

/**************************
	HUI BPTT
***************************/

/******************************************************/
