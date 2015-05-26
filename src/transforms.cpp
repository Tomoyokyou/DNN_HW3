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

Transforms::Transforms(const Transforms& t):_w(t._w){
	_counter=0;
	_pw.resize(_w.getRows(),_w.getCols(),0);
}

//Transforms::Transforms(const mat& w,const mat& b){ RNN
Transforms::Transforms(const mat& w){
	_w=w;
	_pw.resize(_w.getRows(),_w.getCols(),0);
	_counter=0;
}

Transforms::Transforms(size_t inputdim,size_t outputdim,float range){
	_w.resize(outputdim,inputdim);
	rand_init(_w,range); // uniform distribution
	_w/=sqrt((float)inputdim);
	_pw.resize(_w.getRows(),_w.getCols(),0);
	_counter=0;
}

Transforms::Transforms(size_t inputdim,size_t outputdim,myNnGen& ran){
	_w.resize(outputdim,inputdim);
	rand_norm(_w,ran);  // default variance = 0.1 , to change varance head to include/util.h
	_w/=sqrt((float)inputdim);
	_pw.resize(_w.getRows(),_w.getCols(),0);
	_counter=0;
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
void Sigmoid::backPropagate(const mat& fin,const mat& delta, float rate,float regularization,float momentum){
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
	_graMem.resize(_w.getRows(),_w.getCols(),0);	
}
Softmax::Softmax(const mat& w):Transforms(w){
	_graMem.resize(_w.getRows(),_w.getCols(),0);	
}
Softmax::Softmax(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
	_graMem.resize(_w.getRows(),_w.getCols(),0);	
}
Softmax::Softmax(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
	_graMem.resize(_w.getRows(),_w.getCols(),0);	
}
void Softmax::forward(mat& out,const mat& in){
	out=softmax(_w * in);
}

void Softmax::backPropagate(const mat& fin,const mat& delta,float rate, float regularization,float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );
	_pw=delta*~fin + _pw*momentum;
	_w-= (delta* ~fin) + _w * regularization;
}
void Softmax::write(ofstream& out){
	out<<"<softmax> "<<_w.getRows()<<" "<<_w.getCols()<<endl;
	print(out);
}

void Softmax::accGra(const mat& fin,const mat& delta,float rate,float regularization,float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==fin.getCols()) );
	//mat gra1=(delta*~fin) + _w * regularization + momentum*_pw;
	//_pw=gra1;//momentum _pw=delta*~inp + _pw *momentum;
	_pw=delta*~fin + _pw * momentum;
	_graMem+=_pw;
	_counter++;
}

/*****************************************************/
/********************RECURSIVE************************/

Recursive::Recursive(const Recursive& s): Transforms(s),_step(s._step),_h(s._h){	
	_history.push_back(mat(s._h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
	_pwh.resize(_h.getRows(),_h.getCols(),0);
}

Recursive::Recursive(const mat& w,const mat& h,int step): Transforms(w),_step(step),_h(h){
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
	_pwh.resize(_h.getRows(),_h.getCols(),0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,float range,int step): Transforms(inputdim,outputdim,range),_step(step){
	_h.resize(outputdim,outputdim);
	rand_init(_h,range);
	_h/=sqrt((float)outputdim);
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
	_pwh.resize(_h.getRows(),_h.getCols(),0);
}
Recursive::Recursive(size_t inputdim,size_t outputdim,myNnGen& ran,int step): Transforms(inputdim,outputdim,ran),_step(step){
	_h.resize(outputdim,outputdim);
	rand_norm(_h,ran);
	_h/=sqrt((float)outputdim);
	_history.push_back(mat(_h.getRows(),1,0));
	_wmem.resize(_w.getRows(),_w.getCols(),0);
	_hmem.resize(_h.getRows(),_h.getCols(),0);
	_pwh.resize(_h.getRows(),_h.getCols(),0);
}
void Recursive::forward(mat& out,const mat& in){
	//_input.push_back(in);	
	out=sigmoid(_w*in+_h*_history.back());
	_history.push_back(out);
}
void Recursive::forwardFirst(mat& out,mat* in){
	_input.push_back(in);
	out=sigmoid(_w*(*in)+_h*_history.back());
	_history.push_back(out);
}
void Recursive::backPropagate(const mat& fin,const mat& delta,float rate,float regularization,float momentum){
	mat gra=delta;
	bptt(gra,rate,regularization,momentum);
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

void Recursive::bptt(mat& gra,float rate,float regularization,float momentum){
	int iidx=_input.size()-1,hidx=_history.size()-2;
	if(iidx>=0&&hidx>=0){
	for(int count=0;count<_step;count++){
		_pw=gra*~(*_input[iidx])+_pw*momentum;
		_pwh=gra*~_history[hidx]+_pwh*momentum;
		_wmem+= _pw;
		_hmem+= _pwh;
		_counter++;
		iidx--;hidx--;
		if(iidx<0||hidx<0)
			break;
		gra=_history.at(hidx+1)&((float)1.0-_history.at(hidx+1))&(~_h * gra);	
	}
	_input.pop_back();_history.pop_back();
	}
	else{cout<<"no input/history to update recursive weight\n";}
}

/******************************************************/
