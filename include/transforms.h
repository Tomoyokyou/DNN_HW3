#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include "host_matrix.h"
#include <fstream>
#include <string>
#include <vector>
#include "mynngen.h"

using namespace std;

typedef host_matrix<float> mat;

enum ACT{
	SIGMOID,
	SOFTMAX,
	RECURSIVE
};

class Transforms{
	public:
		Transforms(const Transforms& t);
		virtual void forward(mat& out,const mat& in) = 0;
		virtual void backPropagate(const mat& fin,const mat& delta,float rate,float regularization,float momentum) = 0;
		virtual void write(ofstream& out)=0;
		virtual ACT getAct()const=0;
		virtual void resetCounter(float rate)=0;
		size_t getInputDim()const;
		size_t getOutputDim()const;
		mat getWeight()const;
		mat getGradient()const;
		bool isreset()const{return _counter==0;}
		mat multWeightInv(const mat& a){return mat(~_w * a);}
	protected:
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim,float range);
		Transforms(size_t inputdim, size_t outputdim,myNnGen& ran);
		void print(ofstream& out);
		mat _w;
		mat _pw; // momentum
		int _counter;
	private:
};


class Sigmoid : public Transforms{
	public:
	Sigmoid(const Sigmoid& s);
	Sigmoid(const mat& w);
	Sigmoid(size_t inputdim, size_t outputdim,float range=1.0);
	Sigmoid(size_t inputdim, size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin, const mat& delta, float rate,float regularization,float momentum);
	virtual void write(ofstream& out);
	virtual void resetCounter(float rate){
				_counter=0;
				_pw.resize(_w.getRows(),_w.getCols(),0);
			};
	virtual ACT getAct()const {return SIGMOID;};
	private:
};

class Softmax : public Transforms{
	public:
	Softmax(const Softmax& s);
	Softmax(const mat& w);
	Softmax(size_t inputdim,size_t outputdim,float range=1.0);
	Softmax(size_t inputdim,size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin, const mat& delta, float rate,float regularization,float momentum);
	virtual void write(ofstream& out);
	virtual ACT getAct()const{return SOFTMAX;};
	virtual void resetCounter(float rate){
			MatrixXf* temp=_graMem.getData();
			if(_counter>1){*temp/=(float)_counter;}
			*temp=(temp->array()>50).select(50,*temp);
			*temp=(temp->array()<-50).select(50,*temp);
			_w-=(_graMem)*rate;
			_graMem.resize(_w.getRows(),_w.getCols(),0);	
			_pw.resize(_w.getRows(),_w.getCols(),0);
			_counter=0;	
			}
	void accGra(const mat& fin,const mat& delta,float rate,float regularization,float momentum);
	private:
	mat _graMem;
};

class Recursive : public Transforms{
	public:
	Recursive(const Recursive& s);
	Recursive(const mat& w,const mat& h,int step);
	Recursive(size_t inputdim,size_t outputdim,float range=1.0,int step=1);
	Recursive(size_t inputdim,size_t outputdim,myNnGen& ran,int step=1);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin,const mat& delta,float rate,float regularization,float momentum);
	virtual void write(ofstream& out);
	virtual ACT getAct()const{return RECURSIVE;};

	virtual void resetCounter(float rate){
				MatrixXf* wptr=_wmem.getData(),*hptr=_hmem.getData();
				if(_counter>1){
					 (*wptr)/=(float)_counter;(*hptr)/=(float)_counter;
				}
				*wptr=(wptr->array()>50).select(50,*wptr);
				*hptr=(hptr->array()<-50).select(-50,*hptr);
				_w-=(_wmem)*rate ;
				_h-=(_hmem)*rate;
				_history.clear();
				_input.clear();
				//_inputPtr.clear();///
				_history.push_back(mat(_h.getRows(),1,0));
				_wmem.resize(_w.getRows(),_w.getCols(),0);
				_hmem.resize(_h.getRows(),_h.getCols(),0);
				_pw.resize(_w.getRows(),_w.getCols(),0);
				_pwh.resize(_h.getRows(),_h.getCols(),0);
				_counter=0;
				}
	void forwardFirst(mat& out,mat* in);
	int getStep()const {return _step;}
	
	private:
		void bptt(mat& gra,float rate,float regularization,float momentum);
		vector<mat> _history;
		vector<mat> _input;
		//
		//vector<mat*> _inputPtr;
		//
		int _step;
		mat _h;
		mat _pwh;
		mat _wmem;
		mat _hmem;
};

#endif
