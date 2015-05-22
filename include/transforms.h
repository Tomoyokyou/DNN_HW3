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
		virtual void backPropagate(const mat& fin,const mat& delta,float rate,float regularization) = 0;
		virtual void write(ofstream& out)=0;
		virtual ACT getAct()const=0;
		size_t getInputDim()const;
		size_t getOutputDim()const;
		mat getWeight()const;
		mat getGradient()const;
	protected:
		//Transforms(const mat& w,const mat& b); RNN
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim,float range=1.0);
		Transforms(size_t inputdim, size_t outputdim,myNnGen& ran);
		void print(ofstream& out);
		mat _w;
	private:
};


class Sigmoid : public Transforms{
	public:
	Sigmoid(const Sigmoid& s);
	//Sigmoid(const mat& w, const mat& bias);
	Sigmoid(const mat& w);
	Sigmoid(size_t inputdim, size_t outputdim,float range=1.0);
	Sigmoid(size_t inputdim, size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin, const mat& delta, float rate,float regularization);
	virtual void write(ofstream& out);
	virtual ACT getAct()const {return SIGMOID;};
	private:
};

class Softmax : public Transforms{
	public:
	Softmax(const Softmax& s);
	//Softmax(const mat& w, const mat& bias);
	Softmax(const mat& w);
	Softmax(size_t inputdim,size_t outputdim,float range=1.0);
	Softmax(size_t inputdim,size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin, const mat& delta, float rate,float regularization);
	virtual void write(ofstream& out);
	virtual ACT getAct()const{return SOFTMAX;};
	private:
};

class Recursive : public Transforms{
	public:
	Recursive(const Recursive& s);
	Recursive(const mat& w,const mat& h,int step);
	Recursive(size_t inputdim,size_t outputdim,float range=1.0,int step=1);
	Recursive(size_t inputdim,size_t outputdim,myNnGen& ran,int step=1);
	virtual void forward(mat& out,const mat& in);
	virtual void backPropagate(const mat& fin,const mat& delta,float rate,float regularization);
	virtual void write(ofstream& out);
	virtual ACT getAct()const{return RECURSIVE;};

	void resetCounter(){
				_w-=(_counter==0)? _wmem : _wmem/(float)_counter;
				_h-=(_counter==0)? _hmem : _hmem/(float)_counter;
					_history.clear();_input.clear();
				_history.push_back(mat(_h.getRows(),1,0));
				_wmem.resize(_w.getRows(),_w.getCols(),0);
				_hmem.resize(_h.getRows(),_h.getCols(),0);
				_counter=0;
				}
	int getStep()const {return _step;}
	
	private:
		void bptt(mat& gra,float rate,float regularization);
		vector<mat> _history;
		vector<mat> _input;
		int _counter;
		int _step;
		mat _h;
		mat _wmem;
		mat _hmem;
};

#endif
