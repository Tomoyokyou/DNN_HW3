#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include "host_matrix.h"
#include <fstream>
#include <string>
#include "mynngen.h"

using namespace std;

typedef host_matrix<float> mat;

class Transforms{
	public:
		Transforms(const Transforms& t);
		virtual void forward(mat& out,const mat& in,bool train) = 0;
		virtual void backPropagate(mat& out,const mat& delta,float rate,float momentum,float regularization) = 0;
		virtual void write(ofstream& out)=0;
		size_t getInputDim()const;
		size_t getOutputDim()const;
	protected:
		//Transforms(const mat& w,const mat& b); RNN
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim,float range=1.0);
		Transforms(size_t inputdim, size_t outputdim,myNnGen& ran);
		void print(ofstream& out);
		mat _w;
		mat _i;
		mat _pw;
	private:
};


class Sigmoid : public Transforms{
	public:
	Sigmoid(const Sigmoid& s);
	//Sigmoid(const mat& w, const mat& bias);
	Sigmoid(const mat& w);
	Sigmoid(size_t inputdim, size_t outputdim,float range=1.0);
	Sigmoid(size_t inputdim, size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in,bool train);
	virtual void backPropagate(mat& out, const mat& delta, float rate,float momentum,float regularization);
	virtual void write(ofstream& out);
//	Sigmoid& operator = (const Sigmoid s);

	private:
};

class Softmax : public Transforms{
	public:
	Softmax(const Softmax& s);
	//Softmax(const mat& w, const mat& bias);
	Softmax(const mat& w);
	Softmax(size_t inputdim,size_t outputdim,float range=1.0);
	Softmax(size_t inputdim,size_t outputdim,myNnGen& ran);
	virtual void forward(mat& out,const mat& in,bool train);
	virtual void backPropagate(mat& out, const mat& delta, float rate,float momentum,float regularization);
	virtual void write(ofstream& out);
	private:
};

#endif
