#ifndef RNN_H
#define RNN_H
#include <vector>
#include <string>
#include "host_matrix.h"
#include "transforms.h"
#include "dataset.h"
using namespace std;

typedef host_matrix<float> mat;

enum Method{
	ALL, 
	BATCH, 
	ONE
};

enum Init{
	UNIFORM,
	NORMAL,
	RBM,
};

class RNN{
public:
	RNN();
	RNN(float learningRate,float momentum,float reg,float variance,Init init, const vector<size_t>& v, Method method, int step);
	~RNN();

	void train(Dataset& labeledData, size_t maxEpoch, float trainRation, float alpha);
	void predict(Dataset& testData, const string& outName);
	void getHiddenForward(mat& outputMat, const mat& inputMat);

	//void setDataset(Dataset* pData);
	void setLearningRate(float learningRate);
	void setMomentum(float momentum);
	void setReg(float reg);
	size_t getInputDimension();
	size_t getOutputDimension();
	size_t getNumLayers();
	void save(const string& fn);
	bool load(const string& fn);

private:
	void feedForward(const mat& inputMat, vector<mat>& fout);
	void backPropagate(float learningRate, float momentum,float regularization,const vector<mat>& fin,const mat& answer);
	//Dataset* _pData;
	float _learningRate;
	float _momentum;
	float _reg;
	Method _method;
	vector<Transforms*> _transforms;
	int _classNum;
	vector<float> _validateAccuracy;

};


#endif
