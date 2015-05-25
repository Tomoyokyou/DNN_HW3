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
	RNN(float learningRate,float momentum,float reg,float variance,Init init, const vector<size_t>& v, Method method, int step, Dataset& data);
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
	float calAcc();
private:
	void feedForward(const mat& inputMat, vector<mat>& fout, int classLabel);
	void backPropagate(const vector<pair<vector<mat>,vector<mat>>>& fromForward,const vector<int>& classLabel);
	//Dataset* _pData;
	float _learningRate;
	float _momentum;
	float _reg;
	Method _method;
	vector<Transforms*> _transforms;
	vector<Transforms*> _outSoftmax;
	vector<float> _validateAccuracy;
	void status()const;
};


#endif
