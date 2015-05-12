#include "host_matrix.h"
#include <Eigen/Dense>
#include <cmath>

using namespace std;
using Eigen::MatrixXf;
using namespace Eigen;

template<class T>
host_matrix<T> sigmoid(const host_matrix<T>& src){
	host_matrix<T> temp(src);
	MatrixXf* tmp=temp.getData();
	*tmp=(-1*(*src.getData())).array().exp();
	MatrixXf id=MatrixXf::Ones(tmp->rows(),tmp->cols());
	*tmp+=id;
	*tmp=id.array() / tmp->array();
	return temp;
}

template<class T>
host_matrix<T> softmax(const host_matrix<T>& src){
	host_matrix<T> temp(src);
	MatrixXf* tmpptr=temp.getData();
	VectorXf colMax=tmpptr->colwise().maxCoeff();
	tmpptr->rowwise() -= colMax.transpose();
	*tmpptr=tmpptr->array().exp();
	VectorXf sumProb=tmpptr->colwise().sum();
	MatrixXf sumP=MatrixXf::Zero(tmpptr->rows(),tmpptr->cols());
	sumP.rowwise() += sumProb.transpose();
	(*tmpptr).array()/= sumP.array();
	return temp;
}

