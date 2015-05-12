#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H
#include <iostream>
#include <cassert>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>


using namespace std;
using Eigen::MatrixXf;
using namespace Eigen;
template<typename T>
class host_matrix{
public:
	class Transpose {
	public:
		host_matrix<T> operator + (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m._rows,_m.cols);
			host_geam(_m,rhs,result,(T)1.0,(T)1.0,true,false);
			return result;
		}
		host_matrix<T> operator - (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m._rows,_m.cols);
			host_geam(_m,rhs,result,(T)1.0,(T)-1.0,true,false);
			return result;
		}

		host_matrix<T> operator * (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m.cols,rhs._cols);
			host_gemm(_m,rhs,result,(T)1.0,(T)0.0,true,false);
			return result;
		}
		host_matrix<T> operator * (const Transpose rhs) {
			host_matrix<T> result(_m.cols,rhs._m._rows);
			host_gemm(_m,rhs,result,(T)1.0,(T)0.0,true,true);
			return result;
		}
		void print(int precision=5){
			cout<<fixed<<setprecision(precision);
			for(size_t y=0;y<_m._cols;++y){
				for(size_t x=0;x<_m._rows;++x)
					cout<<setw(precision+4)<<(*_m._d)(x,y);
				cout<<endl;
			}
		}
		Transpose(const host_matrix<T>& m): _m(m){}
		const host_matrix<T> _m;
	};

	public:
	
	host_matrix();
	host_matrix(size_t r, size_t c);
	host_matrix(size_t r, size_t c, T value);
	host_matrix(T* data, size_t r,size_t c);
	host_matrix(const MatrixXf& src);
	host_matrix(const host_matrix<T>& src);
	host_matrix(const Transpose& src);

	~host_matrix();

	host_matrix<T>& operator += (T val);
	host_matrix<T> operator + (T val) const;
	
	host_matrix<T>& operator += (const host_matrix<T>& rhs);
	host_matrix<T> operator + (const host_matrix<T>& rhs) const;

	host_matrix<T>& operator += (const Transpose& rhs);
	host_matrix<T> operator + (const Transpose& rhs) const;
	
	host_matrix<T>& operator -= (T val);
	host_matrix<T> operator - (T val) const;
	
	host_matrix<T>& operator -= (const host_matrix<T>& rhs);
	host_matrix<T> operator - (const host_matrix<T>& rhs);

	host_matrix<T>& operator -= (const Transpose& rhs);
	host_matrix<T> operator - (const Transpose& rhs) const;

	host_matrix<T>& operator /= (T val);
	host_matrix<T> operator / (T val) const;
	
	host_matrix<T>& operator *= (T val);
	host_matrix<T> operator * (T val) const;
	
	host_matrix<T>& operator *= (const host_matrix<T>& rhs);
	host_matrix<T> operator * (const host_matrix<T>& ths) const;

	host_matrix<T>& operator *= (const Transpose& rhs);
	host_matrix<T> operator * (const Transpose& rhs) const;
	
	host_matrix<T>& operator &= (const host_matrix<T>& rhs);
	host_matrix<T> operator & (const host_matrix<T>& rhs) const;

	host_matrix<T>& operator = (const host_matrix<T>& rhs);

	Transpose operator ~ () const;

	void resize(size_t r,size_t c);
	void resize(size_t r,size_t c,T val);
	void print(int precision=5) const;
	
	void fillwith(T val);
	size_t size() const {return _d->rows()*_d->cols();}
	size_t getRows() const {return _d->rows();}
	size_t getCols() const {return _d->cols();}

	MatrixXf* getData() const {return _d;}
		
private:
	MatrixXf* _d;
};

template<class T>
host_matrix<T> operator + (T val, const host_matrix<T>& m){
	return m + (T) val;
}

template<class T>
host_matrix<T> operator - (T val, const host_matrix<T>& m){
	return (m - (T) val) * -1.0;
}

template<class T>
host_matrix<T> operator * (T val, const host_matrix<T>& m){
	return m * (T) val;
}

template<class T>
host_matrix<T> operator / (T val, const host_matrix<T>& m){
	return m * (T)1/(T)val;
}


template<class T>
host_matrix<T>::host_matrix(){
	_d=new MatrixXf;
}
template<class T>
host_matrix<T>::host_matrix(size_t r,size_t c){
	_d=new MatrixXf(r,c);
}
template<class T>
host_matrix<T>::host_matrix(size_t r,size_t c,T value){
	_d=new MatrixXf(r,c);
	_d->setConstant(r,c,value);
}
template<class T>
host_matrix<T>::host_matrix(T* data,size_t r,size_t c){
	_d=new MatrixXf(r,c);
	for(size_t x=0;x<c;++x){
		for(size_t y=0;y<r;++y)
			(*_d)(y,x)=data[x*r+y];
	}
}
template<class T>
host_matrix<T>::host_matrix(const MatrixXf& src){
	_d=new MatrixXf;
	_d->resize(src.rows(),src.cols());
	(*_d)=src;
}
template<class T>
host_matrix<T>::host_matrix(const host_matrix<T>& src){
	_d=new MatrixXf(src._d->rows(),src._d->cols());
	*_d=*src._d;
}
template<class T>
host_matrix<T>::host_matrix(const Transpose& src){
	size_t _rows=src._m._d->cols(),_cols=src._m._d->rows();
	_d=new MatrixXf(_rows,_cols);
	*_d=src._m._d->transpose();
}

template<class T>
host_matrix<T>::~host_matrix(){
	if(_d!=NULL)
		delete _d;
}
template<class T>
host_matrix<T>& host_matrix<T>::operator += (T val){
	MatrixXf id=MatrixXf::Ones(_d->rows(),_d->cols());
	id*=val;
	*_d+=id;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (T val) const{
	host_matrix<T> temp(*this);
	MatrixXf id=MatrixXf::Ones(_d->rows(),_d->cols());
	id*=val;
	*temp._d+=id;
	return temp;
}
template<class T>	
host_matrix<T>& host_matrix<T>::operator += (const host_matrix<T>& rhs){
	assert(_d->rows()==rhs._d->rows() && _d->cols()==rhs._d->cols());
	*_d+=*rhs._d;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(*this);
	*temp._d+=*rhs._d;
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator += (const typename host_matrix<T>::Transpose& rhs){
	assert(_d->rows()==rhs._m._d->cols()&&_d->cols()==rhs._m._d->rows());
	*_d+=rhs._m._d->transpose();
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (const typename host_matrix<T>::Transpose& rhs) const{
	host_matrix<T> temp(*this);
	*temp._d+=rhs._m._d->transpose();
	return temp;
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator -= (T val){
	MatrixXf id=MatrixXf::Ones(_d->rows(),_d->cols());
	id*=val;
	*_d-=id;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator - (T val) const{
	host_matrix<T> temp(*this);
	MatrixXf id=MatrixXf::Constant(_d->rows(),_d->cols(),val);
	*(temp._d)-=id;
	return temp;
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator -= (const host_matrix<T>& rhs){
	assert(_d->rows()==rhs._d->rows() && _d->cols()==rhs._d->cols());
	*_d-=*rhs._d;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator - (const host_matrix<T>& rhs){
	host_matrix<T> temp(*this);
	*temp._d-=*rhs._d;
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator -= (const typename host_matrix<T>::Transpose& rhs){
	assert(_d->rows()==rhs._m._d->cols()&&_d->cols()==rhs._m._d->rows());
	*_d-=rhs._m._d->transpose();
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator - (const typename host_matrix<T>::Transpose& rhs) const{
	host_matrix<T> temp(*this);
	*temp._d-=rhs._m._d->transpose();
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator /= (T val){
	assert(val!=(T)0.0);
	*_d*=1.0/(float)val;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator / (T val) const{
	assert(val!=(T)0.0);
	host_matrix<T> temp(*this);
	*(temp._d)*=1.0/(float)val;
	return temp;
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator *= (T val){
	*(_d)*=(float)val;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (T val) const{
	host_matrix<T> temp(*this);
	*(temp._d)*=(float)val;
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator *= (const host_matrix<T>& rhs){
	(*_d)*=*rhs._d;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(_d->rows(),rhs._d->cols());
	*temp._d=(*_d) * (*rhs._d);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator *= (const typename host_matrix::Transpose& rhs){
	*(_d)*=rhs._m._d->transpose();
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (const typename host_matrix::Transpose& rhs) const{
	host_matrix<T> temp(_d->rows(),rhs._m._d->rows());
	*temp._d=*_d*rhs._m._d->transpose();
	return temp;
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator &= (const host_matrix<T>& rhs){
	assert(_d->rows()==rhs._d->rows()&&_d->cols()==rhs._d->cols());
	*_d=_d->cwiseProduct(*rhs._d);
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator & (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(*this);
	*temp._d=_d->cwiseProduct(*rhs._d);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator = (const host_matrix<T>& rhs){
	*_d=*rhs._d;
	return *this;
}

template<class T>
typename host_matrix<T>::Transpose host_matrix<T>::operator ~ () const{
	return host_matrix<T>::Transpose(*this);
}

template<class T>
void host_matrix<T>::resize(size_t r,size_t c){
	if(r==_d->rows() && c==_d->cols())
		return;
	_d->resize(r,c);
}

template<class T>
void host_matrix<T>::resize(size_t r,size_t c,T val){
	this->resize(r,c);
	fillwith(val);	
}

template<class T>
void host_matrix<T>::print(int precision) const{
	cout<<fixed<<setprecision(precision);
	for(size_t y=0;y<_d->rows();++y){
		for(size_t x=0;x<_d->cols();++x)
			cout<<setw(precision+4)<<(*_d)(y,x);
		cout<<endl;
	}
}
	
template<class T>
void host_matrix<T>::fillwith(T val){
	_d->setConstant(_d->rows(),_d->cols(),(float)val);
}


#endif
