#ifndef MYNNGEN_H
#define MYNNGEN_H
#include <random>
#include <ctime>
#include <iostream>
#include <cassert>

using namespace std;

class myNnGen{
	public:
		myNnGen(){
			_dis = new normal_distribution<float>(0,1);
			_ggen.seed((unsigned)time(NULL));
			_m=0;_v=1;
		}
		myNnGen(float mean,float var){
			assert(var>0);
			_m=mean;_v=var;
			_dis = new normal_distribution<float>(mean,var);
			_ggen.seed((unsigned)time(NULL));
		}
		~myNnGen(){delete _dis;}
		void reset(float mean, float var){
			assert(var>0);
			if(mean==_m&&var==_v){}
			else{
			delete _dis;
			_dis=new normal_distribution<float>(mean,var);
			}
		}
		void seed(unsigned s){
			_ggen.seed(s);
		}
		float operator ()(){
			return _dis->operator()(_ggen);
		}
		void showParam(){
			cout<<"mean: "<<_m<<" var: "<<_v<<endl;
		}
	private:
		float _m;
		float _v;
		mt19937 _ggen;
		normal_distribution<float>* _dis;

};

#endif


