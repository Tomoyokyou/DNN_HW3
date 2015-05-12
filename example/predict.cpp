#include "parser.h"
#include "dnn.h"
#include "dataset.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cstdio>
//#include <device_matrix.h>
#include "host_matrix.h"
#include <map>

using namespace std;

//typedef device_matrix<float> mat;
typedef host_matrix<float> hmat;
typedef vector<size_t> seq;

void myUsage(){cerr<<"$cmd [testfile] [modelFile] [frameArk] [map] [outFile] \n options: --batchsize [] --dim [indim-hidnum1-hidnum2-outdim]"<<endl;}

bool parseName(string& str,string& name,char dilimeter){
	size_t begin=str.find_first_not_of(' '),end=str.find_first_of(dilimeter,begin);
	if(end!=string::npos)
		name=str.substr(begin,end-begin);
	else
		name=str.substr(begin,str.length());
	if(begin==str.find_last_not_of(' '))
			name=str[begin];
	begin=str.find_first_not_of(dilimeter,end);
	if(begin!=string::npos)
		str=str.substr(begin);
	else
		str=str.substr(str.find_first_not_of(' '));
	return true;
}
void readMap(string mapN,map<size_t,string>& myMap){
	FILE* fid;
	fid=fopen(mapN.c_str(),"r");
	char a[10],b[10],c[10],d[10];
	while(fscanf(fid,"%s %s %s %s \n",a,b,c,d)!=EOF){
		string phone(b),lab(c);
		myMap.insert(pair<size_t,string>(atoi(lab.c_str()),phone.c_str()));
	}
	fclose(fid);
	cout<<"total map size: "<<myMap.size()<<endl;
}
string getFrameName(string str){
	string n1,n2;
	parseName(str,n1,'_');
	parseName(str,n2,'_');
	return n1+n2;
}

void mapPhone(vector<string>& pho,const vector<seq>& res,map<size_t,string>& myMap){
	map<size_t,string>::iterator it;
	for(size_t t=0;t<res.size();++t){
		for(size_t k=0;k<res[t].size();++k){
			it=myMap.find(res[t].at(k));
			if(it==myMap.end()){cerr<<"ERROR: unrecognized label"<<endl;return;}
			pho.push_back(it->second);
		}
	}
}
void DNNpredict(Dataset& ds,DNN& nn,string frameName,string outNamei,size_t bsize,string mapF);
void write(ofstream& out,const vector<string>& result,const vector<string>& fname);



int main(int argc,char** argv){
	srand((unsigned)time(NULL));
	PARSER p;
	p.addMust("testFilename",false);
	p.addMust("modelFile",false);
	p.addMust("frameNameFile",false);
	p.addMust("mapFileName",false);
	p.addMust("outputName",false);
	p.addOption("--batchsize",true);
	string testF,outF,loadF,fark,mapF;
	size_t b_size;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("testfilename",testF);
	p.getString("modelFile",loadF);
	p.getString("frameNameFile",fark);
	p.getString("mapFileName",mapF);
	p.getString("outputName",outF);
	if(!p.getNum("--batchsize",b_size)){b_size=1000;}
	p.print();
	Dataset testData(testF.c_str());
	
		DNN nnload;
		if(nnload.load(loadF)){
			DNNpredict(testData,nnload,fark,outF,b_size,mapF);
		}
		else{	cerr<<"loading file:"<<loadF<<" failed! please check again..."<<endl;return 1;}
	
	cout<<"end of testing!";
	cout<<"\n result saved as :"<<outF<<endl;
	return 0;
}

void write(ofstream& out,const vector<string>& result,const vector<string>& fname){
	out<<"Id,Prediction"<<endl;
	for(size_t t=0;t<result.size();++t)
		out<<fname[t]<<","<<result[t]<<endl;
}
void DNNpredict(Dataset& ds,DNN& nn,string frameName,string outName,size_t bsize,string mapF){
	size_t dsize=ds.getDataNum();
	size_t fdim=ds.getFeatureDim();
	map<size_t,string> myMap;

	readMap(mapF,myMap);
	
	vector<string> fName;
	vector< vector<size_t> > result;
	
	ifstream infile(frameName.c_str());
	if(!infile){cerr<<"ERROR: inpufile failed!"<<endl;return;}
	
	for(string str;getline(infile,str);){
		string hold;
		parseName(str,hold,' ');
		fName.push_back(hold);
	}
	infile.close();
	size_t acc=0,t;
		hmat batch;
		hmat label_dummy;
		vector<size_t> res;

	for(t=0;t<dsize/bsize;++t){
		ds.getBatch(bsize,batch,label_dummy,false);		
		res.clear();
		nn.predict(res,batch);
		result.push_back(res);
	}
	
	size_t residual=dsize-t*bsize;
	ds.getBatch(residual,batch,label_dummy,false);
	res.clear();
	nn.predict(res,batch);
	result.push_back(res);
	
	vector<string> phores;
	mapPhone(phores,result,myMap);
	ofstream out(outName.c_str());
	if(!out){cerr<<"ERROR: failed opening output file"<<endl;return;}
	
	write(out,phores,fName);
	out.close();
}

