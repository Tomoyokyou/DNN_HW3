#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include "parser.h"


using namespace std;

PARSER::PARSER(){
        }
PARSER::PARSER(const PARSER& p){
        _mustList=p._mustList;
    _mustHave = p._mustHave;
    _mustHaveNum = p._mustHaveNum;
    _options = p._options;
    _optionsNum = p._optionsNum;
    _numMap = p._numMap;
    _stringMap = p._stringMap;

}
PARSER::~PARSER(){}

void PARSER::addMust(string name,bool isNum){
        if(!name.empty()){
        if(!isNum){
                _mustHave.push_back(toUpperCase(name));
        }
        else{
                _mustHaveNum.push_back(toUpperCase(name));
        }
        _mustList.push_back(toUpperCase(name));
        }
}

void PARSER::addOption(string name,bool isNum){
    if(!name.empty()){
        if(!isNum){
            _options.push_back(toUpperCase(name));
        }
        else{
            _optionsNum.push_back(toUpperCase(name));
        }
    }
}

bool PARSER::read(int argc,char** argv){
        vector<string>v;
        for(int i=1;i<argc;++i){
                string tmp(argv[i]);
                if(!tmp.empty())
                v.push_back(tmp);
        }
        if(_mustHave.size()==0&&_mustHaveNum.size()==0&&_options.size()==0&&_optionsNum.size()==0){
            cerr<<"No target to parse!"<<endl;
            return false;
        }
        if(!start(v)){
            cerr<<"Missing arguments, please double check!"<<endl;
            return false;
        }
        return true;
}


bool PARSER::read(string spec){
        vector<string> v;
        parseOptions(spec,v);
        if(_mustHave.size()==0&&_mustHaveNum.size()==0&&_options.size()==0&&_optionsNum.size()==0){
            cerr<<"No target to parse!"<<endl;
            return false;
        }
        if(!start(v)){
            cerr<<"Missing or wrong arguments, please double check!"<<endl;
            return false;
        }
        return true;
}

bool PARSER::start(vector<string> vx){
        vector<string> v;
        if(vx.size()<_mustList.size()){
           // cout<<"vx "<<vx.size()<<" "<<v.size()<<endl;
            return false;
        }
        else{
                for(size_t t=_mustList.size();t<vx.size();++t)
                        v.push_back(vx[t]);
        for(size_t t=0;t<_mustList.size();++t){
           TYPE type;
           type=find(_mustList[t]);
           switch(type){
                   case MUST:
                    _stringMap.insert(pair<string,string>(toUpperCase(_mustList[t]),vx[t]));
                   break;
                   case MUSTNUM:
						if(isNum(vx[t])){
                        _numMap.insert(pair<string,float>(toUpperCase(_mustList[t]),str2Num(vx[t])));
						}
						else
								return false;
                   break;
            default:
                return false;
            break;
           }
        }
        for(size_t t=0;t<v.size();++t){
            TYPE type;
            type=find(toUpperCase(v[t]));
            bool end= ( t+1==v.size() ) ? true : false;
            switch(type){
                case OPTION:
                    if(end){return false;}
                    else
                            _stringMap.insert(pair<string,string>(toUpperCase(v[t]),v[t+1]));
                break;
                case OPTIONNUM:
                    if(end){return false;}
                    if(isNum(v[t+1])){
                        _numMap.insert(pair<string,float>(toUpperCase(v[t]),str2Num(v[t+1])));
                    }
                    else
                            return false;
                break;
                case NONE:
                break;
                defaule:
                    return false;
                break;
            }
        }
            return true;
        }
}

bool PARSER::getString(string name,string& str){
    map<string,string>::iterator it;
    it=_stringMap.find(toUpperCase(name));
    if(it==_stringMap.end())
            return false;
    else{
            str=it->second;
            return true;
    }
}

bool PARSER::getNum(string name,int& num){
    map<string,float>::iterator it;
    it=_numMap.find(toUpperCase(name));
    if(it==_numMap.end())
            return false;
    else{
            num=(int)it->second;
            return true;
    }
}
bool PARSER::getNum(string name,float& num){
    map<string,float>::iterator it;
    it=_numMap.find(toUpperCase(name));
    if(it==_numMap.end())
            return false;
    else{
            num=it->second;
            return true;
    }
}
bool PARSER::getNum(string name,size_t& num){
    map<string,float>::iterator it;
    it=_numMap.find(toUpperCase(name));
    if(it==_numMap.end())
            return false;
    else{
            num=(size_t)it->second;
            return true;
    }
}

void PARSER::print()const{
    int count=0;
    cout<<"\nMust have:\n";
    for(size_t t=0;t<_mustHave.size();++t){
            count++;
            cout<<" "<<_mustHave[t];
            if(count%5==0)
                    cout<<endl;
    }
    for(size_t t=0;t<_mustHaveNum.size();++t){
            count++;
            cout<<" "<<_mustHaveNum[t];
            if(count%5==0)
                    cout<<endl;
    }
    count=0;
    cout<<"\nOptions:\n";
    for(size_t t=0;t<_options.size();++t){
            count++;
            cout<<" "<<_options[t];
            if(count%5==0)
                    cout<<endl;
    }
    for(size_t t=0;t<_optionsNum.size();++t){
            count++;
            cout<<" "<<_optionsNum[t];
            if(count%5==0)
                    cout<<endl;
    }
    cout<<"\nMatch commands:\n";
    map<string,string>::const_iterator it1;
    map<string,float>::const_iterator it2;
    for(it1=_stringMap.begin() ; it1!=_stringMap.end() ; ++it1)
            cout<<"  "<<it1->first<<":["<<it1->second<<"]"<<endl;
    for(it2=_numMap.begin() ; it2!=_numMap.end() ; ++it2)
            cout<<"  "<<it2->first<<":["<<it2->second<<"]"<<endl;
    cout<<endl;
}


//helper functions

TYPE PARSER::find(string name){
        TYPE tmp = NONE;
for(size_t t=0;t<_mustHave.size();++t)
        if(_mustHave[t]==name){
                tmp = MUST;
                }
for(size_t t=0;t<_mustHaveNum.size();++t)
        if(_mustHaveNum[t]==name){
            tmp = MUSTNUM;
        }
for(size_t t=0;t<_options.size();++t)
        if(_options[t]==name){
               tmp = OPTION;
        }
for(size_t t=0;t<_optionsNum.size();++t)
        if(_optionsNum[t]==name){
            tmp = OPTIONNUM;
        }
return tmp;
}

void PARSER::parseOptions(string str,vector<string>& vout){
    size_t begin,end,check;
    string hold;
    begin=str.find_first_not_of(' ');
    while(begin!=string::npos){
        end=str.find_first_of(' ',begin);
        if(end==string::npos){
            hold=str.substr(begin);
            begin=string::npos;
        }
        else{
                hold=str.substr(begin,end-begin);
                begin=str.find_first_not_of(' ',end);
        }
        if(!hold.empty())
                vout.push_back(hold);
    }

}

bool PARSER::parseOneOption(string& str){
    if(str.empty())
            return false;
            size_t begin,end;
            string tmp;
            begin=str.find_first_not_of(' ');
            end=str.find_first_of(' ',begin);
            size_t check=str.find_last_not_of(' ');
            if(end==string::npos||check==end){
                end=str.find_last_not_of(' ');
                str=str.substr(begin,end-begin);
                    return true;
            }
            else
                    return false;
                
}

bool PARSER::isNum(string str){
    bool dot=false;
    if(str.empty())
                return false;
    if((int)str[0]>47&&(int)str[0]<58){
    for(size_t t=1;t<str.size();++t){
        if((int)str[t]>47&&(int)str[t]<58){}
        else if((int)str[t]==46){
                    if(!dot)
                            dot=true;
                    else
                            return false;
                }
        else
                return false;
    }
        return true;
    }
    else
            return false;
}
float PARSER::str2Num(string str){
    return atof(str.c_str());
}
string PARSER::toUpperCase(string str){
    string temp=str;
    for(size_t t=0;t<str.size();++t){
                if((int)str[t]>96&&(int)str[t]<123){
                    int x=(int)str[t]-32;
                    temp[t]=x;
                }
                else
                        temp[t]=str[t];
    }
    return temp;
}
