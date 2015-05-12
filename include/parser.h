#ifndef PARSER_H
#define PARSER_H
#include <map>
#include <vector>
#include <string>

using namespace std;

enum TYPE{
        MUST,
        MUSTNUM,
        OPTION,
        OPTIONNUM,
        NONE,
        ERROR,
        };

class PARSER{
    public:
    PARSER();
    PARSER(const PARSER& p);
    ~PARSER();
    void addMust(string name,bool isNum);
    void addOption(string name,bool isNum);
    
    bool read(int argc,char** argv);
    bool read(string spec);
    
    bool getString(string name,string& str);
    bool getNum(string name,int& num);
    bool getNum(string name,float& num);
    bool getNum(string name,size_t& num);

    void print()const;
    private:
    //helper functions
    TYPE find(string name);
    bool start(vector<string> v);

    void parseOptions(string str,vector<string>& vout);
    bool parseOneOption(string& str);

    bool isNum(string str);
    float str2Num(string str);
    string toUpperCase(string str);

    //data members

    vector<string> _mustList;

    vector<string> _mustHave;
    vector<string> _mustHaveNum;
    vector<string> _options;
    vector<string> _optionsNum;
    map<string,float> _numMap;
    map<string,string> _stringMap;
};

#endif
