#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <queue>
#include <unordered_map>

using namespace std;

typedef unordered_map<string, float> StringCntMap;
typedef pair<string, float> StringCntPair;

int main(int argc, char** argv){
	string wCntPath = "/home/ahpan/DNN_HW3/model/preprocess_2/vocab.txt";
	string classPath = "/home/larry/Documents/MLDS/DNN_HW3/model/classes.sorted.2.txt";
	ifstream ifs(wCntPath);
	if(!ifs.is_open()){
		cerr << "Fail to open file: " << wCntPath << " !\n";
	}
	ofstream ofs(classPath);
	if(!ofs.is_open()){
		cerr << "Fail to open file: " << classPath << " !\n";
	}

	size_t totalCnt = 0;
	size_t under5Cnt = 0;
	StringCntMap scm;
	queue<string> orderedVoc;
	char buf[50];

	while(ifs.getline(buf, sizeof(buf))){
		string temp(buf);
		stringstream ss(temp);
		size_t cnt;
		string voc;
		ss >> voc >> cnt;
		totalCnt += cnt;
		if(cnt >= 5){
			orderedVoc.push(voc);
			StringCntPair scp(voc, cnt);
			scm.insert(scp);
		}
		else{
			under5Cnt++;
		}
	}

	cout << "Finished load file: " << wCntPath << endl;
	cout << "Total word counts: " << totalCnt << endl;
	cout << "Word counts below 5: " << under5Cnt << endl;
	ifs.close();

	size_t quantum = totalCnt/200;
	size_t tempCnt = 0;
	size_t groupId = 0;
	//cout << "Group_" << groupId << endl;

	while(!orderedVoc.empty()){
		string word = orderedVoc.front();
		size_t wCnt = scm[word];
		tempCnt += wCnt;
		ofs << word << " " << groupId << endl;
		if(tempCnt > quantum){
			tempCnt = 0;
			groupId++;
			//cout << "Group_" << groupId << endl;
		}
		orderedVoc.pop();
	}

	cout << "Classified into " << groupId + 1 << " groups.\n";
	ofs.close();
}
