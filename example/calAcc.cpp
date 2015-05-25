#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <queue>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv){
	string prePath(argv[1]);
	string ansPath("/home/ahpan/Data/answer.txt");
	ifstream pre(prePath.c_str());
	ifstream ans(ansPath.c_str());
	if (!pre) cout <<"can't open pre file\n";
	if (!ans) cout << "can't open ans file\n";
	string a, p;
	float acc = 0;
	for (int i = 0; i <= 1040; i++){
		pre >> p;
		ans >> a;
		//cout <<p << " " << a << endl;
		if (p[p.find(",")+1] == a[a.find(",")+1])
			acc++;
	}
	cout << acc/1040 << endl;

}
