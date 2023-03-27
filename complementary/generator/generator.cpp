#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace std;


void initSats(int clauseSize, int maxBit, int satSize){
    srand(time(0));
    string outputFile = "rand"+to_string(maxBit)+"_" + to_string(satSize) + ".cnf";
    ofstream OutputFile(outputFile);
    OutputFile<<"p cnf "+ to_string(maxBit) +" " + to_string(satSize)<<endl;
    for(int i=0;i<satSize;i++){
        for(int ii=0;ii<clauseSize;ii++){
            int isPositive = rand()%2;
            int tmpNumber = rand()%maxBit;
            if(isPositive)  OutputFile<<to_string(tmpNumber+1) << " ";
            else OutputFile<<to_string((tmpNumber+1)*-1) << " ";
        }
        OutputFile<<"0"<< endl;
    }
    OutputFile.close();
}

int main(int argc, char** argv){
    if(argc!=4) throw std::invalid_argument( "invalid argument" );
    int clauseSize = atoi(argv[1]);
    int maxBit = atoi(argv[2]);
    int satSize = atoi(argv[3]);
    initSats(clauseSize,maxBit,satSize);
}