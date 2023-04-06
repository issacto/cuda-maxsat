#include <cstdlib>
#include <iostream>
#include <fstream>
#include <bitset>
#include <fstream>
#include <sstream>
using namespace std;
#define SATCLAUSESIZE 3

// optimum solution needs to be declared here before run time
int optimum[] ={-1,-2,-3,4,5,6,7,-8,9,-10,11,12,-13,14,-15,-16,17,18,-19,-20,21,-22,-23,-24,25,26,27,-28,-29,-30,31,-32,-33,-34,35,36,37,-38,39,40,41,-42,43,-44,45,46,47,48,49,50,51,-52,53,-54,-55,56,57,-58,-59,-60,-61,-62,-63,64};

short *readSatSets(string fileName, int *maxBit, int *satSize)
{
    string tempText;
    // read from the cnf file
    ifstream firstFileRead(fileName);
    ifstream secondFileRead(fileName);
    int tmpSatSize = 0;
    int tmpMaxBit = 0;
    while (getline(firstFileRead, tempText))
    {
        // process each line
        if (tempText[0] == 'p')
        {
            // if the line is about problem definition
            istringstream iss(tempText);
            string s;
            int tmpIndex = 0;
            while (getline(iss, s, ' '))
            {
                // process strings splitted by space in a line
                if (tmpIndex == 2)
                {
                    // get the number of variables
                    tmpMaxBit = atoi(s.c_str());
                }
                else if (tmpIndex == 3)
                {
                    // get the number of clauses
                    tmpSatSize = atoi(s.c_str());
                }
                if (!(tmpIndex >= 2 && atoi(s.c_str()) == 0))
                    tmpIndex += 1;
            }
            break;
        }
    }
    bool isCount = false;
    int index = 0;
    short *h_satSets = new short[tmpSatSize * SATCLAUSESIZE];
    // read the file to get the the literals and clauses of the problem sets
    while (getline(secondFileRead, tempText))
    {
         // process each line
        if (tempText[0] == 'p')
        {
            isCount = true;
        }
        else if (isCount && tempText[0] != 'c')
        {
            // iterate over all the literals in a line
            string tmpStr;
            for (int i = 0; i < tempText.size(); i++)
            {

                if (tempText[i] != ' ')
                {
                    tmpStr += tempText[i];
                }
                else
                {
                    if (tmpStr != "0" && !tmpStr.empty())
                    {
                        short tmpNumber = stoi(tmpStr);
                        h_satSets[index] = tmpNumber;
                        tmpStr = "";
                        index += 1;
                    }
                }
            }
        }
    }
    *maxBit = tmpMaxBit;
    *satSize = tmpSatSize;
    firstFileRead.close();
    secondFileRead.close();
    return h_satSets;
}

int main(int argc, char **argv)
{
    if (argc != 2)
        throw std::invalid_argument("invalid argument");
    string fileName = argv[1];
    int maxBit = 0;
    int satSize = 0;
    short *set = readSatSets(fileName, &maxBit, &satSize);
    int maxIter = satSize * SATCLAUSESIZE;
    int ans = 0;
    for (int i = 0; i < satSize; i++)
    {
        for (int ii = 0; ii < SATCLAUSESIZE; ii++)
        {
            if (set[i * SATCLAUSESIZE + ii] < 0)
            {
                // if literal is negative
                if (optimum[abs(set[i * SATCLAUSESIZE + ii]) - 1] < 0)
                {
                    // if chromsome's variable is negative
                    ans = ans + 1;
                    break;
                }
            }
            else
            {
                // if literal is positive
                if (optimum[abs(set[i * SATCLAUSESIZE + ii]) - 1] > 0)
                {
                    // if chromsome's variable is positive
                    ans = ans + 1;
                    break;
                }
            }
        }
    }
    cout << "Answer: " << endl;
    cout << ans << endl;
}