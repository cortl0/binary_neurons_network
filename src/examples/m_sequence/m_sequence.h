//*************************************************************//
//                                                             //
//   network of binary neurons                                 //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/network_of_binary_neurons_cpp   //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#ifndef MSEQUENCE_H
#define MSEQUENCE_H

class MSequence
{
    int triggers = 1;
public:
    int length;
    MSequence();
    MSequence(int triggersLength);
    bool Next();
    bool GetAt(int future);
    int GetRegisters();
    void SetTriggersLength(int triggersLength)
    {
        length = triggersLength;
    }
};

#endif // MSEQUENCE_H
