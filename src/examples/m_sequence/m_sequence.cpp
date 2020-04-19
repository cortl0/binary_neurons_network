//*************************************************************//
//                                                             //
//   binary neurons network                                    //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/binary_neurons_network          //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#include "m_sequence.h"

MSequence::MSequence()
{
    length = 3;
}

MSequence::MSequence(int triggersLength)
{
    if((triggersLength < 2)||(triggersLength > 31))
        throw ("error");
    length = triggersLength;
}

bool MSequence::Next()
{
    int returnValue = triggers & 1;
    triggers >>= 1;
    triggers |= ((triggers & 1) ^ returnValue) << (length-1);
    return returnValue;
}

bool MSequence::GetAt(int future)
{
    return ((triggers >> future) & 1);
}

int MSequence::GetRegisters()
{
    return triggers;
}
