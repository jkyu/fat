#!/bin/bash
# Example energy and oscillator parser for CASSCF and CASCI calculations.
# The search terms will not be the same for different levels of theory or different electronic structure programs.

dir=$1 
nstates=$2
enefile=$3
oscfile=$4

if [ -f $enefile ]
then 
    rm -v $enefile
fi
if [ -f $oscfile ] 
then
    rm -v $oscfile
fi

for((j=0;j<=99;j++))
do
    i=$(printf "%04d" $j)
    grep -A $(($nstates + 1)) 'Total Energy (a.u.)' $dir/$i/tc.out | tail -n $(($nstates - 1)) | awk '{print $5}' >> $enefile
    grep -A $(($nstates + 2)) 'Singlet state electronic transitions' $dir/$i/tc.out | tail -n $(($nstates - 1)) | awk '{print $8}' >> $oscfile
done

