g++ -o generator generator.cpp
# generate ten random cnf files
for (( n=1; n<=10; n++ ))
do
    echo "$n round "
    ./generator 3 64 $((n * 1000))
done
