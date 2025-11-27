#!/bin/bash

cat in.nitro_temp > aux1000.txt
counter=1001
step=$(($2*10000))

for (( i=1; i<=7; i++ ))
do
	xcoord=$(head -n$(($i+2)) confs.xyz|tail -n1|awk '{print $2}')	
	ycoord=$(head -n$(($i+2)) confs.xyz|tail -n1|awk '{print $3}')	
	zcoord=$(head -n$(($i+2)) confs.xyz|tail -n1|awk '{print $4}')	

	xvel=$(head -n$(($i+2)) confs_vel.xyz|tail -n1|awk '{print $2}')	
	yvel=$(head -n$(($i+2)) confs_vel.xyz|tail -n1|awk '{print $3}')	
	zvel=$(head -n$(($i+2)) confs_vel.xyz|tail -n1|awk '{print $4}')	

	line=$(echo "set atom $counter vx $xvel vy $yvel vz $zvel")

	sed "s/ATOM$counter/$line/" "aux$(($counter-1)).txt" > "aux$counter.txt"
	counter=$(($counter + 1))
done
sed "s/INDEX/$step/" "aux$(($counter-1)).txt" > "aux$counter.txt"
mv aux$counter.txt in.nitro_main
rm aux*.txt
exit 0
