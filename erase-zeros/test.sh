#!/usr/bin/env bash

if [[ $# -ne 0 ]]; then
	tests=("$@")
else
	tests=(0 1 2 3 4 5 6)
fi

tmp=$(mktemp --tmpdir=.) || exit

for i in "${tests[@]}"; do
	echo "  Exemplo $i"

	if ./remove0_seq "es/entrada$i.txt" "$tmp"
	then
		if cmp -s "es/saida$i.txt" "$tmp"
		then
			echo 'Seq.: ✔'
		else
			echo 'Seq.: ✖'
		fi
	fi

	if mpirun -np 5 ./remove0_par "es/entrada$i.txt" "$tmp" # -maxtime 10
	then
		if cmp -s "es/saida$i.txt" "$tmp"
		then
			echo 'Par.: ✔'
		else
			echo 'Par.: ✖'
		fi
	fi

done

rm -f -- "$tmp"
