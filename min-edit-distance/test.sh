#!/usr/bin/env bash

# -1 is a dummy value
results=(-1 3 14 0 280 578 597 4953 8468 17313)

if [[ $# -ne 0 ]]; then
	tests=("$@")
else
	tests=(1 2 3 4 5 6 7 8 9)
fi

for i in "${tests[@]}"; do
	echo "    Exemplo $i    "
	failed=0

	readarray -t out_seq < <(./dist_seq "entradas/e$i.txt")
	if [[ "${out_seq[0]}" == "${results[$i]}" ]]; then
		echo -e "Seq.: ✔  ${out_seq[1]} ms"
	else
		echo 'Seq.: ✖'
		failed=1
	fi

	readarray -t out_par < <(./dist_par "entradas/e$i.txt")
	if [[ "${out_par[0]}" == "${results[$i]}" ]]; then
		echo -e "Par.: ✔  ${out_par[1]} ms"
	else
		echo 'Par.: ✖'
		failed=1
	fi

	if [[ $failed -eq 0 ]]; then
		echo 'Speedup: ' $(bc <<< "scale=3; ${out_seq[1]} / ${out_par[1]}")
	fi
done
