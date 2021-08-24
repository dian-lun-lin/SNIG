num_duplicates_vec=(1 10 100 250 500 1000)
#num_duplicates_vec=(1000)

for num_duplicates in ${num_duplicates_vec[@]}; do
  echo "==========Number of duplicates: $num_duplicates=========="
  ./exp.sh SNIGCapturerUpdate 1024 1920 1 1000 $num_duplicates 5
done
