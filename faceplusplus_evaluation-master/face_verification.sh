#!/bin/bash

img_dir="${1}"
trg_dir="${2}"

save_name="${3}"

img_list=$(ls ${img_dir})

i=0
k=0

for j in $img_list
do
    if [[ $k -ge $i ]]
    then
        ((i++))
        ((k++))
        fake_img="${img_dir}/${j}"
        raw_img="${trg_dir}/${j}"

        confidence=$(curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/compare" \
        -F "api_key=[Yours]" \
        -F "api_secret=[Yours]" \
        -F "image_file1=@${fake_img}" \
        -F "image_file2=@${raw_img}" | jq -r '.confidence')

        data="$k,${j},$confidence"
        echo ${data} >> ${save_name}
    else
        ((k++))
        continue
    fi
done

echo -ne '\007'
