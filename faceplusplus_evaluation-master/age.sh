#!/bin/bash

# img_dir='/media/ray/Ray/database/age/Morph/crop/morhp_whole_256_st/split/0/1'
img_dir="${1}"
save_name="${2}_age.csv"

img_list=$(ls ${img_dir})

i=0
k=0

for j in $img_list
do
    if [[ $k -ge $i ]]
    then
        ((i++))
        ((k++))
        img="${img_dir}/${j}"
        age=$(curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" \
        -F "api_key=[Yours]" \
        -F "api_secret=[Yours]" \
        -F "image_file=@${img}" \
        -F "return_attributes=age" | jq -r '.faces[0].attributes.age.value')
        data="$k,${j},$age"
        echo ${data} >> ${save_name}
    else
        ((k++))
        continue
    fi
done

echo -ne '\007'
