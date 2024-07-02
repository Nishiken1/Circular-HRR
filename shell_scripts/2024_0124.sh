
# 目的 : Wiki10新しいemb
# 動作確認のためepochsは15に設定
# データセットは大西くんからもらったXR-Transformerでのembedding1種類を試す
# emb
cd ..

seeds=(400 500 600)
h_sizes=(768 1024 2048 4096)

for seed in "${seeds[@]}";do
    for h_size in "${h_sizes[@]}";do
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/normal/xlnet_train.txt --te-split ../data/Wiki10-31K/normal/xlnet_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 8 \
        --test-batch-size 8 --th 0.3 --debug  --without-negative --no-grad --epochs 25 --topk 50 --seed ${seed} --h-size ${h_size}
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/normal/xlnet_train.txt --te-split ../data/Wiki10-31K/normal/xlnet_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 4 \
        --test-batch-size 4 --th 0.3 --debug  --without-negative --no-grad --epochs 25 --topk 50 --seed ${seed} --h-size ${h_size}
    done
done