
# 目的 : A6000の初テスト Wiki10の最高性能を出したい
# データセットは大西くんからもらったXR-Transformerでのembedding4種類を試す
# bert, new_xlnet, roberta, projection
# 中間層の違いも見る
cd ..

seeds=(100 200 300)
h_sizes=(512 768 1024 2048)

for h_size in "${h_sizes[@]}";do
    for seed in "${seeds[@]}";do
        #bert
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/bert_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/bert_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
        --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed} --h-size ${h_size}

        #new_xlnet
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/new_xlnet_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/new_xlnet_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
        --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed} --h-size ${h_size}

        #roberta
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/roberta_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/roberta_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
        --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed} --h-size ${h_size}

        #projection
        python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/normal/projection_train.txt --te-split ../data/Wiki10-31K/normal/projection_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
        --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed} --h-size ${h_size}
    done
done


python run_classifier.py --data-file None --tr-split_a ../data/Wiki10-31K/Wiki10_emb/bert_train.txt --te-split_a ../data/Wiki10-31K/Wiki10_emb/bert_test.txt \
        --tr-split_b ../data/Wiki10-31K/Wiki10_emb/new_xlnet_train.txt --te-split_b ../data/Wiki10-31K/Wiki10_emb/new_xlnet_test.txt \
        --tr-split_c ../data/Wiki10-31K/Wiki10_emb/roberta_train.txt --te-split_c ../data/Wiki10-31K/Wiki10_emb/roberta_test.txt \
        --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
        --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 1 --topk 50 