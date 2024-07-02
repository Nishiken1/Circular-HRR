
# 目的 : A6000の初テスト Wiki10の最高性能を出したい
# データセットは大西くんからもらったXR-Transformerでのembedding4種類を試す
# bert, new_xlnet, roberta, projection
cd ..

seeds=(100 200 300)

for seed in "${seeds[@]}";do
    #bert
    python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/bert_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/bert_test.txt \
    --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #new_xlnet
    python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/new_xlnet_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/new_xlnet_test.txt \
    --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #roberta
    python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/Wiki10_emb/roberta_train.txt --te-split ../data/Wiki10-31K/Wiki10_emb/roberta_test.txt \
    --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #projection
    python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/normal/projection_train.txt --te-split ../data/Wiki10-31K/normal/projection_test.txt \
    --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}
done
