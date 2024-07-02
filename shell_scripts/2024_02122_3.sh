
# 目的 : Eurlexの動作確認
# 動作確認のためepochsは2に設定
# データセットは大西くんからもらったXR-Transformerでのembedding3種類を試す
# bert, new_xlnet, roberta
cd ..

seeds=(100)

for seed in "${seeds[@]}";do
    #bert
    python run_classifier.py --data-file None --tr-split ../data/Eurlex/Eurlex_emb/bert_train.txt --te-split ../data/Eurlex/Eurlex_emb/bert_test.txt \
    --save ../data/model+results/Eurlex --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 2 --topk 50 --seed ${seed}

    #new_xlnet
    python run_classifier.py --data-file None --tr-split ../data/Eurlex/Eurlex_emb/new_xlnet_train.txt --te-split ../data/Eurlex/Eurlex_emb/new_xlnet_test.txt \
    --save ../data/model+results/Eurlex --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 2 --topk 50 --seed ${seed}

    #roberta
    python run_classifier.py --data-file None --tr-split ../data/Eurlex/Eurlex_emb/roberta_train.txt --te-split ../data/Eurlex/Eurlex_emb/roberta_test.txt \
    --save ../data/model+results/Eurlex --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 2 --topk 50 --seed ${seed}

done
