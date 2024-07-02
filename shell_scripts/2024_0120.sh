
#目的 : A6000の初テスト Wiki10の最高性能を出したい
#データセットはWiki10のnormal, xlnet, glove, concatの4種類で行う
#batch-sizeを大きくしてみる
#次元数は2500までなら行けそう
cd ..

seeds=(100 200 300)
datasets=()

for seed in "${seeds[@]}";do
    #normal
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/train.txt --te-split data/Wiki10-31K/test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #xlnet
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/xlnet_train.txt --te-split data/Wiki10-31K/normal/xlnet_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #glove
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/glove_train.txt --te-split data/Wiki10-31K/normal/glove_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #concat(tfidf+xlnet)
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/concat_train.txt --te-split data/Wiki10-31K/normal/concat_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}
done

#バッチサイズの変更
for seed in "${seeds[@]}";do
    #xlnet batch 8
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/xlnet_train.txt --te-split data/Wiki10-31K/normal/xlnet_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 8 \
    --test-batch-size 8 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}
done

for seed in "${seeds[@]}";do
    #normal
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/train.txt --te-split data/Wiki10-31K/test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 32 \
    --test-batch-size 32 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #xlnet
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/xlnet_train.txt --te-split data/Wiki10-31K/normal/xlnet_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 32 \
    --test-batch-size 32 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #glove
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/glove_train.txt --te-split data/Wiki10-31K/normal/glove_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 32 \
    --test-batch-size 32 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}

    #concat(tfidf+xlnet)
    python run_classifier.py --data-file None --tr-split data/Wiki10-31K/normal/concat_train.txt --te-split data/Wiki10-31K/normal/concat_test.txt \
    --save data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 32 \
    --test-batch-size 32 --th 0.3 --debug  --without-negative --no-grad --epochs 1 --topk 50 --seed ${seed}
done