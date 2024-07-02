
#目的 : A6000の初テスト Wiki10の最高性能を出したい
#データセットはWconcatの1種類で行う
#batch-sizeを大きくしてみる
#次元数は2500までなら行けそう
cd ..

seeds=(100 200 300)
datasets=()

for seed in "${seeds[@]}";do
    #concat(tfidf+xlnet)
    python run_classifier.py --data-file None --tr-split ../data/Wiki10-31K/normal/concat_train.txt --te-split ../data/Wiki10-31K/normal/concat_test.txt \
    --save ../data/model+results/Wiki10-31K --name neurips-2021-results --batch-size 16 \
    --test-batch-size 16 --th 0.3 --debug  --without-negative --no-grad --epochs 15 --topk 50 --seed ${seed}
done
