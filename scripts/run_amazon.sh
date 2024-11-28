tasks="0-15"
for dataset in amazon
do  
    for objective in erm esrm
    do
        python scripts/lbfgs.py --dataset $dataset --objective $objective
        for optim in dp_sgd
        do
            for epsilon in 2 4 8
            do
                for batch_size in  256 
                do
                    python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --n_epochs 128 --batch_size $batch_size --dataset_length 10000 --epsilon $epsilon
                done
            done
        done
    done
done