tasks="0-15"
for dataset in  acsincome
do  
    for objective in extremile superquantile esrm erm
    do
        python scripts/lbfgs.py --dataset $dataset --objective $objective
        for optim in sgd dp_sgd 
        do
            for epsilon in 4
            do
                for batch_size in 512 
                do
                    python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --n_epochs 128 --batch_size $batch_size --dataset_length 4000 --epsilon $epsilon
                done
            done
        done
    done
done
