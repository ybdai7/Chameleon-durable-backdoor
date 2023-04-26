##### Semantic Car with Stripe BG
##### Neurotoxin
python training.py --params utils/params_car_stripe_bg.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 100
##### Baseline
python training.py --params utils/params_car_stripe_bg.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 100
#####Chameleon
python training.py --params utils/params_car_stripe_bg.yaml --GPU_id "0" --size_of_secret_dataset 100
#####Anticipate
python training.py --params utils/params_car_stripe_bg.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1

##### Green Cars
##### Neurotoxin
python training.py --params utils/params_green_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 100 --persistence_diff 0.001
##### Baseline
python training.py --params utils/params_green_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 100 --persistence_diff 0.001
#####Chameleon
python training.py --params utils/params_green_car.yaml --GPU_id "0" --size_of_secret_dataset 100 --persistence_diff 0.001
#####Anticipate
python training.py --params utils/params_green_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1 --persistence_diff 0.001

##### Car with Racing Stripes
##### Neurotoxin
python training.py --params utils/params_racing_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 100
##### Baseline
python training.py --params utils/params_racing_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 100
#####Chameleon
python training.py --params utils/params_racing_car.yaml --GPU_id "0" --size_of_secret_dataset 100
#####Anticipate
python training.py --params utils/params_racing_car.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1


##### Pixel Pattern CIFAR10
##### Chameleon
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 1 --size_of_secret_dataset 25 --persistence_diff 0.03 --contrastive_loss_scale_weight 6.0
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.1 --size_of_secret_dataset 25 --persistence_diff 0.03 --contrastive_loss_scale_weight 6.0
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.5 --size_of_secret_dataset 25 --persistence_diff 0.03 --contrastive_loss_scale_weight 9.0
##### Neurotoxin
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.5 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
##### Baseline
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.5 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
##### Anticipate
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03 --anticipate 1
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03 --anticipate 1
python training.py --params utils/params_pixel_trigger_cifar10.yaml --GPU_id "0" --pattern_type 2 --pattern_diff 0.5 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03 --anticipate 1

##### Pixel Pattern EMNIST
##### Chameleon
python training.py --params utils/params_pixel_trigger_emnist.yaml --GPU_id "0" --pattern_type 1 --size_of_secret_dataset 25 --persistence_diff 0.03 --contrastive_loss_scale_weight 3.0
##### Baseline
python training.py --params utils/params_pixel_trigger_emnist.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
##### Neurotoxin
python training.py --params utils/params_pixel_trigger_emnist.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03
##### Anticipate
python training.py --params utils/params_pixel_trigger_emnist.yaml --GPU_id "0" --pattern_type 1 --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --persistence_diff 0.03 --anticipate 1

##### Edge-case Cifar10
##### Chameleon
python training.py --params utils/params_edge_case_cifar10.yaml --GPU_id "0" --size_of_secret_dataset 25 --edge_case 1
##### Baseline
python training.py --params utils/params_edge_case_cifar10.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --edge_case 1
##### Baseline
python training.py --params utils/params_edge_case_cifar10.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --edge_case 1
##### Anticipate
python training.py --params utils/params_edge_case_cifar10.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1 --edge_case 1


##### Semantic Car with Stripe BG cifar100
##### Neurotoxin
python training.py --params utils/params_car_stripe_bg_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 25
##### Baseline
python training.py --params utils/params_car_stripe_bg_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25
##### Chameleon
python training.py --params utils/params_car_stripe_bg_cifar100.yaml --GPU_id "0" --size_of_secret_dataset 100
##### Anticipate
python training.py --params utils/params_car_stripe_bg_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1

##### Black bicycle cifar100
##### Neurotoxin
python training.py --params utils/params_black_bicycle_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 0.99 --lr_gamma 0.01 --size_of_secret_dataset 100
##### Baseline
python training.py --params utils/params_black_bicycle_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 5 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 100
##### Chameleon
python training.py --params utils/params_black_bicycle_cifar100.yaml --GPU_id "0" --size_of_secret_dataset 100
##### Anticipate
python training.py --params utils/params_black_bicycle_cifar100.yaml --GPU_id "0" --is_frozen_params 0 --retrain_poison_contrastive 0 --retrain_poison 3 --poison_lr 0.1 --gradmask_ratio 1 --lr_gamma 0.01 --size_of_secret_dataset 25 --anticipate 1