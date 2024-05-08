# Taking a Moment for Robustness
## CelebA
There should be a folder called 'CelebA' which contains a folder called 'data'. This data folder should contain list_eval_partition.csv, 'img_align_celeba.csv', and a folder called list_attr_celeba.csv. They can be downloaded on this kaggle link (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

Here's how you can run any of the methods (DRO, MRO, Adv-Moment) on CelebA.
-DRO: python run_expt.py *flags*
-MRO: python run_expt_centered_DRO.py *flags*
-Adv-moment: python run_expt_moment.py *flags* 


The flags here are the same as in https://github.com/kohpangwei/group_DRO/ but includes slightly more flags for MRO and Adv-moment. For instance, MRO needs to be centered around the minimum risk that can be achieved for each of the groups. This can be specified via '--group_min_loss' The paramter a_n for Adv-moment can be set via --test_l2_weight. 

Furthermore, one can run the experiment on a noisy dataset by setting the flag --flip_y_prob to be >0. 


Sample commands to run DRO, MRO, and Adv-moment
-DRO: python run_expt.py -s confounder -d CelebA --fraction 1.0 --flip_y_prob 0.5 --robust -t Blond_Hair -c Male --lr 0.00001 --batch_size 32 --show_progress --robust_step_size 0.000 --save_step 1 --save_last --save_best --root_dir celebA --weight_decay 0.1 --log_every 10 --model resnet50 --n_epochs 10 --reweight_groups --gamma 0.1 --generalization_adjustment 0 --log_dir /scratch/users/csj93/DRO_0_no_test_MW_05_ERM_no_robust

-MRO: python run_expt_centered_DRO.py -s confounder -d CelebA --fraction 1.0 --flip_y_prob 0.0 --group_min_loss '0.0002192510146414860, 0.000251323712291196, 0.0007384616183117030, 0.010234777815640000' --robust -t Blond_Hair -c Male  --lr 0.00001 --batch_size 32 --show_progress --robust_step_size 0.001 --save_step 1 --save_last --save_best --root_dir celebA --weight_decay 0.1 --log_every 10 --model resnet50 --n_epochs 10 --reweight_groups --gamma 0.1 --generalization_adjustment 0 --log_dir /scratch/users/csj93/CDRO_0_no_test_MW_final

-Adv-moment: python run_expt_moment.py -s confounder -d CelebA --fraction 1.0 --robust -t Blond_Hair -c Male --flip_y_prob 0.5 --lr 0.00001 --batch_size 32 --show_progress --robust_step_size 0.001 --save_step 1 --save_last --save_best --root_dir celebA --weight_decay 0.1 --log_every 10 --test_l2_weight 0.5 --model resnet50 --n_epochs 10 --reweight_groups --gamma 0.1 --generalization_adjustment 0 --log_dir /scratch/users/csj93/moment_05_no_test_MW_05
