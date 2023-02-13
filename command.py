'python main.py --batch_size 4 --img_size 28 --dataset shape --model_name NPN --device 5 --random_seed 7'
'python main.py --batch_size 4 --img_size 224 --dataset shape --model_name NPN --device 4 --random_seed 2022'
'python main.py --batch_size 4 --img_size 224 --dataset CCS --model_name VGG16 --device 4 --random_seed 2022'
'python main.py --batch_size 4 --img_size 224 --dataset CCS --model_name NPN224 --device 4 --random_seed 2022'
'python main.py --batch_size 4 --img_size 224 --dataset CCS --model_name ResNet18 --device 4 --random_seed 2022'

'CCS4____________________________'
#In CCS4, the lr of NPN need to setting 0.00001
'python main.py --batch_size 64 --class_num 4 --img_size 224 --dataset CCS --model_name NPNCCS --device 0 --lr 0.00001 --random_seed 2022'
'python main.py --batch_size 64 --class_num 4 --img_size 224 --dataset CCS --model_name AlexNet --device 0 --random_seed 2022'
'python main.py --batch_size 64 --class_num 4 --img_size 224 --dataset CCS --model_name ResNet18 --device 1 --random_seed 2022'
'python main.py --batch_size 64 --class_num 4 --img_size 224 --dataset CCS --model_name VGG10 --device 0 --random_seed 2022'

'CCS8____________________________'
'python main.py --batch_size 64 --class_num 8 --img_size 224 --dataset CCS --model_name NPNCCS --device 2 --lr 0.0001 --random_seed 2022'
'python main.py --batch_size 64 --class_num 8 --img_size 224 --dataset CCS --model_name VGG10 --device 0 --lr 0.0001 --random_seed 2022'
'python main.py --batch_size 64 --class_num 8 --img_size 224 --dataset CCS --model_name ResNet18 --device 4 --lr 0.0001 --random_seed 2022'
'python main.py --batch_size 64 --class_num 8 --img_size 224 --dataset CCS --model_name AlexNet --device 7 --lr 0.0001 --random_seed 2022'

