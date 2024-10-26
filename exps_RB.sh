#SSL
python main.py --exp RB --config codano_big --ntrain 40000

# Finetuning
python main.py --exp RB --config ft_codano_RB --ntrain 5 --epochs 50 --batch_size 1
python main.py --exp RB --config ft_codano_RB --ntrain 10 --epochs 50 --batch_size 2
python main.py --exp RB --config ft_codano_RB --ntrain 25 --epochs 50 --batch_size 5

python main.py --exp RB --config unet --ntrain 5 --epochs 50 --batch_size 1
python main.py --exp RB --config unet --ntrain 10 --epochs 50 --batch_size 2
python main.py --exp RB --config unet --ntrain 25 --epochs 50 --batch_size 5

python main.py --exp RB --config fno --ntrain 5 --epochs 50 --batch_size 1
python main.py --exp RB --config fno --ntrain 10 --epochs 50 --batch_size 2
python main.py --exp RB --config fno --ntrain 25 --epochs 50 --batch_size 5

python main.py --exp RB --config codano_RB --ntrain 5 --epochs 50 --batch_size 1
python main.py --exp RB --config codano_RB --ntrain 10 --epochs 50 --batch_size 2
python main.py --exp RB --config codano_RB --ntrain 25 --epochs 50 --batch_size 5