
# Fine tuning Model( Pretrained with NS+EW Datset) on NS only dataset.

python main.py --config ft_NSES_NS_1 --ntrain 5
python main.py --config ft_NSES_NS_5 --ntrain 5
python main.py --config ft_NSES_NS_10 --ntrain 5

python main.py --config ft_NSES_NS_1 --ntrain 10
python main.py --config ft_NSES_NS_5 --ntrain 10
python main.py --config ft_NSES_NS_10 --ntrain 10

python main.py --config ft_NSES_NS_1 --ntrain 25
python main.py --config ft_NSES_NS_5 --ntrain 25
python main.py --config ft_NSES_NS_10 --ntrain 25

python main.py --config ft_NSES_NS_1 --ntrain 50
python main.py --config ft_NSES_NS_5 --ntrain 50
python main.py --config ft_NSES_NS_10 --ntrain 50


# Fine tuning Model( Pretrained with NS Datset) on NS only dataset.
python main.py --config ft_NS_NS_1 --ntrain 5
python main.py --config ft_NS_NS_5 --ntrain 5
python main.py --config ft_NS_NS_10 --ntrain 5

python main.py --config ft_NS_NS_1 --ntrain 10
python main.py --config ft_NS_NS_5 --ntrain 10
python main.py --config ft_NS_NS_10 --ntrain 10

python main.py --config ft_NS_NS_1 --ntrain 25
python main.py --config ft_NS_NS_5 --ntrain 25
python main.py --config ft_NS_NS_10 --ntrain 25

python main.py --config ft_NS_NS_1 --ntrain 50
python main.py --config ft_NS_NS_5 --ntrain 50
python main.py --config ft_NS_NS_10 --ntrain 50