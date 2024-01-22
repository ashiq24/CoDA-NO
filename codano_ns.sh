
# using pre-trained model in NS+ES dataset
python main.py --config ft_NSES_NS_1 --ntrain 100
python main.py --config ft_NSES_NS_5 --ntrain 100
python main.py --config ft_NSES_NS_10 --ntrain 100

python main.py --config ft_NSES_NS_1 --ntrain 250
python main.py --config ft_NSES_NS_5 --ntrain 250
python main.py --config ft_NSES_NS_10 --ntrain 250

python main.py --config ft_NSES_NS_1 --ntrain 500
python main.py --config ft_NSES_NS_5 --ntrain 500
python main.py --config ft_NSES_NS_10 --ntrain 500


# using pre-trained model on NS only dataset

python main.py --config ft_NS_NS_1 --ntrain 100
python main.py --config ft_NS_NS_5 --ntrain 100
python main.py --config ft_NS_NS_10 --ntrain 100

python main.py --config ft_NS_NS_1 --ntrain 250
python main.py --config ft_NS_NS_5 --ntrain 250
python main.py --config ft_NS_NS_10 --ntrain 250

python main.py --config ft_NS_NS_1 --ntrain 500
python main.py --config ft_NS_NS_5 --ntrain 500
python main.py --config ft_NS_NS_10 --ntrain 500