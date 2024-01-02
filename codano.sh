# ssl on NS+ES dataset
python main.py --config codano_gno_NS_ES --ntrain 8000

# Now fine tuning on different number training data
python main.py --config ft_NSES_NSES_1 --ntrain 100
python main.py --config ft_NSES_NSES_5 --ntrain 100
python main.py --config ft_NSES_NSES_10 --ntrain 100

python main.py --config ft_NSES_NSES_1 --ntrain 500
python main.py --config ft_NSES_NSES_5 --ntrain 500
python main.py --config ft_NSES_NSES_10 --ntrain 500

python main.py --config ft_NSES_NSES_1 --ntrain 1000
python main.py --config ft_NSES_NSES_5 --ntrain 1000
python main.py --config ft_NSES_NSES_10 --ntrain 1000


## ssl on NS only dataset
python main.py --config codano_gno_NS --ntrain 8000

# fine tuning
python main.py --config ft_NS_NSES_1 --ntrain 100
python main.py --config ft_NS_NSES_5 --ntrain 100
python main.py --config ft_NS_NSES_10 --ntrain 100

python main.py --config ft_NS_NSES_1 --ntrain 500
python main.py --config ft_NS_NSES_5 --ntrain 500
python main.py --config ft_NS_NSES_10 --ntrain 500

python main.py --config ft_NS_NSES_1 --ntrain 1000
python main.py --config ft_NS_NSES_5 --ntrain 1000
python main.py --config ft_NS_NSES_10 --ntrain 1000