# SSL on NS+ES dataset
# python main.py --config codano_gno_NS_ES --ntrain 8000

####
# Finetuning on NS+ES dataset
####

# python main.py --config ft_NSES_NSES_1 --ntrain 5
# python main.py --config ft_NSES_NSES_5 --ntrain 5
# python main.py --config ft_NSES_NSES_10 --ntrain 5

# python main.py --config ft_NSES_NSES_1 --ntrain 10
# python main.py --config ft_NSES_NSES_5 --ntrain 10
# python main.py --config ft_NSES_NSES_10 --ntrain 10

# python main.py --config ft_NSES_NSES_1 --ntrain 25
# python main.py --config ft_NSES_NSES_5 --ntrain 25
# python main.py --config ft_NSES_NSES_10 --ntrain 25

# python main.py --config ft_NSES_NSES_1 --ntrain 50
# python main.py --config ft_NSES_NSES_5 --ntrain 50
# python main.py --config ft_NSES_NSES_10 --ntrain 50


# ## ssl on NS only dataset
# python main.py --config codano_gno_NS --ntrain 8000

# fine tuning on NS+ES dataset
python main.py --config ft_NS_NSES_1 --ntrain 5
python main.py --config ft_NS_NSES_5 --ntrain 5
python main.py --config ft_NS_NSES_10 --ntrain 5

python main.py --config ft_NS_NSES_1 --ntrain 10
python main.py --config ft_NS_NSES_5 --ntrain 10
python main.py --config ft_NS_NSES_10 --ntrain 10

python main.py --config ft_NS_NSES_1 --ntrain 25
python main.py --config ft_NS_NSES_5 --ntrain 25
python main.py --config ft_NS_NSES_10 --ntrain 25

python main.py --config ft_NS_NSES_1 --ntrain 50
python main.py --config ft_NS_NSES_5 --ntrain 50
python main.py --config ft_NS_NSES_10 --ntrain 50

